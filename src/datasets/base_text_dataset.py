import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.utils.io_utils import ROOT_PATH, read_json, write_json

logger = logging.getLogger(__name__)

# number of tokens collected in RAM before they are flushed to disk
WRITE_BUFFER_SIZE = 1_000_000


class BaseTextDataset(Dataset):
    """
    Base class for language modeling datasets.

    The data goes through three stages:
        1. __init__: download raw data if it is missing (_download).
        2. prepare(tokenizer): tokenize all raw texts once and save them
           to disk as one flat array of token ids (raw numpy binary
           file + JSON with metadata).
        3. __getitem__: read fixed-size windows from the memory-mapped
           array, so only the requested part of the data is loaded.

    Nested classes define _is_downloaded, _download, and iter_texts.
    Until prepare() is called, the dataset has zero length.
    """

    def __init__(self, data_dir, part, block_size):
        """
        Args:
            data_dir (str): directory of the dataset. Relative paths are
                resolved against the project root.
            part (str): partition name (train, val, test, ...).
            block_size (int): number of tokens in one training example.
        """
        self.data_dir = ROOT_PATH / data_dir
        self.part = part
        self.block_size = block_size

        self.tokens_path = self.data_dir / "tokens" / f"{part}.bin"
        self.meta_path = self.tokens_path.with_suffix(".json")
        self._meta = None
        self._tokens = None  # memmap is opened lazily in each process

        if not self._is_downloaded():
            logger.info(f"Downloading {type(self).__name__} ({part})...")
            self._download()

    # ---------- to be defined in nested classes ----------

    def _is_downloaded(self):
        """
        Returns:
            is_downloaded (bool): True if raw data is ready to be read.
        """
        raise NotImplementedError()

    def _download(self):
        """
        Download raw data into self.data_dir.
        """
        raise NotImplementedError()

    def iter_texts(self):
        """
        Iterate over raw documents of the partition. EOS is added after
        each document during tokenization.

        Yields:
            text (str): one document.
        """
        raise NotImplementedError()

    # ---------- tokenization ----------

    def prepare(self, tokenizer, force=False):
        """
        Tokenize the partition if there is no up-to-date cache on disk.

        Args:
            tokenizer (BaseTokenizer): fitted tokenizer.
            force (bool): if True, retokenize even if the cache exists.
                Must be set if the tokenizer was refitted, since the
                cache cannot detect a change of merges by itself.
        """
        meta = read_json(str(self.meta_path)) if self.meta_path.exists() else None
        if force or meta is None or meta["vocab_size"] != tokenizer.vocab_size:
            meta = self._tokenize(tokenizer)
        self._meta = meta
        self._tokens = None

    def _tokenize(self, tokenizer):
        """
        Tokenize all documents and write them to disk in chunks.
        The data is written into a temporary file first, so an
        interrupted run does not leave a broken cache.

        Args:
            tokenizer (BaseTokenizer): fitted tokenizer.
        Returns:
            meta (dict): metadata of the saved array.
        """
        dtype = "uint16" if tokenizer.vocab_size <= 2**16 else "uint32"
        self.tokens_path.parent.mkdir(exist_ok=True, parents=True)
        tmp_path = self.tokens_path.with_suffix(".tmp")

        n_tokens, n_docs = 0, 0
        buffer = []
        with tmp_path.open("wb") as f:
            texts = tqdm(self.iter_texts(), desc=f"Tokenizing {self.part}", unit="doc")
            for text in texts:
                buffer.extend(tokenizer.encode(text, add_eos=True))
                n_docs += 1
                if len(buffer) >= WRITE_BUFFER_SIZE:
                    np.asarray(buffer, dtype=dtype).tofile(f)
                    n_tokens += len(buffer)
                    buffer = []
            np.asarray(buffer, dtype=dtype).tofile(f)
            n_tokens += len(buffer)
        tmp_path.replace(self.tokens_path)

        meta = {
            "n_tokens": n_tokens,
            "n_docs": n_docs,
            "dtype": dtype,
            "vocab_size": tokenizer.vocab_size,
        }
        write_json(meta, str(self.meta_path))
        logger.info(f"{self.part}: {n_docs} documents, {n_tokens} tokens")
        return meta

    # ---------- loading ----------

    @property
    def tokens(self):
        assert self._meta is not None, "Call prepare(tokenizer) first"
        if self._tokens is None:
            self._tokens = np.memmap(
                self.tokens_path,
                dtype=self._meta["dtype"],
                mode="r",
                shape=(self._meta["n_tokens"],),
            )
        return self._tokens

    def __len__(self):
        """
        Number of non-overlapping windows. Each window takes
        block_size + 1 tokens, since labels are shifted by one.
        """
        if self._meta is None:
            return 0
        return (self._meta["n_tokens"] - 1) // self.block_size

    def __getitem__(self, ind):
        """
        Args:
            ind (int): index of the window.
        Returns:
            instance_data (dict): dict with input_ids and labels, both
                of length block_size. labels are input_ids shifted by one.
        """
        start = ind * self.block_size
        window = self.tokens[start : start + self.block_size + 1].astype(np.int64)
        window = torch.from_numpy(window)
        return {"input_ids": window[:-1], "labels": window[1:]}

    def __getstate__(self):
        # do not pickle the memmap into dataloader workers,
        # each worker opens its own
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state
