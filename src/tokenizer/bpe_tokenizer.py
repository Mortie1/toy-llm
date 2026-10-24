import re
import unicodedata
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from src.tokenizer.base_tokenizer import BaseTokenizer
from src.utils.io_utils import ROOT_PATH, read_json, write_json

# GPT-2-like pre-tokenization written with the stdlib `re`:
# words (with an optional leading space), numbers, punctuation runs,
# and whitespace. [^\W\d_] is "any unicode letter".
# Merges never cross the boundaries of these chunks.
PRETOKENIZE_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
WORD_PATTERN = re.compile(r" ?[^\W\d_]+")

BOS, EOS, PAD = "<|bos|>", "<|eos|>", "<|pad|>"
# Markers that make word normalization reversible:
# NOSP - the next word had no leading space in the original text;
# CAP - the next word was capitalized in the original text.
NOSP, CAP = "<|nosp|>", "<|cap|>"
DEFAULT_SPECIAL_TOKENS = (BOS, EOS, PAD, NOSP, CAP)

N_BYTES = 256


def _merge(ids, pair, new_id):
    """
    Replace every occurrence of the pair in ids with new_id.

    Args:
        ids (list[int]): token ids.
        pair (tuple[int, int]): pair of ids to merge.
        new_id (int): id of the merged token.
    Returns:
        new_ids (list[int]): ids after the merge.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def _normalize_chunk(chunk):
    """
    Args:
        chunk (str): pre-tokenized chunk.
    Returns:
        markers (list[str]): markers to emit before the chunk.
        chunk (str): normalized chunk.
    """
    if not WORD_PATTERN.fullmatch(chunk):
        return [], chunk

    markers = []
    word = chunk.lstrip(" ")
    if len(word) == len(chunk):
        markers.append(NOSP)

    first, rest = word[0], word[1:]
    # the check on first.lower().upper() skips letters like "İ" whose
    # lowercase form does not map back to the same character
    if (
        first.isupper()
        and rest == rest.lower()
        and len(first.lower()) == 1
        and first.lower().upper() == first
    ):
        markers.append(CAP)
        word = first.lower() + rest

    return markers, " " + word


class BPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer.

    Training is split into two phases so the corpus never has to fit in RAM:
        1. update(texts) - may be called many times on chunks of data.
           Only frequencies of unique pre-tokenized chunks are accumulated.
        2. fit(vocab_size) - learns merges from all accumulated frequencies.
    More data can be added with update() afterwards, and fit() relearns
    merges from scratch on the updated statistics.

    Words are normalized to " lowercase" form, with <|nosp|> and <|cap|>
    markers keeping the information needed to restore the original text.
    """

    def __init__(self, path=None, special_tokens=DEFAULT_SPECIAL_TOKENS):
        """
        Args:
            path (str | Path | None): path to the tokenizer file. If the
                file exists, the tokenizer is loaded from it. Otherwise,
                the tokenizer stays unfitted and save() writes to this
                path by default. Relative paths are resolved against
                the project root.
            special_tokens (Sequence[str]): special tokens. Their ids go
                after the byte and merge tokens. Must include BOS, EOS,
                PAD, NOSP, and CAP.
        """
        self.path = None if path is None else self._resolve_path(path)
        self.special_tokens = list(special_tokens)
        self.merges = {}  # (id, id) -> new id, in the order of learning
        self._chunk_counts = Counter()  # normalized chunk -> frequency
        self.is_fitted = False
        self._build_vocab()

        if self.path is not None and self.path.exists():
            self._load_state(read_json(str(self.path)))

    # ---------- training ----------

    def update(self, texts):
        """
        Accumulate chunk frequencies from a piece of data.

        Args:
            texts (str | Iterable[str]): text or texts to count.
        """
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            for chunk in self._pretokenize(text):
                self._chunk_counts[_normalize_chunk(chunk)[1]] += 1

    def prune(self, min_count):
        """
        Drop rare chunks from the accumulated statistics to save memory.

        Args:
            min_count (int): chunks with frequency below it are removed.
        """
        self._chunk_counts = Counter(
            {chunk: c for chunk, c in self._chunk_counts.items() if c >= min_count}
        )

    def fit(self, vocab_size, verbose=True):
        """
        Learn merges from all accumulated chunk frequencies.
        Previously learned merges are discarded.

        Args:
            vocab_size (int): target vocabulary size, including special
                tokens. The result may be smaller if the data runs out
                of pairs to merge.
            verbose (bool): if True, show a progress bar.
        """
        n_merges = vocab_size - N_BYTES - len(self.special_tokens)
        assert n_merges >= 0, (
            f"vocab_size must be at least {N_BYTES + len(self.special_tokens)}"
        )
        assert len(self._chunk_counts) > 0, "No data, call update() first"

        words = [list(chunk.encode("utf-8")) for chunk in self._chunk_counts]
        freqs = list(self._chunk_counts.values())

        self.merges = {}
        for i in tqdm(range(n_merges), desc="BPE merges", disable=not verbose):
            stats = self._get_pair_stats(words, freqs)
            if not stats:
                break
            # ties are broken by the pair itself to keep fit deterministic
            pair = max(stats, key=lambda p: (stats[p], p))
            new_id = N_BYTES + i
            words = [_merge(w, pair, new_id) if len(w) > 1 else w for w in words]
            self.merges[pair] = new_id

        self.is_fitted = True
        self._build_vocab()

    @staticmethod
    def _get_pair_stats(words, freqs):
        """
        Count adjacent pairs weighted by chunk frequency.

        Args:
            words (list[list[int]]): chunks as token ids.
            freqs (list[int]): frequency of each chunk.
        Returns:
            stats (Counter): pair -> weighted count.
        """
        stats = Counter()
        for word, freq in zip(words, freqs):
            for pair in zip(word, word[1:]):
                stats[pair] += freq
        return stats

    def _build_vocab(self):
        """
        Rebuild id -> bytes vocabulary and special token ids from merges.
        """
        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        first_special_id = N_BYTES + len(self.merges)
        self.special_to_id = {
            token: first_special_id + i for i, token in enumerate(self.special_tokens)
        }
        self.id_to_special = {i: token for token, i in self.special_to_id.items()}
        self._cache = {}

    # ---------- encode / decode ----------

    @staticmethod
    def _pretokenize(text):
        text = unicodedata.normalize("NFC", text)
        return PRETOKENIZE_PATTERN.findall(text)

    def encode(self, text, add_bos=False, add_eos=False):
        """
        Convert text into a list of token ids.

        Special tokens inside the text are not parsed, they are encoded
        as regular text. Use add_bos/add_eos to insert them.

        Args:
            text (str): input text.
            add_bos (bool): if True, prepend the BOS token.
            add_eos (bool): if True, append the EOS token.
        Returns:
            ids (list[int]): token ids.
        """
        ids = [self.bos_id] if add_bos else []
        for chunk in self._pretokenize(text):
            markers, chunk = _normalize_chunk(chunk)
            ids.extend(self.special_to_id[m] for m in markers)
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def _encode_chunk(self, chunk):
        """
        Encode a single pre-tokenized chunk, applying merges in the order
        they were learned. Results are cached.

        Args:
            chunk (str): pre-tokenized chunk.
        Returns:
            ids (list[int]): token ids.
        """
        if chunk in self._cache:
            return self._cache[chunk]

        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # the earliest learned merge among present pairs goes first
            pair = min(
                zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])

        self._cache[chunk] = ids
        return ids

    def decode(self, ids, skip_special_tokens=False):
        """
        Convert a list of token ids back into text.

        NOSP and CAP markers are always consumed: they are applied to the
        text that follows them. Markers not followed by a word (possible
        for model outputs) are ignored.

        Args:
            ids (list[int]): token ids.
            skip_special_tokens (bool): if True, drop BOS/EOS/PAD.
        Returns:
            text (str): decoded text. Invalid byte sequences (possible
                for arbitrary model outputs) are replaced with U+FFFD.
        """
        parts = []
        buffer = []  # bytes since the last special token
        pending = set()  # markers waiting for the next text

        def flush():
            if not buffer:
                return
            text = b"".join(buffer).decode("utf-8", errors="replace")
            buffer.clear()
            if NOSP in pending and text.startswith(" "):
                text = text[1:]
            if CAP in pending:
                j = 1 if text.startswith(" ") else 0
                text = text[:j] + text[j : j + 1].upper() + text[j + 1 :]
            pending.clear()
            parts.append(text)

        for i in ids:
            token = self.id_to_special.get(i)
            if token is None:
                buffer.append(self.vocab[i])
                continue
            flush()
            if token in (NOSP, CAP):
                pending.add(token)
            elif not skip_special_tokens:
                pending.clear()
                parts.append(token)
        flush()

        return "".join(parts)

    # ---------- properties ----------

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    @property
    def bos_id(self):
        return self.special_to_id[BOS]

    @property
    def eos_id(self):
        return self.special_to_id[EOS]

    @property
    def pad_id(self):
        return self.special_to_id[PAD]

    # ---------- save / load ----------

    def save(self, path=None, with_counts=True):
        """
        Save the tokenizer to a JSON file.

        Args:
            path (str | Path | None): path to the file. Relative paths are
                resolved against the project root. If None, self.path
                is used.
            with_counts (bool): if True, also save accumulated chunk
                frequencies so training can be continued later.
        """
        path = self.path if path is None else self._resolve_path(path)
        assert path is not None, "Provide path to save the tokenizer"
        path.parent.mkdir(exist_ok=True, parents=True)
        state = {
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],  # ids are implied by order
        }
        if with_counts:
            state["chunk_counts"] = dict(self._chunk_counts)
        write_json(state, str(path))

    @classmethod
    def load(cls, path):
        tokenizer = cls(path=path)
        assert tokenizer.is_fitted, f"Tokenizer file {tokenizer.path} does not exist"
        return tokenizer

    def _load_state(self, state):
        self.special_tokens = list(state["special_tokens"])
        self.merges = {
            tuple(pair): N_BYTES + i for i, pair in enumerate(state["merges"])
        }
        self._chunk_counts = Counter(state.get("chunk_counts", {}))
        self.is_fitted = True
        self._build_vocab()

    @staticmethod
    def _resolve_path(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT_PATH / path
