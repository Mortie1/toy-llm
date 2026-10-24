from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """
    Base class for tokenizers.

    Defines the interface used by datasets (encode), models (vocab_size),
    and generation (decode).
    """

    @abstractmethod
    def encode(self, text, add_bos=False, add_eos=False):
        """
        Convert text into a list of token ids.

        Args:
            text (str): input text.
            add_bos (bool): if True, prepend the BOS token.
            add_eos (bool): if True, append the EOS token.
        Returns:
            ids (list[int]): token ids.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, ids):
        """
        Convert a list of token ids back into text.

        Args:
            ids (list[int]): token ids.
        Returns:
            text (str): decoded text.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def vocab_size(self):
        """
        Total number of tokens, including special tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        """
        Save the tokenizer to disk.

        Args:
            path (str | Path): path to the file.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path):
        """
        Load the tokenizer from disk.

        Args:
            path (str | Path): path to the file.
        Returns:
            tokenizer (BaseTokenizer): loaded tokenizer.
        """
        raise NotImplementedError()

    def __len__(self):
        return self.vocab_size
