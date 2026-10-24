from src.datasets.base_text_dataset import BaseTextDataset
from src.utils.io_utils import ROOT_PATH


class TextFilesDataset(BaseTextDataset):
    """
    Dataset of local UTF-8 text files. Nothing is downloaded.

    Used until the format of the real data is known, and as an example
    of a nested BaseTextDataset class.
    """

    def __init__(self, files, part, block_size, data_dir, doc_per_line=False):
        """
        Args:
            files (str): glob of text files, relative to the project root.
            part (str): partition name.
            block_size (int): number of tokens in one training example.
            data_dir (str): directory for the tokenized cache.
            doc_per_line (bool): if True, every non-empty line is a
                separate document. Otherwise, every file is a document.
        """
        self.files = sorted(ROOT_PATH.glob(files))
        self.files_glob = files
        self.doc_per_line = doc_per_line
        super().__init__(data_dir=data_dir, part=part, block_size=block_size)

    def _is_downloaded(self):
        return len(self.files) > 0

    def _download(self):
        raise FileNotFoundError(f"No files match {self.files_glob}")

    def iter_texts(self):
        for file in self.files:
            with file.open("rt", encoding="utf-8") as f:
                if self.doc_per_line:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
                else:
                    yield f.read()
