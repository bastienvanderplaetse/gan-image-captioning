import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils import explorer_helper as exh
from utils import vocab as uvoc

class CaptionDataset(Dataset):
    """A PyTorch dataset for sentences. Adapted from nmtpytorch framework

    Arguments:
        fname (str): A string object giving the corpus.
        vocab (Vocabulary): A ``Vocabulary`` instance for the given corpus.
        bos (bool, optional): If ``True``, special beginning-of-sentence
            "<bos>" and ending-of-sentence "<eos> markers will be prepended to sentences.
    """

    def __init__(self, fname, vocab, bos=True, key=None, mode=None):
        self.fname = fname
        captions = exh.read_file(fname)
        if mode == 'train':
            captions = captions[:50000]
        self.data = []
        self.lengths = []
        self.key = key

        for caption in captions:
            tokens = uvoc.words2tokens(caption, vocab, bos)
            self.data.append(tokens)
            self.lengths.append(len(tokens))

        self.size = len(self.data)

    @staticmethod
    def to_torch(batch):
        return pad_sequence(
            [torch.tensor(b, dtype=torch.long) for b in batch], batch_first=False)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.fname, self.__len__())
        return s
