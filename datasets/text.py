# -*- coding: utf-8 -*-
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils import explorer_helper as exh


class TextDataset(Dataset):
    """A PyTorch dataset for sentences.

    Arguments:
        fname (str): A string object giving the corpus.
    """

    def __init__(self, fname, mode=None):
        self.fname = fname
        self.data = exh.read_file(fname)
        if mode == 'train':
            self.data = self.data[:50000]
        self.lens = [len(sentence) for sentence in self.data]

        self.size = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} sentences)\n".format(
            self.__class__.__name__, self.fname, self.__len__())
        return s
