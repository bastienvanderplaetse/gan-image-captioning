import numpy as np

from . import get_collate
from .caption import CaptionDataset
from .numpy import NumpyDataset
from .text import TextDataset
from samplers.bucket import BucketBatchSampler
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from utils import explorer_helper as exh

class CaptioningDataset(Dataset):
    def __init__(self, data_files, mode, vocab, sampler_config):
        self.datasets = dict()

        # Image features
        self.datasets['feats'] = NumpyDataset(data_files['features'], key="feats", mode=mode)
        # Sentences in human language
        self.datasets['text'] = TextDataset(data_files['captions'], mode=mode)
        # Sentences in tokens without <bos> and <eos>
        self.datasets['captions'] = CaptionDataset(data_files['captions'], vocab, bos=False, key="captions", mode=mode)
        # Sentences in tokens with <bos> and <eos>
        self.datasets['tokenized'] = CaptionDataset(data_files['captions'], vocab, key="tokenized", mode=mode)
        # Image file name
        self.datasets['image_file'] = TextDataset(data_files['links'], mode=mode)

        # Checks that data are aligned
        sizes = set([len(dataset) for dataset in self.datasets.values()])
        assert len(sizes) == 1, "Non-parallel datasets are not supported."

        self.size = list(sizes)[0]


        self.required_keys = ["feats", "captions", "tokenized"]
        self.collate_fn = get_collate([self.datasets[key] for key in self.required_keys])

        if mode == "beam" or mode == "test":
            sampler = SequentialSampler(self)
            self.sampler = BatchSampler(sampler, batch_size=sampler_config['batch_size'], drop_last=sampler_config['drop_last'])
        else:
            self.sampler = BucketBatchSampler(
                batch_size=sampler_config['batch_size'],
                sort_lens=self.datasets['captions'].lengths,
                max_len=sampler_config['max_len'],
                store_indices=mode != 'train'
            )

    def __getitem__(self, idx):
        return {
            k: v[idx] for k, v in self.datasets.items() if k in self.required_keys
        }

    def get(self, idx):
        return {
            k: v[idx] for k, v in self.datasets.items()
        }

    def __len__(self):
        return self.size

    def __repr__(self):
        pass