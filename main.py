import sys
import torch
import utils.explorer_helper as exh

from datasets.captioning import CaptioningDataset
from torch.utils.data import DataLoader
from utils import check_args, fix_seed

def run(args):
    # Get configuration
    config = exh.load_json(args.CONFIG)

    # Global initialization
    torch.cuda.init()
    device = torch.device(config['cuda']['device'] if (torch.cuda.is_available() and config['cuda']['ngpu'] > 0) else "cpu")
    seed = fix_seed(config['seed'])

    # Load vocabulary
    vocab = exh.load_json(config['data']['vocab'])

    # Prepare datasets and dataloaders
    training_dataset = CaptioningDataset(config['data']['train'], "train", vocab, config['sampler']['train'])
    train_iterator = DataLoader(
        training_dataset,
        batch_sampler=training_dataset.sampler,
        collate_fn=training_dataset.collate_fn,
        pin_memory=config['iterator']['train']['pin_memory'],
        num_workers=config['iterator']['train']['num_workers']
    )

    vloss_dataset = CaptioningDataset(config['data']['val'], "eval", vocab, config['sampler']['val'])
    vloss_iterator = DataLoader(
        vloss_dataset,
        batch_sampler=vloss_dataset.sampler,
        collate_fn=vloss_dataset.collate_fn,
        pin_memory=config['iterator']['val']['pin_memory'],
        num_workers=config['iterator']['val']['num_workers']
    )

    beam_dataset = CaptioningDataset(config['data']['val'], "beam", vocab, config['sampler']['beam'])
    beam_iterator = DataLoader(
        beam_dataset,
        batch_sampler=beam_dataset.sampler,
        collate_fn=beam_dataset.collate_fn,
        pin_memory=config['iterator']['beam']['pin_memory'],
        num_workers=config['iterator']['beam']['num_workers']
    )


if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)