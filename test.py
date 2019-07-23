import sys
import torch
import utils.explorer_helper as exh
import utils.vocab as uvoc

from datasets.captioning import CaptioningDataset
from metrics.scores import bleu_score, prepare_references
from models.wgan import WGAN
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

    # Prepare references
    references = exh.read_file(config['data']['test']['captions'])
    references = prepare_references(references)

    beam_dataset = CaptioningDataset(config['data']['test'], "test", vocab, config['sampler']['test'])
    beam_iterator = DataLoader(
        beam_dataset,
        batch_sampler=beam_dataset.sampler,
        collate_fn=beam_dataset.collate_fn,
        pin_memory=config['iterator']['test']['pin_memory'],
        num_workers=config['iterator']['test']['num_workers']
    )

    # Prepare model
    weights = None
    if len(config['model']['embeddings']) > 0:
        weights = uvoc.init_weights(vocab, config['model']['emb_dim'])
        uvoc.glove_weights(weights, config['model']['embeddings'], vocab)

    model = WGAN(len(vocab['token_list']), config['model'], weights)
    model.load_state_dict(torch.load(config['load_dict']), strict=False)

    # lr = config['model']['optimizers']['lr']
    # betas = (config['model']['optimizers']['betas']['min'], config['model']['optimizers']['betas']['max'])
    # weight_decay = config['model']['optimizers']['weight_decay']
    
    # optim_D = optim.Adam(model.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    # optim_G = optim.Adam(model.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    model.to(device)

    fix_seed(config['seed'] + 1)

    # generator_trained = config['model']['generator']['train_iteration']

    # scores = {
    #     "BLEU": [],
    #     "G_loss_train": [],
    #     "D_loss_train": []
    #     # "G_loss_val": [],
    #     # "D_loss_val": []
    # }
    # max_bleu = config['BLEU']['max_bleu']
    # bleus = [[]] * max_bleu
    # best_bleu = (0, 1)

    # torch.autograd.set_detect_anomaly(True)
    model.train(False)
    torch.set_grad_enabled(False)



if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)