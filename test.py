import os
import sys
import torch
import torch.optim as optim
import utils.explorer_helper as exh
import utils.vocab as uvoc

from datasets.captioning import CaptioningDataset
from metrics.scores import bleu_score, prepare_references
from metrics.search import beam_search, max_search
from models.wgan import WGAN
from torch.utils.data import DataLoader
from utils import check_args, fix_seed

def run(args):
    print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = True
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

    beam_dataset = CaptioningDataset(config['data']['test'], "beam", vocab, config['sampler']['test'])
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

    model.reset_parameters()
    print("The state dict keys: \n\n", model.state_dict().keys())

    model.load_state_dict(torch.load(config['load_dict']))
    for param in list(model.parameters()):
        param.requires_grad = False

    c = torch.load(config['load_dict'])
    for x in model.state_dict():
        if len(model.state_dict()[x].shape) == 1:
            model.state_dict()[x][:] = c[x]
        elif len(model.state_dict()[x].shape) == 2:
            model.state_dict()[x][:,:] = c[x]

    model.to(device)

    fix_seed(config['seed'] + 1)


    model.train(False)
    torch.set_grad_enabled(False)
    model.eval()

    model.G.emb.weight.data = c['G.emb.weight']

    generated_sentences = max_search(model, beam_iterator, vocab, max_len=config['beam_search']['max_len'], device=device)
    output_file = 'output_argmax'
    output_sentences = output_file
    exh.write_text('\n'.join(generated_sentences), output_sentences)
    score = bleu_score(references, generated_sentences, 4)
    print(score)
    generated_sentences = beam_search([model], beam_iterator, vocab, beam_size=config['beam_search']['beam_size'], max_len=config['beam_search']['max_len'], device=device)
    output_file = 'output_beam'
    output_sentences = output_file
    exh.write_text('\n'.join(generated_sentences), output_sentences)
    score = bleu_score(references, generated_sentences, 4)
    print(score)



if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)
