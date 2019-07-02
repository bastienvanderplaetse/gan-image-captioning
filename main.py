import os
import sys
import torch
import torch.optim as optim
import utils.explorer_helper as exh

from datasets.captioning import CaptioningDataset
from metrics.scores import bleu_score_4, prepare_references
from metrics.search import beam_search
from models.wgan import WGAN
from torch.utils.data import DataLoader
from utils import check_args, fix_seed, memory_usage

def run(args):
    print(torch.backends.cudnn.benchmark)
    # Get configuration
    config = exh.load_json(args.CONFIG)

    # Prepare folders for logging
    logging = config['logging']['activate']
    if logging:
        exh.create_directory("output")
        output = os.path.join("output", config['logging']['output_folder'])
        exh.create_directory(output)

    # Global initialization
    torch.cuda.init()
    device = torch.device(config['cuda']['device'] if (torch.cuda.is_available() and config['cuda']['ngpu'] > 0) else "cpu")
    seed = fix_seed(config['seed'])

    # Load vocabulary
    vocab = exh.load_json(config['data']['vocab'])

    # Prepare references
    references = exh.read_file(config['data']['val']['captions'])
    references = prepare_references(references)

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

    # Prepare model
    model = WGAN(len(vocab['token_list']), config['model'])

    lr = config['model']['optimizers']['lr']
    betas = (config['model']['optimizers']['betas']['min'], config['model']['optimizers']['betas']['max'])
    weight_decay = config['model']['optimizers']['weight_decay']
    
    optim_D = optim.Adam(model.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optim_G = optim.Adam(model.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    model.to(device)

    fix_seed(config['seed'] + 1)

    generator_trained = config['model']['generator']['train_iteration']

    best_bleu = (0, 1)

    scores = {
        "bleu": [],
        "G_loss_train": [],
        "D_loss_train": [],
        "G_loss_val": [],
        "D_loss_val": []
    }
    bleus = []

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['max_epoch']):
        print("Starting Epoch {}".format(epoch + 1))

        iteration = 1

        d_batch = 0
        g_batch = 0
        d_loss = 0
        g_loss = 0

        for batch in train_iterator:
            batch.device(device)
            
            out = model(batch, optim_G, optim_D, epoch, iteration)
            
            d_loss += out['D_loss']
            d_batch += 1

            if epoch+1 == 31:
                print(out['D_loss'])
                print(d_loss)
                print("============")

            if iteration % generator_trained == 0:
                g_loss += out['G_loss']
                g_batch += 1

            iteration += 1
                
        print("Training : Mean G loss : {} / Mean D loss : {}".format(g_loss/g_batch, d_loss/d_batch))
        scores['G_loss_train'].append((g_loss/g_batch))
        scores['D_loss_train'].append((d_loss/d_batch))
        
        # Validation
        model.train(False)
        torch.set_grad_enabled(False)

        # Loss
        out = model.test_performance(vloss_iterator, device)
        print("Validation Loss : G loss : {} / D loss : {}".format(out['G_loss'], out['D_loss']))
        scores['G_loss_val'].append(out['G_loss'])
        scores['D_loss_val'].append(out['D_loss'])

        # Beam search
        print("Beam search...")
        # generated_sentences = beam_search(model.G, beam_iterator, vocab, config['beam_search'], device)
        generated_sentences = beam_search([model], beam_iterator, vocab, device=device)

        # BLEU score
        score = bleu_score_4(references, generated_sentences)
        print("BLEU score : {}".format(score))
        scores['bleu'].append(score)

        if score > best_bleu[0]:
            best_bleu = (score, epoch + 1)

        print("Best BLEU so far : {} (Epoch {})".format(best_bleu[0], best_bleu[1]))

        if logging:
            output_file = 'output_{}'.format(epoch + 1)
            output_sentences = os.path.join(output, output_file)
            exh.write_text('\n'.join(generated_sentences), output_sentences)
        
        model.train(True)
        torch.set_grad_enabled(True)
    
    if logging:
        output_scores = os.path.join(output, 'scores.json')
        exh.write_json(scores, output_scores)
        print("Scores saved in {}".format(output_scores))        


if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)