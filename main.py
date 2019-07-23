import os
import sys
import time
import torch
import torch.optim as optim
import utils.explorer_helper as exh
import utils.vocab as uvoc

from datasets.captioning import CaptioningDataset
from metrics.scores import bleu_score, prepare_references
from metrics.search import beam_search, max_search
from models.wgan import WGAN
from models.relativistic_gan import RelativisticGAN
from torch.utils.data import DataLoader
from utils import check_args, fix_seed, memory_usage

from torchviz import make_dot, make_dot_from_trace

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
    references = exh.read_file(config['data']['beam']['captions'])
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

    # vloss_dataset = CaptioningDataset(config['data']['val'], "eval", vocab, config['sampler']['val'])
    # vloss_iterator = DataLoader(
    #     vloss_dataset,
    #     batch_sampler=vloss_dataset.sampler,
    #     collate_fn=vloss_dataset.collate_fn,
    #     pin_memory=config['iterator']['val']['pin_memory'],
    #     num_workers=config['iterator']['val']['num_workers']
    # )

    beam_dataset = CaptioningDataset(config['data']['beam'], "beam", vocab, config['sampler']['beam'])
    beam_iterator = DataLoader(
        beam_dataset,
        batch_sampler=beam_dataset.sampler,
        collate_fn=beam_dataset.collate_fn,
        pin_memory=config['iterator']['beam']['pin_memory'],
        num_workers=config['iterator']['beam']['num_workers']
    )

    # Prepare model
    weights = None
    if len(config['model']['embeddings']) > 0:
        weights = uvoc.init_weights(vocab, config['model']['emb_dim'])
        uvoc.glove_weights(weights, config['model']['embeddings'], vocab)

    model = WGAN(len(vocab['token_list']), config['model'], weights)
    # model = RelativisticGAN(len(vocab['token_list']), config['model'], weights)
    model.reset_parameters()

    lr = config['model']['optimizers']['lr']
    betas = (config['model']['optimizers']['betas']['min'], config['model']['optimizers']['betas']['max'])
    weight_decay = config['model']['optimizers']['weight_decay']

    optim_D = optim.Adam(model.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    optim_G = optim.Adam(model.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    model.to(device)

    fix_seed(config['seed'] + 1)

    generator_trained = config['model']['generator']['train_iteration']

    scores = {
        "BLEU": [],
        "G_loss_train": [],
        "D_loss_train": []
        # "G_loss_val": [],
        # "D_loss_val": []
    }
    max_bleu = config['BLEU']['max_bleu']
    bleus = [[]] * max_bleu
    best_bleu = (0, 1)

    # torch.autograd.set_detect_anomaly(True)
    model.train(True)
    torch.set_grad_enabled(True)

    # for epoch in range(config['max_epoch']):
    epoch = 1
    cpt = 0
    while True:
        secs = time.time()
        print("Starting Epoch {}".format(epoch))

        iteration = 1

        d_batch = 0
        g_batch = 0
        d_loss = 0
        g_loss = 0
        for batch in train_iterator:
            # if time.time()-secs <= 30*60:
            batch.device(device)

            out = model(batch, optim_G, optim_D, epoch, iteration)

            d_loss += out['D_loss']
            d_batch += 1
            g_loss += out['G_loss']
            g_batch += 1

            # print(time.time()-secs)

            # if iteration % generator_trained == 0:
            #     g_loss += out['G_loss']
            #     g_batch += 1

            iteration += 1

        print("Training : Mean G loss : {} / Mean D loss : {} ({} seconds elapsed)".format(g_loss/g_batch, d_loss/d_batch, time.time()-secs))
        scores['G_loss_train'].append((g_loss/g_batch))
        scores['D_loss_train'].append((d_loss/d_batch))

        # Validation
        model.train(False)
        torch.set_grad_enabled(False)

        # Loss
        # out = model.test_performance(vloss_iterator, device)
        # print("Validation Loss : G loss : {} / D loss : {}".format(out['G_loss'], out['D_loss']))
        # scores['G_loss_val'].append(out['G_loss'])
        # scores['D_loss_val'].append(out['D_loss'])

        # Beam search
        print("Beam search...")
        # generated_sentences = beam_search(model.G, beam_iterator, vocab, config['beam_search'], device)
        # generated_sentences = beam_search([model], beam_iterator, vocab, beam_size=config['beam_search']['beam_size'], max_len=config['beam_search']['max_len'], device=device)
        generated_sentences = max_search(model, beam_iterator, vocab, max_len=config['beam_search']['max_len'], device=device)

        # BLEU score
        for n in range(3,max_bleu):
            score = bleu_score(references, generated_sentences, n+1)
            bleus[n].append(score)
            print("BLEU-{} score : {}".format(n+1, score))


        # score = bleu_score(references, generated_sentences)
        # print("BLEU score : {}".format(score))
        # scores['bleu'].append(score)

        if score > best_bleu[0]:
            best_bleu = (score, epoch)
            filename = 'output_epoch{}_bleu{}'.format(epoch,score)
            out_file = os.path.join(output, filename)
            torch.save(model.state_dict(), out_file)

        print("Best BLEU so far : {} (Epoch {})".format(best_bleu[0], best_bleu[1]))

        if logging:
            output_file = 'output_{}'.format(epoch)
            output_sentences = os.path.join(output, output_file)
            exh.write_text('\n'.join(generated_sentences), output_sentences)

        model.train(True)
        torch.set_grad_enabled(True)
        print("Epoch finished in {} seconds".format(time.time()-secs))

        # if epoch - best_bleu[1] == 5:
        #     break

        epoch += 1



    if logging:
        scores['BLEU'] = bleus
        output_scores = os.path.join(output, 'scores.json')
        exh.write_json(scores, output_scores)
        print("Scores saved in {}".format(output_scores))


if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)
