import torch

from utils import vocab as uvoc
from utils import memory_usage

def beam_search(generator, data_loader, vocab, config, device):
    print("BS")
    memory_usage()
    import sys
    sys.exit(0)
    max_batch_size = data_loader.batch_sampler.batch_size
    max_len = config['max_len']
    k = config['beam_size']
    n_vocab = len(vocab['token_list'])
    inf = -1000
    
    nll_storage = torch.zeros(max_batch_size, device=device)    
    beam_storage = torch.zeros(max_len+1, max_batch_size, k, dtype=torch.long, device=device)
    mask = torch.arange(max_batch_size * k, device=device)
    
    results = []
    for batch in data_loader:
        batch.device(device)
            
        feats = (batch['feats'])
        features = {'feats': (feats, None)}
        # features['feats'][0].shape => 1 x batch_size x 2048

        h = generator.f_init(features) #h.shape => batch_size x 256

        log_proba = torch.zeros(batch.size, n_vocab, device=device) # batch_size x n_vocab

        tile = range(batch.size)

        idxs = torch.LongTensor(batch.size).fill_(vocab['<bos>']['id']).to(device)

        nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1) # batch_size x 1

        beam = beam_storage.narrow(1, 0, batch.size).zero_()

        nk_mask = mask.narrow(0, 0, batch.size * k)

        for tstep in range(max_len):
            ctx_dict = tile_ctx_dict(features, tile)

            log_proba, h = generator.f_next(ctx_dict, generator.emb(idxs), log_proba[tile], h[tile])
            
            idxs = (idxs == 2).nonzero()
            if idxs.numel():
                if idxs.numel() == batch.size * k:
                    break
                idxs.squeeze_(-1)
                log_proba.index_fill_(0, idxs, inf)
                log_proba.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)
            
            nll, beam[tstep] = nll.unsqueeze_(2).add(log_proba.view(batch.size, -1, n_vocab)).view(batch.size, -1).topk(k, sorted=False, largest=True)
            
            pdxs = beam[tstep] / n_vocab
            beam[tstep].remainder_(n_vocab)
            idxs = beam[tstep].view(-1)
            
            tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)
            
            if tstep > 0:
                # Permute all hypothesis history according to new order
                beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))
        
        beam[max_len] = vocab['<eos>']['id']
        
        top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
        hyps = beam[:, range(batch.size), top_hyps].t().to("cpu")
        results.extend(hyps.tolist())
    
    sentences = []
    for row in results:
        sentences.append(uvoc.tokens2words(row, vocab))

    return sentences

def tile_ctx_dict(ctx_dict, idxs):
    """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
    # 1st: tensor, 2nd optional mask
    return {
        k: (t[:, idxs], None if mask is None else mask[:, idxs])
        for k, (t, mask) in ctx_dict.items()
    }