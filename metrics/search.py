import torch

from utils import vocab as uvoc
from utils import memory_usage

from pprint import pprint

def beam_search(models, data_loader, vocab, beam_size=5, max_len=15, lp_alpha=0., suppress_unk=False, n_best=False, device=None):
     # This is the batch-size requested by the user but with sorted
    # batches, efficient batch-size will be <= max_batch_size
    max_batch_size = data_loader.batch_sampler.batch_size
    k = beam_size
    inf = -1000
    results = []
    enc_args = {}

    # For classical models that have single encoder, decoder and
    # target vocabulary
    decs = [m.get_generator() for m in models]
    f_inits = [dec.f_init for dec in decs]
    f_nexts = [dec.f_next for dec in decs]
    f_probs = [dec.f_probs for dec in decs]

    # Common parts
    encoders = [m.encode for m in models]
    unk = vocab['<unk>']['id']
    eos = vocab['<eos>']['id']
    n_vocab = len(vocab['token_list'])

    # Tensorized beam that will shrink and grow up to max_batch_size
    beam_storage = torch.zeros(
        max_len, max_batch_size, k, dtype=torch.long, device=device)
    mask = torch.arange(max_batch_size * k, device=device)
    nll_storage = torch.zeros(max_batch_size, device=device)

    for batch in data_loader:
        batch.device(device)

        # Always use the initial storage
        beam = beam_storage.narrow(1, 0, batch.size).zero_()

        # Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = mask.narrow(0, 0, batch.size * k)

        # nll: batch_size x 1 (will get expanded further)
        nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

        # Tile indices to use in the loop to expand first dim
        tile = range(batch.size)

        # Encode source modalities
        ctx_dicts = [encode(batch, **enc_args) for encode in encoders]

        # Get initial decoder state (N*H)
        h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]


        # we always have <bos> tokens except that the returned embeddings
        # may differ from one model to another.
        idxs = torch.LongTensor(batch.size).fill_(vocab['<bos>']['id']).to(device)
        log_ps = [f_prob(batch.size, n_vocab, idxs.device) for f_prob in f_probs]

        for tstep in range(max_len):
            # Select correct positions from source context
            ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

            # Get log probabilities and next state
            # log_p: batch_size x vocab_size (t = 0)
            #        batch_size*beam_size x vocab_size (t > 0)
            # NOTE: get_emb does not exist in some models, fix this.
            log_ps, h_ts = zip(
                    *[f_next(cd, dec.emb(idxs), p_t[tile], h_t[tile]) for
                  f_next, dec, cd, h_t, p_t in zip(f_nexts, decs, ctx_dicts, h_ts, log_ps)])


            # Do the actual averaging of log-probabilities
            log_p = sum(log_ps).data

            if suppress_unk:
                log_p[:, unk] = inf

            # Detect <eos>'d hyps
            idxs = (idxs == 2).nonzero()
            if idxs.numel():
                if idxs.numel() == batch.size * k:
                    break
                idxs.squeeze_(-1)
                # Unfavor all candidates
                log_p.index_fill_(0, idxs, inf)
                # Favor <eos> so that it gets selected
                log_p.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)

            # Expand to 3D, cross-sum scores and reduce back to 2D
            # log_p: batch_size x vocab_size ( t = 0 )
            #   nll: batch_size x beam_size (x 1)
            # nll becomes: batch_size x beam_size*vocab_size here
            # Reduce (N, K*V) to k-best
            nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                    k, sorted=False, largest=True)

            # previous indices into the beam and current token indices
            pdxs = beam[tstep] / n_vocab
            beam[tstep].remainder_(n_vocab)
            idxs = beam[tstep].view(-1)

            # Compute correct previous indices
            # Mask is needed since we're in flattened regime
            tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)

            if tstep > 0:
                # Permute all hypothesis history according to new order
                beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

        # Put an explicit <eos> to make idxs_to_sent happy
        beam[max_len - 1] = eos

        # Find lengths by summing tokens not in (pad,bos,eos)
        len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

        if lp_alpha > 0.:
            len_penalty = ((5 + len_penalty)**lp_alpha) / 6**lp_alpha

        # Apply length normalization
        nll.div_(len_penalty)

        top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
        hyps = beam[:, range(batch.size), top_hyps].t().to("cpu")
        results.extend(hyps.tolist())

        # if n_best:
        #     # each elem is sample, then candidate
        #     tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
        #     scores = nll.to('cpu').tolist()
        #     results.extend(
        #         [(vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
        # else:
        #     # Get best-1 hypotheses
        #     top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
        #     hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
        #     results.extend(vocab.list_of_idxs_to_sents(hyps.tolist()))

    # Recover order of the samples if necessary
    # if getattr(data_loader.batch_sampler, 'store_indices', False):
    #     results = [results[i] for i, j in sorted(
    #         enumerate(data_loader.batch_sampler.orig_idxs), key=lambda k: k[1])]
    # return results
    sentences = []
    for row in results:
        sentences.append(uvoc.tokens2words(row, vocab))

    return sentences

def _beam_search(generator, data_loader, vocab, config, device):
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
            # print("=============================")
            ctx_dict = tile_ctx_dict(features, tile)
            # print('001')
            # print(pdxs)
            # print("============")
            # print(generator.emb)
            # print("-----------------")
            # print(idxs)
            # print("-----------------")
            # print(generator.emb(idxs))
            # print("-----------------")
            # print(log_proba)
            # print(log_proba.shape)
            # print(batch.size)
            # print("-----------------")
            # print(tile)
            # if tstep > 0:
            #     print(tile.shape)
            # print("-----------------")
            # print(log_proba[tile])
            # print("-----------------")
            # print(h)
            # print("-----------------")
            # print(h[tile])
            log_proba, h = generator.f_next(ctx_dict, generator.emb(idxs), log_proba[tile], h[tile])
            # print('002')
            # print(pdxs)
            # print(log_proba.shape)
            # print(sum(log_proba).shape)
            # print(batch.size)
            idxs = (idxs == 2).nonzero()
            # print('003')
            # print(pdxs)
            if idxs.numel():
                if idxs.numel() == batch.size * k:
                    break
                idxs.squeeze_(-1)
                log_proba.index_fill_(0, idxs, inf)
                log_proba.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)
            # print('004')
            # print(log_proba)
            # print(log_proba.view(batch.size, -1, n_vocab))
            # print(log_proba.shape)
            # print(log_proba.view(batch.size, -1, n_vocab).shape)
            # print(nll)
            # print(nll.unsqueeze_(2))
            # print(nll.shape)
            # print(nll.unsqueeze_(2).shape)
            # print(nll.unsqueeze_(2).add(log_proba.view(batch.size, -1, n_vocab)))
            # print(nll.unsqueeze_(2).add(log_proba.view(batch.size, -1, n_vocab).view(batch.size, -1)))
            nll, beam[tstep] = nll.unsqueeze_(2).add(log_proba.view(batch.size, -1, n_vocab)).view(batch.size, -1).topk(k, sorted=False, largest=True)
            # print('005')
            # print(pdxs)

            pdxs = beam[tstep] / n_vocab
            # print('006')
            # print(beam)
            # print(tstep)
            # print(n_vocab)
            # print(beam[tstep])
            # print(pdxs)
            beam[tstep].remainder_(n_vocab)
            # print('007')
            # print(pdxs)
            idxs = beam[tstep].view(-1)
            # print('008')
            # print(batch.size)
            # print(pdxs)
            # print(pdxs.shape)
            # print(pdxs.view(-1))
            # print(pdxs.view(-1).shape)
            # print(nk_mask)
            # print(nk_mask.shape)
            # print(k)
            # print(nk_mask / k)
            # print((nk_mask / k).shape)
            tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)
            # print('009')
            # print(tile.shape)
            # print(tile)
            # print(pdxs)

            if tstep > 0:
                # Permute all hypothesis history according to new order
                # print(len(pdxs))
                # print(pdxs)
                # print(tstep)
                # print(pdxs.repeat(tstep, 1, 1))
                # print(beam.shape)
                # print(beam)
                # print(beam[:tstep])
                # print("===========")
                beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1)) # Try by replacing tstep by tsep+1
            # print('010')
            # print(pdxs)

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
    # print(ctx_dict)
    # for k, (t, mask) in ctx_dict.items():
    #     print(k)
    #     print(t)
    #     print(mask)
    #     print(idxs)
    # print("---____________________________________________________---")
    return {
        k: (t[:, idxs], None if mask is None else mask[:, idxs])
        for k, (t, mask) in ctx_dict.items()
    }

def max_search(model, data_loader, vocab, max_len=15, device=None):
    generator = model.G
    n_vocab = len(vocab['token_list'])
    results = []

    for batch in data_loader:
        batch.device(device)
        # print(batch)

        features = model.encode(batch)

        sentences= batch['tokenized']
        h = generator.f_init(features)
        prob = torch.zeros(sentences.shape[1], n_vocab, device=device)
        y_t = generator.emb(sentences[0])
        tokens = torch.zeros(max_len, batch.size, device=device)

        for tstep in range(max_len):
            prob, h = generator.f_next(features, y_t, prob, h)
            # m = 0
            # m_i = 0
            # t = 0
            # for i in range(prob[0].shape[0]):
            #     t = t + prob[0][i]
            #     if prob[0][i] > m:
            #         m = prob[0][i]
            #         m_i = i

            # print(torch.sum(prob, dim=0))
            # print(torch.sum(prob, dim=1))
            # print(prob.shape)
            # b1 = torch.argmax(prob,dim=0)
            # print(b1.shape)
            y_t = torch.argmax(prob,dim=1)
            # with open('aaa.txt', 'a') as f:
            #     print((y_t, m_i, m, t), file=f)
            tokens[tstep] = y_t
            # print(tokens)
            # print(y_t)
            # print(y_t.shape)
            y_t = generator.emb(y_t)
            # print(y_t)
            # print(y_t.shape)


        tokens = tokens.to('cpu')
        tokens = tokens[:,range(batch.size)].t().tolist()
        results.extend(tokens)
        # break

    sentences = []
    for row in results:
        sentences.append(uvoc.tokens2words(row, vocab))

    return sentences
