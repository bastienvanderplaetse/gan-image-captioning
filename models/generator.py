import torch
import torch.nn.functional as F

from . import get_rnn_hidden_state
from .ff import FF
from .grucell import GRUCell
from collections import defaultdict
from torch import nn

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, n_vocab, emb, feature_size, config_model):
        super().__init__()
        self.n_vocab = n_vocab
        self.dropout_emb = config_model['dropout_emb']
        self.dropout_state = config_model['dropout_state']
        self.local_dropout = config_model['dropout_type'] == "local"
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.z = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.ctx_name = 'feats'

        dec_init = config_model['dec_init_type']
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        self.emb = emb
        # print(self.emb.weight.require_grad)

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(p=self.dropout_emb)

        if self.dropout_state > 0:
            self.do_state = nn.Dropout(p=self.dropout_state)

        self.dec0 = GRUCell(self.input_size, self.hidden_size)
        self.dec1 = GRUCell(self.hidden_size, self.hidden_size)

        att_activ = config_model['att_activ']
        self.att = FF(feature_size, self.hidden_size, activ=att_activ)

        self.hid2out = FF(self.hidden_size, self.input_size, bias_zero=True, activ="tanh")
        self.out2prob = FF(self.input_size, self.n_vocab)

        self.n_states = 1

    def forward(self, features, y):
        # print(self.emb.weight.require_grad)
        y = self.emb(y)
        # y.shape (X, Y, Z) => X = taille des phrases / Y = nbre de phrases / Z = vecteur d'embedding des mots

        # Get initial hidden state
        h = self.f_init(features)

        probs = torch.zeros(y.shape[0], y.shape[1], self.n_vocab, device=y.device)

        prob = torch.zeros(y.shape[1], self.n_vocab, device=y.device)

        for t in range(y.shape[0]):
            # y[t].shape -> (Y, Z) => Y = nbre de phrase, Z = vecteur d'embed des mots
            prob, h = self.f_next(features, y[t], prob, h)
            probs[t] = prob
        return probs

    def f_next(self, features, y, prob, h):
        current_batch_size = len(y)

        if self.dropout_emb:
            if self.local_dropout:
                ones = torch.ones(current_batch_size)
                ones = self.do_emb(ones)
                dropped_y = torch.zeros_like(y, device=y.device)

                for i in range(current_batch_size):
                    if ones[i] == 0.0:
                        dropped_y[i] = y[i] - y[i]
                    else:
                        dropped_y[i] = y[i]
            else:
                dropped_y = self.do_emb(y)
        else:
            dropped_y = y

        # hidden_state from first decoder
        h1_c1 = self.dec0(dropped_y, h)
        h1 = get_rnn_hidden_state(h1_c1)

        if self.dropout_state > 0:
            h1 = self.do_state(h1)

        ct = self.att(features[self.ctx_name][0]).squeeze(0)
        h1_ct = torch.mul(h1, ct)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(h1_ct, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        logit = self.hid2out(h2)

        prob = F.softmax(self.out2prob(logit), dim=-1)
        # prob = F.log_softmax(self.out2prob(logit), dim=-1)

        return prob, h2


    def f_probs(self, batch_size, n_vocab, device):
        # return torch.zeros(batch_size, n_vocab, device=device)
        return torch.ones(batch_size, n_vocab, device=device)

    def f_init(self, ctx_dict):
        """Returns the initial h_0, c_0 for the decoder."""
        self.history = defaultdict(list)
        return self._init_func(*ctx_dict[self.ctx_name])

    def _rnn_init_random(self, ctx, ctx_mask):
        """Returns the initial h_0, c_0 for the decoder."""
        return self.z.rsample(torch.Size([ctx.shape[1], self.hidden_size])).squeeze(-1).to(ctx.device)

    def _rnn_init_zero(self, ctx, ctx_mask):
        h = torch.zeros(
            ctx.shape[1], self.hidden_size, device=ctx.device)
        if self.n_states == 2:
            return (h,h)
        return h
