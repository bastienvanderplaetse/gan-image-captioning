import torch

from . import get_rnn_hidden_state
from .ff import FF
from .grucell import GRUCell
from collections import defaultdict
from torch import nn

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, n_vocab, emb, feature_size, config_model):
        super().__init__()
        self.emb = emb
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab

        self.ctx_name = 'feats'
        dec_init_type = config_model['dec_init_type']
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init_type))

        self.dec0 = GRUCell(self.input_size, self.hidden_size)
        self.dec1 = GRUCell(self.hidden_size, self.hidden_size)

        att_activ = config_model['att_activ']
        self.att = FF(feature_size, self.hidden_size, activ=att_activ)

        self.out2prob_classif = FF(self.hidden_size, 1, activ="sigmoid")
        # self.out2prob_classif = FF(self.hidden_size, 1, activ="tanh")
        # self.out2prob_classif = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(self.hidden_size, 1)
        # )


        self.n_states = 1

    def forward(self, features, y, one_hot=True, epoch=0):
        # Convert token indices to embeddings -> T*B*E
        if one_hot:
            y_emb = self.emb(y)
        else:
            y_emb = torch.matmul(y, self.emb.weight)

        # Get initial hidden state
        h = self.f_init(features)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0]):
            o, h = self.f_next(features, y_emb[t], h)

        valid = self.out2prob_classif(o)

        return valid

    def f_next(self, features, y, h):
        h1_c1 = self.dec0(y, h)
        h1 = get_rnn_hidden_state(h1_c1)

        ct = self.att(features[self.ctx_name][0]).squeeze(0)
        h1_ct = torch.mul(h1, ct)

        o = self.dec1(h1_ct, h1_c1)

        return o, o

    def f_init(self, ctx_dict):
        """Returns the initial h_0, c_0 for the decoder."""
        self.history = defaultdict(list)
        return self._init_func(*ctx_dict[self.ctx_name])

    def _rnn_init_zero(self, ctx, ctx_mask):
        h = torch.zeros(
            ctx.shape[1], self.hidden_size, device=ctx.device)
        if self.n_states == 2:
            return (h,h)
        return h
