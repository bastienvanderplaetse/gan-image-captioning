import math
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """
    An implementation of GRUCell. From nmtpytorch framework

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        # x = x.view(-1, x.size(1))
        try:
            gate_x = self.x2h(x)
            gate_h = self.h2h(hidden)

            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + (resetgate * h_n))

            hy = newgate + inputgate * (hidden - newgate)
        except RuntimeError as e:
            print(e)
        return hy