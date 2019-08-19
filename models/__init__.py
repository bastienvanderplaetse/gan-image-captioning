import torch
import torch.nn.functional as F

__all__ = ['wgan','relativistic_gan', 'wganbase', 'wgangp', 'wganlip']

def get_activation_fn(name):
    """Returns a callable activation function from torch. From nmtpytorch framework"""
    if name in (None, 'linear'):
        return lambda x: x
    elif name in ('sigmoid', 'tanh'):
        return getattr(torch, name)
    else:
        return getattr(F, name)

def get_rnn_hidden_state(h):
    """Returns h_t transparently regardless of RNN type.  From nmtpytorch framework"""
    return h if not isinstance(h, tuple) else h[0]

def onehot_batch_data(idxs, n_classes):
    """Returns a binary batch_size x n_classes one-hot tensor. From nmtpytorch framework"""
    out = torch.zeros(
        idxs.shape[0], idxs.shape[1], n_classes, device=idxs.device)

    for i, id in zip(out, idxs):
        for j, indices in zip(i, id):
            j.scatter_(0, indices, 1)
    return out
