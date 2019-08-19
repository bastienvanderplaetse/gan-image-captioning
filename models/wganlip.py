import numpy as np
import torch
import torch.autograd as autograd

import math

from . import onehot_batch_data
from .discriminator import Discriminator
from .generator import Generator
from torch import nn
from torch.autograd import Variable

class WGANLip(nn.Module):

    def __init__(self, n_vocab, config_model, weights):
        super(WGANLip, self).__init__()
        self.n_trg_vocab = n_vocab

        emb_dim = config_model['emb_dim']
        dec_dim = config_model['dec_dim']

        self.clip = config_model['clip']
        self.gradient_weight = config_model['gradient_weight']
        self.generator_training = config_model['generator']['train_iteration']
        feature_size = config_model['feature_size']

        self.emb_G = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.emb_D = nn.Embedding(n_vocab, emb_dim, padding_idx=0)

        if weights is not None:
            self.emb_G.weight.data = torch.Tensor(weights)
            self.emb_D.weight.data = torch.Tensor(weights)
            self.emb_G.weight.require_grad = False
            self.emb_D.weight.require_grad = False

        self.G = Generator(
            input_size=emb_dim,
            hidden_size=dec_dim,
            n_vocab=n_vocab,
            emb=self.emb_G,
            feature_size=feature_size,
            config_model=config_model['generator']
        )

        self.D = Discriminator(
            input_size=emb_dim,
            hidden_size=dec_dim,
            n_vocab=n_vocab,
            emb=self.emb_D,
            feature_size=feature_size,
            config_model=config_model['discriminator']
        )

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

    def get_generator(self):
        return self.G

    def encode(self, batch):
        feats = (batch['feats'])
        return {'feats': (feats, None)}

    def forward(self, batch, optimG, optimD, epoch, iteration):
        sentences = batch['tokenized']
        sentences_G = sentences[:-1]
        sentences_D = sentences[1:]

        features = self.encode(batch)

        g_loss = 0
        d_loss = 0

        gen_s = self.G(features, sentences_G)

        # Train Discriminator
        for i in range(self.generator_training):
            optimD.zero_grad()

            real = self.D(features, sentences_D, epoch=epoch+1)
            fake = self.D(features, gen_s, one_hot=False, epoch=epoch+1)

            gradient_penalty = self.compute_gradient_penalty_lip(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab).data, gen_s.data, real, fake)
            d_loss = -torch.mean(real) + torch.mean(fake) + self.gradient_weight * gradient_penalty

            d_loss.backward(retain_graph=i!=(self.generator_training-1))
            optimD.step()

        optimG.zero_grad()
        gen_s = self.G(features, sentences_G)
        fake = self.D(features, gen_s, one_hot=False)
        g_loss = -torch.mean(fake)
        g_loss.backward()
        optimG.step()

        return {"G_loss": g_loss.to("cpu").item(), "D_loss": d_loss.to("cpu").item()}


    def compute_gradient_penalty_lip(self, D, feature, real_samples, fake_samples, realD, fakeD):
        x = real_samples.view(real_samples.size(1), -1)
        y = fake_samples.view(fake_samples.size(1), -1)
        dist = torch.sqrt(torch.sum((x-y) ** 2, dim=1))
        grad = (realD.view(-1) - fakeD.view(-1)).abs()
        lip = grad / (dist+1e-8)
        lip_loss = ((1.0-lip)**2).mean()

        return lip_loss

    def test_performance(self, data_loader, device):
        n_batch = 0
        cumul_loss_G = 0
        cumul_loss_D = 0

        for batch in data_loader:
            batch.device(device)

            features = self.encode(batch)

            sentences = batch['tokenized']
            sentences_G = sentences[:-1]
            sentences_D = sentences[1:]

            gen_s = self.G(features, sentences_G)
            real = self.D(features, sentences_D)
            fake = self.D(features, gen_s, one_hot=False)

            g_loss = -torch.mean(fake)
            d_loss = -torch.mean(real) + torch.mean(fake)

            cumul_loss_D += d_loss.to("cpu").item()
            cumul_loss_G += g_loss.to("cpu").item()
            n_batch += 1

        return {"G_loss": cumul_loss_G/n_batch, "D_loss": cumul_loss_D/n_batch}

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
