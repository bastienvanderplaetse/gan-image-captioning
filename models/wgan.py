import numpy as np
import torch
import torch.autograd as autograd

import math

from . import onehot_batch_data
from .discriminator import Discriminator
from .generator import Generator
from torch import nn
from torch.autograd import Variable

class WGAN(nn.Module):

    def __init__(self, n_vocab, config_model, weights):
        super(WGAN, self).__init__()
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

    def ffforward(self, batch, optimG, optimD, epoch, iteration):
        sentences= batch['tokenized']
        sentences_G = sentences[:-1]
        sentences_D = sentences[1:]

        features = self.encode(batch)

        one = torch.cuda.FloatTensor([1])
        mone = one * -1

        for i in range(self.generator_training):
            self.D.zero_grad()

            D_real = self.D(features, sentences_D, epoch=epoch+1)
            D_real = torch.mean(D_real)
            D_real.backward(mone,retain_graph=True)

            fake_samples = self.G(features, sentences_G)
            D_fake = self.D(features, fake_samples, one_hot=False, epoch=epoch+1)
            D_fake = torch.mean(D_fake)
            D_fake.backward(one,retain_graph=True)

            gradient_penalty = self.compute_gradient_penalty(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab), fake_samples)
            gradient_penalty.backward(retain_graph=i!=(self.generator_training-1))

            D_cost = D_fake - D_real + gradient_penalty * self.gradient_weight
            Wasserstein_D = D_real - D_fake
            optimD.step()

        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        fake_g = self.G(features, sentences_G)
        fake = self.D(features, fake_g, one_hot=False)
        fake = torch.mean(fake)
        fake.backward(mone)
        G_cost = -fake
        optimG.step()

        for p in self.D.parameters():
            p.requires_grad = True

        self.sig = nn.LogSigmoid()

        return {"G_loss": G_cost.to("cpu").item(), "D_loss": D_cost.to("cpu").item()}

    def forward(self, batch, optimG, optimD, epoch, iteration):
        # print("=========================")
        sentences = batch['tokenized']
        sentences_G = sentences[:-1]
        sentences_D = sentences[1:]

        # Curriculum learning
        # sen = batch['captions']
        # sen_len = sen.size(0)
        # min_len = min(sen_len, math.ceil(epoch/2))
        # sentences_G = sentences_G[:min_len+1]
        # sentences_D = sentences_D[:min_len]
        # sentences_D = torch.cat((sentences_D, sentences[-1:]), dim=0)

        features = self.encode(batch)

        g_loss = 0
        d_loss = 0

        gen_s = self.G(features, sentences_G)

        # Train Discriminator
        for i in range(self.generator_training):
            optimD.zero_grad()

            real = self.D(features, sentences_D, epoch=epoch+1)
            fake = self.D(features, gen_s, one_hot=False, epoch=epoch+1)

            # gradient_penalty = self.compute_gradient_penalty2(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab).data, gen_s.data)
            gradient_penalty = self.compute_gradient_penalty2(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab).data, gen_s.data)
            # gradient_penalty = self.compute_gradient_penalty(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab), gen_s, real, fake)
            # import sys
            # sys.exit(0)
            print(torch.mean(real))
            print(torch.mean(fake))
            # print(self.sig(torch.mean(fake)))
            # print(self.sig(1-torch.mean(fake)))
            print(gradient_penalty)
            # print(self.compute_gradient_penalty2(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab), gen_s))
            print("---------------------")
            d_loss = -torch.mean(real) + torch.mean(fake) + self.gradient_weight * gradient_penalty
            # d_loss = -self.sig(torch.mean(fake)) - nn.LogSigmoid(1-torch.mean(fake)) + gradient_penalty
            # d_loss = torch.mean(real) - torch.mean(fake) + self.gradient_weight * gradient_penalty

            d_loss.backward(retain_graph=i!=(self.generator_training-1))
            # d_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.clip)
            optimD.step()
        print("========================")

        optimG.zero_grad()
        # if iteration % self.generator_training == 0:
        gen_s = self.G(features, sentences_G)
        fake = self.D(features, gen_s, one_hot=False)
        g_loss = -torch.mean(fake)
        # g_loss = -nn.LogSigmoid(torch.mean(fake))
        # g_loss = torch.mean(fake)
        g_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.clip)
        optimG.step()

        return {"G_loss": g_loss.to("cpu").item(), "D_loss": d_loss.to("cpu").item()}

        # return {"G_loss": g_loss.to("cpu").item(), "D_loss": d_loss.to("cpu").item()}

        # return {"G_loss": g_loss, "D_loss": d_loss.to("cpu").item()}


    def compute_gradient_penalty(self, D, feature, real_samples, fake_samples, realD, fakeD):
        with open('file.txt', 'a') as f:
            # print(real_samples.shape)
            # real [13, 512, 4004]
            # print(fake_samples.shape)
            # fake [13, 512, 4004]
            x = real_samples.view(real_samples.size(1), -1)
            print(x, file=f)
            # print(x.shape)
            # x [512, 13*4004]
            y = fake_samples.view(fake_samples.size(1), -1)
            print(y, file=f)
            # print(y.shape)
            # y [512, 13*4004]
            dist = torch.sqrt(torch.sum((x-y) ** 2, dim=1))
            print(dist, file=f)
            # print(dist.shape)
            # dist [512]
            grad = (realD.view(-1) - fakeD.view(-1)).abs()
            print(grad, file=f)
            # print(grad.shape)
            # grad [512]
            lip = grad / (dist+1e-8)
            print(lip, file=f)

            lip_loss = ((1.0-lip)**2).mean()
            print(lip_loss, file=f)

            return lip_loss



        # print(real_samples.shape)
        # print(fake_samples.shape)
        # dist = torch.sqrt(torch.sum((real_samples-fake_samples) ** 2, dim=1))
        # print(dist.shape)
        # gradient = torch.abs(realD - fakeD)
        # print(realD.shape)
        # print(fakeD.shape)
        # print(gradient.shape)
        # gradient = torch.div(gradient, dist) - 1
        # gradient = torch.pow(gradient, 2)
        # gradient = torch.mean(gradient)
        # return gradient

    def compute_gradient_penalty2(self, D, feature, real_samples, fake_samples):
        with open('file.txt', 'a') as f:
            print("============================================",file=f)
            Tensor = torch.cuda.FloatTensor
            # print(real_samples.shape)
            # print(fake_samples.shape)
            """Calculates the gradient penalty loss for WGAN GP"""
            # Random weight term for interpolation between real and fake samples
            # print(real_samples.size(1))
            alpha = Tensor(np.random.random((1, real_samples.size(1), 1)))
            print(alpha,file=f)
            # print(alpha)
            # print(alpha.shape)
            # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
            print(interpolates,file=f)
            # print(interpolates)
            # print(interpolates.shape)
            d_interpolates = D(feature, interpolates, one_hot=False)
            print(d_interpolates,file=f)
            # print(d_interpolates)
            # print(d_interpolates.shape)
            # print(real_samples.size(1))
            fake = Variable(Tensor(real_samples.size(1), 1).fill_(-1.0), requires_grad=False)
            print(fake,file=f)
            # print(fake)
            # print(fake.shape)
            # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
            # Get gradient w.r.t. interpolates
            gradients = autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                )[0]
            # print(gradients)
            # print(gradients.shape)
            # print(gradients.size(0))
            gradients = gradients.view(gradients.size(1), -1)
            print(gradients,file=f)
            # print(gradients.shape)
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-5)
            print(gradients_norm,file=f)
            # gradients_norm = torch.sqrt(torch.sum((gradients+1e-12) ** 2, dim=1))
            # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty = ((gradients_norm - 1) ** 2).mean()
            print(gradient_penalty,file=f)
            # print(gradient_penalty.shape)
            return gradient_penalty

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
