import numpy as np
import torch
import torch.autograd as autograd

from . import onehot_batch_data
from .discriminator import Discriminator
from .generator import Generator
from torch import nn
from torch.autograd import Variable

class WGAN(nn.Module):

    def __init__(self, n_vocab, config_model):
        super(WGAN, self).__init__()
        self.n_trg_vocab = n_vocab

        emb_dim = config_model['emb_dim']
        dec_dim = config_model['dec_dim']

        self.gradient_weight = config_model['gradient_weight']
        self.generator_training = config_model['generator']['train_iteration']
        feature_size = config_model['feature_size']

        self.emb_G = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.emb_D = nn.Embedding(n_vocab, emb_dim, padding_idx=0)

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
        # print("=========================")
        sentences = batch['tokenized']
        sentences_G = sentences[:-1]
        sentences_D = sentences[1:]

        features = self.encode(batch)

        g_loss = 0
        d_loss = 0

        # Train Discriminator
        optimD.zero_grad()
        gen_s = self.G(features, sentences_G)
        real = self.D(features, sentences_D, epoch=epoch+1)
        fake = self.D(features, gen_s, one_hot=False, epoch=epoch+1)

        gradient_penalty = self.compute_gradient_penalty(self.D, features, onehot_batch_data(sentences_D, self.n_trg_vocab), gen_s)
        if epoch > 30:
            print(gradient_penalty)
            print(batch['feats'])
        d_loss = -torch.mean(real) + torch.mean(fake) + self.gradient_weight * gradient_penalty

        # print(d_loss.grad_fn)
        
        # print("\t{0}".format(d_loss.grad_fn.next_functions[0][0]))
        # print("\t\t{0}".format(d_loss.grad_fn.next_functions[0][0].next_functions[0][0]))
        # print("\t\t{0}".format(d_loss.grad_fn.next_functions[0][0].next_functions[1][0]))
        
        # print("\t{0}".format(d_loss.grad_fn.next_functions[1][0]))
        # print("\t\t{0}".format(d_loss.grad_fn.next_functions[1][0].next_functions[0][0]))
        # print("\t\t{0}".format(d_loss.grad_fn.next_functions[1][0].next_functions[1][0]))

        # import sys
        # sys.exit()

        d_loss.backward()
        clip = 1.0
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), clip)
        optimD.step()
        
        if torch.isnan(d_loss):
            print(real)
            print(torch.isnan(real))
            print(fake)
            print(torch.isnan(fake))
            print(gradient_penalty)
            print(torch.isnan(gradient_penalty))
            print(torch.mean(real))
            print(torch.isnan(torch.mean(real)))
            print(torch.mean(fake))
            print(torch.isnan(torch.mean(fake)))
            import sys
            sys.exit()

        optimG.zero_grad()

        if iteration % self.generator_training == 0:
            # Train Generator
            gen_s = self.G(features, sentences_G)
            fake = self.D(features, gen_s, one_hot=False)
            g_loss = -torch.mean(fake)
            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.G.parameters(), clip)
            optimG.step()

            return {"G_loss": g_loss.to("cpu").item(), "D_loss": d_loss.to("cpu").item()}

        return {"G_loss": g_loss, "D_loss": d_loss.to("cpu").item()}
        

        
    def compute_gradient_penalty(self, D, feature, real_samples, fake_samples):
        Tensor = torch.cuda.FloatTensor
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((1, real_samples.size(1), 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(feature, interpolates, one_hot=False)
        fake = Variable(Tensor(real_samples.size(1), 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
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