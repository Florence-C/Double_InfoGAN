import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
from utils import mmd
import os
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

class MM_cVAE(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            background_latent_size: int,
            salient_latent_size: int,
            background_disentanglement_penalty,
            salient_disentanglement_penalty,
            output_activation=None, 
    ):
        super(MM_cVAE, self).__init__()

        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size

        bias = True
        self.z_h = nn.Linear(input_dim, 400, bias=bias)
        self.z_mu = nn.Linear(400, background_latent_size, bias=bias)
        self.z_var = nn.Linear(400, background_latent_size, bias=bias)

        self.s_h = nn.Linear(input_dim, 400, bias=bias)
        self.s_mu = nn.Linear(400, salient_latent_size, bias=bias)
        self.s_var = nn.Linear(400, salient_latent_size, bias=bias)

        total_latent_size = background_latent_size + salient_latent_size
        self.total_latent_size = total_latent_size

        self.fc3 = nn.Linear(self.total_latent_size, 400, bias=bias)
        self.fc4 = nn.Linear(400, input_dim, bias=bias)

        self.background_disentanglement_penalty = background_disentanglement_penalty
        self.salient_disentanglement_penalty = salient_disentanglement_penalty
        self.output_activation = output_activation

    def encode(self, x):
        hz = F.relu(self.z_h(x))
        hs = F.relu(self.s_h(x))

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(self.fc4(h3))
        else:
            return self.fc4(h3)

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s

    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)

        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)

        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def embed_shared(self, x):
        mu_z, _, _, _ = self.encode(x)
        return mu_z

    def embed_salient(self, x):
        _, _, mu_s, _ = self.encode(x)
        return mu_s

    def training_step(self, batch, batch_idx):
        x, labels = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

        # Corresponds to "our" implementation

        gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        background_mmd_loss = self.background_disentanglement_penalty * mmd(z_bg, z_tar, gammas=gammas, device=self.device)
        salient_mmd_loss = self.salient_disentanglement_penalty*mmd(s_bg, torch.zeros_like(s_bg), gammas=gammas, device=self.device)
        loss += background_mmd_loss + salient_mmd_loss

        self.log('background_mmd_loss', background_mmd_loss, prog_bar=True)
        self.log('salient_mmd_loss', salient_mmd_loss, prog_bar=True)
        self.log('KLD_z_bg', KLD_z_bg, prog_bar=True)
        self.log('KLD_z_tar', KLD_z_tar, prog_bar=True)
        self.log('KLD_s_tar', KLD_s_tar, prog_bar=True)

        self.log('MSE_bg', MSE_bg, prog_bar=True)
        self.log('MSE_tar', MSE_tar, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_h.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.s_h.parameters(),
            self.s_mu.parameters(),
            self.s_var.parameters(),
            self.fc3.parameters(),
            self.fc4.parameters()
        )

        opt = torch.optim.Adam(params)
        return opt


class Conv_MM_cVAE(pl.LightningModule):
    def __init__(self, background_disentanglement_penalty, salient_disentanglement_penalty,in_channels=3, save_path='', salient_latent_size=6, background_latent_size=16, batch_test=None, save_img_epoch=100, config=None):
        super(Conv_MM_cVAE, self).__init__()
        self.save_img_path = save_path + "imgs/"
        os.makedirs(self.save_img_path, exist_ok=True)
        dec_channels = 32
        # salient_latent_size = salient_latent_size
        # background_latent_size = 16
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        bias = False
        self.save_img_epoch = save_img_epoch

        self.z_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),


            nn.Conv2d(dec_channels, dec_channels * 2,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels*2),


            nn.Conv2d(dec_channels * 2, dec_channels * 4,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.s_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.Conv2d(dec_channels, dec_channels * 2,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.Conv2d(dec_channels * 2, dec_channels * 4,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8,
                      kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.z_mu = nn.Linear(dec_channels * 8 * 4 * 4, background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 4 * 4, background_latent_size, bias=bias)

        self.s_mu = nn.Linear(dec_channels * 8 * 4 * 4, salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 4 * 4, salient_latent_size, bias=bias)

        self.decode_convs = nn.Sequential(
            nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.ConvTranspose2d(dec_channels * 2, dec_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.ConvTranspose2d(dec_channels, in_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.Sigmoid()
        )

        total_latent_size = salient_latent_size + background_latent_size

        self.background_disentanglement_penalty = background_disentanglement_penalty
        self.salient_disentanglement_penalty = salient_disentanglement_penalty

        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 4 * 4)

        self.batch_test = batch_test

        self.save_hyperparameters(ignore=["batch_test"])

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        hz = self.z_convs(x)
        hs = self.s_convs(x)

        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)
        hs = hs.view(-1, self.dec_channels * 8 * 4 * 4)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def decode(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)

        return self.decode_convs(z)

    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)
        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s


    def test_reconstruction(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        # z = self.reparameterize(mu_z, logvar_z)
        # s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)
        zero_bg = torch.zeros(x.shape[0],self.background_latent_size).to(self.device)
        img_recon = self.decode(torch.cat([mu_z, mu_s], dim=1))
        only_bg = self.decode(torch.cat([mu_z, salient_var_vector], dim=1))
        only_s = self.decode(torch.cat([zero_bg, mu_s], dim=1))
        return img_recon, only_bg, only_s

    def save_swaped_image(self, batch_test):

        x, labels = batch_test
        background = x[labels == 0].to(self.device)
        targets = x[labels != 0].to(self.device)

        mu_z_bg, _, mu_s_bg, _ = self.encode(background)
        mu_z_t, _, mu_s_t, _ = self.encode(targets)

        print('min max background = ', torch.min(background), torch.max(background))
        print('min max targets = ', torch.min(targets), torch.max(targets))


        img_recon_bg = self.decode(torch.cat([mu_z_bg, mu_s_bg], dim=1))
        img_recon_t = self.decode(torch.cat([mu_z_t, mu_s_t], dim=1))

        print('min max recon background = ', torch.min(img_recon_bg), torch.max(img_recon_bg))
        print('min max recon targets = ', torch.min(img_recon_t), torch.max(img_recon_t))

        salient_var_vector = torch.zeros_like(mu_s_bg)

        swap_img_zbg_st = self.decode(torch.cat([mu_z_bg, mu_s_t], dim=1))
        swap_img_zt_zeros = self.decode(torch.cat([mu_z_t, salient_var_vector], dim=1))

        output = torch.cat((background, targets, img_recon_bg, img_recon_t, swap_img_zbg_st, swap_img_zt_zeros), 0)

        img_name = self.save_img_path + 'sepochs_' + str(self.current_epoch) + '_img_swap.png'

        reshape_background = background.detach().cpu().numpy()
        reshape_background = reshape_background.reshape(64,64,64,self.in_channels).astype('float32')

        reshape_targets = targets.detach().cpu().numpy()
        reshape_targets = reshape_targets.reshape(64,64,64,self.in_channels).astype('float32')

        reshape_img_recon_bg = img_recon_bg.detach().cpu().numpy()
        reshape_img_recon_bg = reshape_img_recon_bg.reshape(64,64,64,self.in_channels).astype('float32')

        reshape_img_recon_t = img_recon_t.detach().cpu().numpy()
        reshape_img_recon_t = reshape_img_recon_t.reshape(64,64,64,self.in_channels).astype('float32')

        reshape_swap_img_zbg_st = swap_img_zbg_st.detach().cpu().numpy()
        reshape_swap_img_zbg_st = reshape_swap_img_zbg_st.reshape(64,64,64,self.in_channels).astype('float32')

        reshape_swap_img_zt_zeros = swap_img_zt_zeros.detach().cpu().numpy()
        reshape_swap_img_zt_zeros = reshape_swap_img_zt_zeros.reshape(64,64,64,self.in_channels).astype('float32')

        img_to_save2 = np.zeros((64*6,64*64,self.in_channels))
        for i in range(64): 
            img_to_save2[0:64,64*i:64*(i+1),:] = reshape_background[i]
            img_to_save2[64:128,64*i:64*(i+1),:] = reshape_targets[i]
            img_to_save2[128:192,64*i:64*(i+1),:] = reshape_img_recon_bg[i]
            img_to_save2[192:256,64*i:64*(i+1),:] = reshape_img_recon_t[i]
            img_to_save2[256:320,64*i:64*(i+1),:] = reshape_swap_img_zbg_st[i]
            img_to_save2[320:384,64*i:64*(i+1),:] = reshape_swap_img_zt_zeros[i]

        plt.imsave(img_name, img_to_save2)

        return background, targets, img_recon_bg, img_recon_t, swap_img_zbg_st, swap_img_zt_zeros


    def embed_shared(self, x):
        mu_z, _, _, _ = self.encode(x)
        return mu_z

    def embed_salient(self, x):
        _, _, mu_s, _ = self.encode(x)
        return mu_s

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s

    def training_step(self, batch, batch_idx):
        x, labels = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        if self.current_epoch % self.save_img_epoch ==0 and batch_idx ==0:
            self.save_swaped_image(self.batch_test)

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        # if batch_idx ==10 : 
        #     img_0 = background[0]
        #     num_images_bg = recon_batch_tar.shape[0]
        #     img_bg = torch.cat((background, recon_batch_bg), dim=0)
        #     grid_bg = make_grid(img_bg, nrow=2)

        #     save_image(grid_bg, str(self.current_epoch) + '_bg_recon.png', normalize=True)


        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

        gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        background_mmd_loss = self.background_disentanglement_penalty * mmd(z_bg, z_tar, gammas=gammas, device=self.device)
        salient_mmd_loss = self.salient_disentanglement_penalty*mmd(s_bg, torch.zeros_like(s_bg), gammas=gammas, device=self.device)
        loss += background_mmd_loss + salient_mmd_loss


        # print('*** LOSS ***')
        # print('KLD_z_bg : ', KLD_z_bg)
        # print('KLD_z_tar : ', KLD_z_tar)
        # print('KLD_s_tar : ', KLD_s_tar)
        # print('MSE_bg : ', MSE_bg)
        # print('MSE_tar : ', MSE_tar)
        # print('background_mmd_loss : ', background_mmd_loss)
        # print('salient_mmd_loss', salient_mmd_loss)
        # print("******")

        self.log('background_mmd_loss', background_mmd_loss, prog_bar=True)
        self.log('salient_mmd_loss', salient_mmd_loss, prog_bar=True)


        self.log('MSE_bg', MSE_bg, prog_bar=True)
        self.log('MSE_tar', MSE_tar, prog_bar=True)
        self.log('KLD_z_bg', KLD_z_bg, prog_bar=True)
        self.log('KLD_z_tar', KLD_z_tar, prog_bar=True)
        self.log('KLD_s_tar', KLD_s_tar, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_convs.parameters(),
            self.s_convs.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.s_mu.parameters(),
            self.s_var.parameters(),
            self.d_fc_1.parameters(),
            self.decode_convs.parameters(),
        )

        opt = torch.optim.Adam(params)
        return opt
