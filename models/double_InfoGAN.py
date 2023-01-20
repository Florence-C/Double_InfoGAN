import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain
from utils import mmd
import os
from .load_model import create_model
from .cr_copy_index import *

from torchvision.utils import save_image, make_grid

#from metrics import FactorVAEMetricDouble

# class Generator(pl.LightningModule):
#     def __init__(self, ???):
#         super(Generator, self).__init__()

# class Discriminator(pl.LightningModule):
#     def __init__(self, ???):
#         super(Discriminator, self).__init__()

class Double_InfoGAN(pl.LightningModule):
    def __init__(self, config=None, save_path='', batch_test=None):
        super(Double_InfoGAN, self).__init__()
        self.save_path = save_path
        self.save_img_path = self.save_path + "imgs/"
        os.makedirs(self.save_img_path, exist_ok=True)
        self.save_linspace_image_path = self.save_img_path + 'linspace/'
        os.makedirs(self.save_linspace_image_path, exist_ok=True)
        self.config = config
        #in_channels = 3
        # dec_channels = 32
        self.in_channels = config['in_channels']
        self.dec_channels = config['dec_channels']
        self.salient_latent_size = config['salient_latent_size']
        self.background_latent_size = config['background_latent_size']

        self.generator, self.discriminator, self.cr_net = create_model(config)

        self.batch_test = batch_test        

        # fixed noise for img display : 
        if batch_test is not None : 
            
            x_test, labels_test = batch_test
            bg_size, _, _, _  = x_test[labels_test == 0].shape

            self.fixed_z_bg, s_bg = self.sample_latent(bg_size)
            self.fixed_s_bg = torch.zeros_like(s_bg)

            self.fixed_z_t, self.fixed_s_t = self.sample_latent(bg_size)


            self.num_img = 10 

            self.z_t_linspace, self.s_t_linspace = self.sample_latent(self.num_img)

            print('x_test = ', x_test.shape)
            print(torch.min(x_test), torch.max(x_test))


       
        self.save_hyperparameters(ignore=["batch_test"])

    def setup(self, stage=None, batch_test=None) : 
        print("in setup !!")
        print("device = ", self.device)


        if batch_test is not None : 

            self.batch_test = batch_test
            
            x_test, labels_test = batch_test
            bg_size, _, _, _  = x_test[labels_test == 0].shape

            self.fixed_z_bg, s_bg = self.sample_latent(bg_size)
            self.fixed_s_bg = torch.zeros_like(s_bg)

            self.fixed_z_t, self.fixed_s_t = self.sample_latent(bg_size)


            self.num_img = 10 

            self.z_t_linspace, self.s_t_linspace = self.sample_latent(self.num_img)



    def discriminator_step(self, x):

        validity, classe, z, s = self.discriminator(x)

        return validity, classe, z, s


    def generator_step(self, latent):

        return self.generator(latent)

    def cr_step(self, image1, image2):

        images = torch.cat([image1, image2], dim=1)

        return self.cr_net(images) 


    def sample_latent(self, batch_size): 

        if self.config['latent_distribution'] == 'normal' : 
            z = torch.randn((batch_size, self.background_latent_size), device=self.device)
            s = torch.randn((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'uniform' : 
            z = torch.rand((batch_size, self.background_latent_size), device=self.device)
            s = torch.rand((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'uniform5' : 
            z = 5*torch.rand((batch_size, self.background_latent_size), device=self.device)
            s = 5*torch.rand((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'log_normal' : 
            z = torch.zeros((batch_size, self.background_latent_size), device=self.device)
            s = torch.zeros((batch_size, self.salient_latent_size), device=self.device)
            z.log_normal_()
            s.log_normal_()
        elif self.config['latent_distribution'] == 'dual' : 
            z = torch.randn((batch_size, self.background_latent_size), device=self.device)
            s = torch.rand((batch_size, self.salient_latent_size), device=self.device)
        else:
            raise NotImplementedError(' Latent distribution [%s] is not found' % self.config['latent_distribution'])

        return z, s

    def sample_latent_cr(self, batch_size, gap) : 

        if self.config['latent_distribution'] == 'normal' : 
            z = torch.randn((batch_size, self.background_latent_size), device=self.device)
            s1 = torch.randn((batch_size, self.salient_latent_size), device=self.device)
            s2 = torch.randn((batch_size, self.salient_latent_size), device=self.device)
            s3 = torch.randn((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'uniform' : 
            z = torch.rand((batch_size, self.background_latent_size), device=self.device)
            s1 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s2 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s3 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'dual' : 
            z = torch.randn((batch_size, self.background_latent_size), device=self.device)
            s1 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s2 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s3 = torch.rand((batch_size, self.salient_latent_size), device=self.device)
        elif self.config['latent_distribution'] == 'uniform5' : 
            z = 5*torch.rand((batch_size, self.background_latent_size), device=self.device)
            s1 = 5*torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s2 = 5*torch.rand((batch_size, self.salient_latent_size), device=self.device)
            s3 = 5*torch.rand((batch_size, self.salient_latent_size), device=self.device)

        rand_index = torch.randint(low=0, high=self.salient_latent_size, size=(batch_size,), device=self.device)

        #rand_index = rand_index.type(LongTensor)

        copy_s1, new_s2 = copy_index(s1,s2, s3,rand_index, gap)

        return z, copy_s1, new_s2, rand_index


    def generate_fake_img(self, n): 

        z,s = self.sample_latent(n)

        fake_img = self.generator_step(torch.cat([z,s],dim=1))

        return fake_img

    def test_reconstruction(self, x):

        _, _ , z_pred, s_pred = self.discriminator_step(x)

        zero_s = torch.zeros_like(s_pred, device=self.device)
        zero_bg = torch.zeros_like(z_pred, device=self.device)

        img_recon = self.generator_step(torch.cat([z_pred, s_pred], dim=1))
        only_bg = self.generator_step(torch.cat([z_pred, zero_s], dim=1))
        only_s = self.generator_step(torch.cat([zero_bg, s_pred], dim=1))
        return img_recon, only_bg, only_s


    def training_step_G(self,x, background, targets): 

        batch_size, _, _, _ = x.shape

        z_bg, s_bg = self.sample_latent(batch_size)
        s_bg = torch.zeros_like(s_bg)

        z_t, s_t = self.sample_latent(batch_size)

        fake_img_bg = self.generator_step(torch.cat([z_bg, s_bg], dim=1))
        fake_img_t = self.generator_step(torch.cat([z_t, s_t], dim=1))

        validity_bg, classe_bg, z_pred_bg, s_pred_bg = self.discriminator_step(fake_img_bg)

        validity_t, classe_t, z_pred_t, s_pred_t = self.discriminator_step(fake_img_t)


        _, _, z_pred_real_bg, s_pred_real_bg = self.discriminator_step(background)
        _, _, z_pred_real_t, s_pred_real_t = self.discriminator_step(targets)

        #img_recon_bg = self.generator_step(torch.cat([z_pred_real_bg, s_pred_real_bg], dim=1))
        s_pred_real_bg_zero = torch.zeros_like(s_pred_real_bg)
        img_recon_bg = self.generator_step(torch.cat([z_pred_real_bg, s_pred_real_bg_zero], dim=1))
        img_recon_t = self.generator_step(torch.cat([z_pred_real_t, s_pred_real_t], dim=1))

        #### Adv GAN Loss
        eps = 1e-8
        g_loss_img_bg = -torch.mean(torch.log(validity_bg + eps))
        g_loss_img_t = -torch.mean(torch.log(validity_t + eps))

        g_adv_loss = self.config['w_bg'] * g_loss_img_bg + self.config['w_t'] * g_loss_img_t

        self.log('g_loss/g_adv_loss', g_adv_loss, prog_bar=False)

        self.log('g_adv_loss/g_loss_img_bg', g_loss_img_bg, prog_bar=False)
        self.log('g_adv_loss/g_loss_img_s', g_loss_img_t, prog_bar=False)
        self.log('g_adv_loss/validity_bg', torch.mean(validity_bg), prog_bar=False)
        self.log('g_adv_loss/validity_t', torch.mean(validity_t), prog_bar=False)


        #### BCE Loss for class
        c_bg = torch.zeros((batch_size,1), device=self.device)
        c_t = torch.ones((batch_size,1), device=self.device)
        g_loss_class_bg = F.binary_cross_entropy(classe_bg, c_bg)
        g_loss_class_t = F.binary_cross_entropy(classe_t, c_t)

        g_class_loss = self.config['w_bg'] * g_loss_class_bg + self.config['w_t'] * g_loss_class_t

        self.log('g_loss/g_class_loss', g_class_loss, prog_bar=False)

        self.log('g_class_loss/g_loss_class_bg', g_loss_class_bg, prog_bar=False)
        self.log('g_class_loss/g_loss_class_t', g_loss_class_t, prog_bar=False)


        #### Info Loss (L1)
        if self.config['infoloss'] == 'gauss' : 
            var = torch.full((z_pred_bg.shape[0], 1), self.config['var'], device=self.device, requires_grad=False)
            g_info_loss_zbg = F.gaussian_nll_loss(z_pred_bg, z_bg, var, full=True, reduction='mean')
            g_info_loss_zt = F.gaussian_nll_loss(z_pred_t, z_t, var, full=True, reduction='mean')
            g_info_loss_st = F.gaussian_nll_loss(s_pred_t, s_t, var, full=True, reduction='mean')

        else : 
            g_info_loss_zbg = F.l1_loss(z_pred_bg, z_bg)
            g_info_loss_zt = F.l1_loss(z_pred_t, z_t)
            g_info_loss_st = F.l1_loss(s_pred_t, s_t)

        g_info_loss_sbg = F.l1_loss(s_pred_bg, s_bg)

        g_info_loss_z = self.config['w_bg'] * g_info_loss_zbg + self.config['w_t'] * g_info_loss_zt 
        g_info_loss_s = self.config['w_bg'] * g_info_loss_sbg + self.config['w_t'] * g_info_loss_st

        g_info_loss = self.config['wi_z']*g_info_loss_z + self.config['wi_s']*g_info_loss_s

        self.log('g_loss/g_info_loss_z', g_info_loss_z, prog_bar=False)
        self.log('g_loss/g_info_loss_s', g_info_loss_s, prog_bar=False)

        self.log('g_info_loss/g_info_loss_zbg', g_info_loss_zbg, prog_bar=False)
        self.log('g_info_loss/g_info_loss_zt', g_info_loss_zt, prog_bar=False)
        self.log('g_info_loss/g_info_loss_sbg', g_info_loss_sbg, prog_bar=False)
        self.log('g_info_loss/g_info_loss_st', g_info_loss_st, prog_bar=False)

        #### Image reconstruction


        g_img_recon_loss_bg = F.l1_loss(img_recon_bg, background)
        g_img_recon_los_t = F.l1_loss(img_recon_t, targets)

        g_img_recon_loss = self.config['w_bg'] * g_img_recon_loss_bg + self.config['w_t'] * g_img_recon_los_t

        self.log('g_loss/g_img_recon_loss', g_img_recon_loss, prog_bar=False)


        ### CR 

        z, s1, s2 , rand_index = self.sample_latent_cr(batch_size, gap=self.config['cr']['gap'])

        img1 = self.generator_step(torch.cat([z, s1], dim=1))

        img2 = self.generator_step(torch.cat([z, s2], dim=1))

        logits = self.cr_step(img1, img2)

        g_cr_loss = F.cross_entropy(logits, rand_index) 

        self.log('g_loss/g_cr_loss', g_cr_loss, prog_bar=True)


        # print(s1[:2])
        # print(s2[:2])
        # print(rand_index[:2])

        # g_img_recon_loss = 0

        # self.log('g_loss/g_img_recon_loss', g_img_recon_loss, prog_bar=False)

        return g_adv_loss, g_class_loss, g_info_loss, g_img_recon_loss, g_cr_loss

    def training_step_CR(self): 

        batch_size = self.config['batch_size']

        z, s1, s2 , rand_index = self.sample_latent_cr(batch_size, gap=self.config['cr']['gap'])

        img1 = self.generator_step(torch.cat([z, s1], dim=1))

        img2 = self.generator_step(torch.cat([z, s2], dim=1))

        logits = self.cr_step(img1, img2)

        cr_loss = F.cross_entropy(logits, rand_index) 

        return cr_loss

    def training_step_D(self, x, background, targets):

        batch_size, _, _, _ = x.shape
        bg_size, _, _, _ = background.shape
        t_size = targets.shape[0]

        z_bg, s_bg = self.sample_latent(batch_size)
        s_bg = torch.zeros_like(s_bg)

        z_t, s_t = self.sample_latent(batch_size)

        fake_img_bg = self.generator_step(torch.cat([z_bg, s_bg], dim=1))
        fake_img_t = self.generator_step(torch.cat([z_t, s_t], dim=1))

        validity_fake_bg, _, z_pred_bg, s_pred_bg = self.discriminator_step(fake_img_bg)
        validity_fake_t, _, z_pred_t, s_pred_t = self.discriminator_step(fake_img_t)

        validity_real_bg, classe_bg, z_pred_real_bg, s_pred_real_bg = self.discriminator_step(background)
        validity_real_t, classe_t, z_pred_real_t, s_pred_real_t = self.discriminator_step(targets)

        #img_recon_bg = self.generator_step(torch.cat([z_pred_real_bg, s_pred_real_bg], dim=1))
        s_pred_real_bg_zero = torch.zeros_like(s_pred_real_bg)
        img_recon_bg = self.generator_step(torch.cat([z_pred_real_bg, s_pred_real_bg_zero], dim=1))
        img_recon_t = self.generator_step(torch.cat([z_pred_real_t, s_pred_real_t], dim=1))

        #### Adv GAN Loss
        eps = 1e-8
        d_real_loss_bg = -torch.mean(torch.log(validity_real_bg + eps))
        d_fake_loss_bg = -torch.mean(torch.log(1 - validity_fake_bg + eps))

        d_real_loss_t = -torch.mean(torch.log(validity_real_t + eps))
        d_fake_loss_t = -torch.mean(torch.log(1 - validity_fake_t + eps))
       
        d_adv_loss = self.config['w_bg']*(d_real_loss_bg + d_fake_loss_bg) +\
                    self.config['w_t'] * (d_real_loss_t + d_fake_loss_t)

        self.log('d_loss/d_adv_loss', d_adv_loss, prog_bar=False)

        self.log('d_adv_loss/d_real_loss_bg', d_real_loss_bg, prog_bar=False)
        self.log('d_adv_loss/d_fake_loss_bg', d_fake_loss_bg, prog_bar=False)
        self.log('d_adv_loss/d_real_loss_t', d_real_loss_t, prog_bar=False)
        self.log('d_adv_loss/d_fake_loss_t', d_fake_loss_t, prog_bar=False)

        self.log('d_adv_loss/validity_real_bg', torch.mean(validity_real_bg), prog_bar=False)
        self.log('d_adv_loss/validity_fake_bg', torch.mean(validity_fake_bg), prog_bar=False)
        self.log('d_adv_loss/validity_real_t', torch.mean(validity_real_t), prog_bar=False)
        self.log('d_adv_loss/validity_fake_t', torch.mean(validity_fake_t), prog_bar=False)

        #### BCE Loss for class
        c_bg = torch.zeros((bg_size,1), device=self.device)
        c_t = torch.ones((t_size,1), device=self.device)
        d_loss_class_bg = F.binary_cross_entropy(classe_bg, c_bg)
        d_loss_class_t = F.binary_cross_entropy(classe_t, c_t)

        d_class_loss = self.config['w_bg'] * d_loss_class_bg + self.config['w_t'] * d_loss_class_t

        self.log('d_loss/d_class_loss', d_class_loss, prog_bar=False)

        self.log('d_class_loss/d_loss_class_bg', d_loss_class_bg, prog_bar=False)
        self.log('d_class_loss/d_loss_class_t', d_loss_class_t, prog_bar=False)


        #### Info Loss (L1)
        if self.config['infoloss'] == 'gauss' : 
            var = torch.full((z_pred_bg.shape[0], 1), self.config['var'], device=self.device, requires_grad=False)
            d_info_loss_zbg = F.gaussian_nll_loss(z_pred_bg, z_bg, var, full=True, reduction='mean')
            d_info_loss_zt = F.gaussian_nll_loss(z_pred_t, z_t, var, full=True, reduction='mean')
            d_info_loss_st = F.gaussian_nll_loss(s_pred_t, s_t, var, full=True, reduction='mean')

        else : 
            d_info_loss_zbg = F.l1_loss(z_pred_bg, z_bg)
            d_info_loss_zt = F.l1_loss(z_pred_t, z_t)
            d_info_loss_st = F.l1_loss(s_pred_t, s_t)

        d_info_loss_sbg = F.l1_loss(s_pred_bg, s_bg)

        # d_info_loss_zbg = F.l1_loss(z_pred_bg, z_bg)
        # d_info_loss_zt = F.l1_loss(z_pred_t, z_t)
        # d_info_loss_sbg = F.l1_loss(s_pred_bg, s_bg)
        # d_info_loss_st = F.l1_loss(s_pred_t, s_t)

        d_info_loss_z = self.config['w_bg']* d_info_loss_zbg + self.config['w_t'] * d_info_loss_zt
        d_info_loss_s = self.config['w_bg'] * d_info_loss_sbg + self.config['w_t'] * d_info_loss_st

        d_info_loss = self.config['wi_z']*d_info_loss_z + self.config['wi_s']*d_info_loss_s


        self.log('d_loss/d_info_loss_z', d_info_loss_z, prog_bar=False)
        self.log('d_loss/d_info_loss_s', d_info_loss_s, prog_bar=False)

        self.log('d_info_loss/d_info_loss_zbg', d_info_loss_zbg, prog_bar=False)
        self.log('d_info_loss/d_info_loss_zt', d_info_loss_zt, prog_bar=False)
        self.log('d_info_loss/d_info_loss_sbg', d_info_loss_sbg, prog_bar=False)
        self.log('d_info_loss/d_info_loss_st', d_info_loss_st, prog_bar=False)

        ### D info loss real image bg : s==0

        s_pred_real_bg_zero = torch.zeros_like(s_pred_real_bg)
        d_info_loss_s_real_bg = F.l1_loss(s_pred_real_bg, s_pred_real_bg_zero)

        self.log('d_loss/d_info_loss_s_real_bg', d_info_loss_s_real_bg, prog_bar=False)


        ### Image reconstruction

        d_img_recon_loss_bg = F.l1_loss(img_recon_bg, background)
        d_img_recon_los_t = F.l1_loss(img_recon_t, targets)

        d_img_recon_loss = self.config['w_bg'] * d_img_recon_loss_bg +\
                           self.config['w_t'] * d_img_recon_los_t

        self.log('d_loss/d_img_recon_loss', d_img_recon_loss, prog_bar=False)


        return d_adv_loss, d_class_loss, d_info_loss, d_img_recon_loss, d_info_loss_s_real_bg

    def save_swaped_image(self, batch_test): 

        x_test, labels_test = batch_test
        background = x_test[labels_test == 0].to(self.device)
        targets = x_test[labels_test != 0].to(self.device)

        _, _ , z_pred_bg, s_pred_bg = self.discriminator_step(background)

        _, _ , z_pred_t, s_pred_t = self.discriminator_step(targets)

        zero_s = torch.zeros_like(s_pred_bg, device=self.device)

        img_recon_bg = self.generator_step(torch.cat([z_pred_bg, s_pred_bg], dim=1))
        img_recon_t = self.generator_step(torch.cat([z_pred_t, s_pred_t], dim=1))

        swap_img_zbg_st = self.generator_step(torch.cat([z_pred_bg, s_pred_t], dim=1))
        swap_img_zt_zeros = self.generator_step(torch.cat([z_pred_t, zero_s], dim=1))

        output = torch.cat((background, targets, img_recon_bg, img_recon_t, swap_img_zbg_st, swap_img_zt_zeros), 0)

        img_name = self.save_img_path + 'swaped_img_epoch_' + str(self.current_epoch) + '.png'

        save_image(output.data, img_name, nrow=64, normalize=True)

    def save_fake_image(self, batch_test): 

        x_test, labels_test = batch_test
        background = x_test[labels_test == 0].to(self.device)
        targets = x_test[labels_test != 0].to(self.device)

        fake_img_bg = self.generator_step(torch.cat([self.fixed_z_bg.to(self.device), self.fixed_s_bg.to(self.device)], dim=1))
        fake_img_t = self.generator_step(torch.cat([self.fixed_z_t.to(self.device), self.fixed_s_t.to(self.device)], dim=1))


        output = torch.cat((background, targets, fake_img_bg, fake_img_t), 0)

        img_name = self.save_img_path + 'fake_img_epoch_' + str(self.current_epoch) + '.png'

        save_image(output.data, img_name, nrow=64, normalize=True)


    def save_linspace_image(self): 

        z_t_r, s_t_r = self.z_t_linspace.repeat_interleave(repeats = 11, dim=0).to(self.device), self.s_t_linspace.repeat_interleave(repeats = 11, dim=0).to(self.device)

        s_bg = torch.zeros_like(s_t_r)

        if self.config["latent_distribution"] == 'uniform' : 
            linspace_line = torch.linspace(0,1,11)
        elif self.config["latent_distribution"] == 'dual' : 
            linspace_line = torch.linspace(0,1,11)
        elif self.config["latent_distribution"] == 'uniform5' : 
            linspace_line = torch.linspace(0,5,11)
        elif self.config["latent_distribution"] == 'normal': 
            linspace_line = torch.linspace(-1.5, 1.5,11)
        elif self.config["latent_distribution"] == 'log_normal': 
            linspace_line = torch.linspace(0,5,11)

        l2 = linspace_line.repeat(self.num_img)

        for i in range(self.salient_latent_size): 
            z_t_r_i = z_t_r.clone().detach() 
            s_t_r_i = s_t_r.clone().detach() 
            s_bg_i = s_bg.clone().detach()
            s_t_r_i[:,i]= l2
            s_bg_i[:,i]=l2
            fake_image_i = self.generator_step(torch.cat([z_t_r_i, s_t_r_i], dim=1))
            fake_image_bg_i = self.generator_step(torch.cat([z_t_r_i, s_bg_i], dim=1))
            img_name_i = self.save_linspace_image_path + 'epoch_'+ str(self.current_epoch) + '_target_fake_img_linspace_' +str(i) + '.png'
            img_name_bg_i = self.save_linspace_image_path + 'epoch_'+ str(self.current_epoch) + '_bg_fake_img_linspace_' +str(i) + '.png'
            save_image(fake_image_i.data, img_name_i, nrow=11, normalize=True)
            save_image(fake_image_bg_i.data, img_name_bg_i, nrow=11, normalize=True)

    def training_step(self, batch, batch_idx, optimizer_idx):

        x, labels = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        if self.current_epoch % self.config['save_img_epoch'] ==0 and batch_idx ==0 and optimizer_idx==0: 
            self.save_swaped_image(self.batch_test)
            self.save_fake_image(self.batch_test)

        if self.current_epoch % self.config['save_img_linspace_epoch'] ==0 and batch_idx ==0 and optimizer_idx==0: 
            self.save_linspace_image()

        # if self.evaluate_fvae and self.current_epoch % self.config['evaluate_fvae_epoch'] ==0 and batch_idx ==0 and optimizer_idx==0: 
        #     metric = self.fvaem.evaluate()
        #     print(metric)

        # print("after save, self.training ? ", self.training)

        #if optimizer_idx ==0 or optimizer_idx ==1 : 
        if optimizer_idx < self.config['loop_g']:

            #print('in loop_g, optimizer_idx : ', optimizer_idx)

            g_adv_loss, g_class_loss, g_info_loss, g_img_recon_loss, g_cr_loss = self.training_step_G(x, background, targets)

            g_loss = self.config['wadv'] * g_adv_loss + self.config['wc']*g_class_loss +\
                         g_info_loss + self.config['wii']*g_img_recon_loss  +\
                         self.config['wcr']*g_cr_loss

            self.log('Loss/g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        #if optimizer_idx == 2: 
        elif optimizer_idx < self.config['loop_g'] + self.config['loop_d']:

            #print('in loop_d, optimizer_idx : ', optimizer_idx)

            d_adv_loss, d_class_loss, d_info_loss, d_img_recon_loss, d_info_loss_s_real_bg = self.training_step_D(x,background, targets)

            d_loss = self.config['wadv'] * d_adv_loss + self.config['wc']*d_class_loss + \
                     d_info_loss + self.config['wii']*d_img_recon_loss + \
                     self.config['wi_real_s']*d_info_loss_s_real_bg

            self.log('Loss/d_loss', d_loss, prog_bar=True)

            return d_loss

        else : 

            #print('in loop_cr, optimizer_idx : ', optimizer_idx)

            cr_loss = self.training_step_CR()

            self.log('Loss/cr_loss', cr_loss, prog_bar=True)

            return cr_loss


    def configure_optimizers(self):
        #lr = 0.0002
        b1 = 0.5
        b2 = 0.999

        params_d = chain(
            # self.common_layers_D.parameters(),
            # self.validity.parameters(),
            # self.classe.parameters(),
            # self.z.parameters(),
            # self.s.parameters()
            self.discriminator.parameters()
            ) 

        params_g = chain(
            self.generator.parameters(), 
            #self.g_fc_1.parameters()
            )

        params_cr = chain(
            self.cr_net.parameters(),
            )

        opt_g = torch.optim.Adam(params_g, lr=self.config['lr_g'], betas=(b1, b2))
        opt_d = torch.optim.Adam(params_d, lr=self.config['lr_d'], betas=(b1, b2))
        opt_cr = torch.optim.Adam(params_cr, lr=self.config['lr_cr'], betas=(b1, b2))

        list_opt = []
        for i in range(self.config['loop_g']): 
            list_opt.append(opt_g)

        for i in range(self.config['loop_d']): 
            list_opt.append(opt_d)

        for i in range(self.config['loop_cr']): 
            list_opt.append(opt_cr)

        print(list_opt)
        return list_opt, []

        #return [opt_g, opt_g, opt_d], []

    # def configure_optimizers(self):

    #     params = chain(
    #         self.z_convs.parameters(),
    #         self.s_convs.parameters(),
    #         self.z_mu.parameters(),
    #         self.z_var.parameters(),
    #         self.s_mu.parameters(),
    #         self.s_var.parameters(),
    #         self.d_fc_1.parameters(),
    #         self.decode_convs.parameters(),
    #     )

    #     opt = torch.optim.Adam(params)
    #     return opt

