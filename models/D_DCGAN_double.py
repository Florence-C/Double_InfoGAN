import torch.nn as nn
import torch
import functools



class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    # def forward(self, x):
    #     if self.training:
    #         return x + torch.empty_like(x).normal_(std=self.std)
    #     else:
    #         return x

    def forward(self, x):
        if self.std ==0 or not self.training: 
            return x 
        else: 
            return x + torch.empty_like(x).normal_(std=self.std)


class D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(D_Block, self).__init__()

        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True))

        self.lrelu = nn.LeakyReLU(0.2)


    def forward(self, x):

        out = self.lrelu(self.conv(x))
        return out




class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args

        self.GaussNoise = GaussianNoise(std=args['discriminator']['std'])

        channels_list = self.args['discriminator']['channels']


        self.blocks =  nn.ModuleList() 

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(self.args['in_channels'], channels_list[0], kernel_size=4, stride=2, padding=1))
       
        self.lrelu = nn.LeakyReLU(0.2)

        #resolution //= 2

        #initial_block = D_Block(self.args['channels'], channels_list[0],norm_layer=self.norm_layer_2D, spectral_norm=sn)
        initial_block = nn.Sequential(self.conv1, self.lrelu)
        self.blocks.append(initial_block)

        for index in range(len(channels_list)-1):
            self.blocks.append(
            D_Block(channels_list[index], channels_list[index+1])
            )


        self.adv_layer = nn.Linear(512 *4*4 , 1)
        self.sigmoid = nn.Sigmoid()

        self.class_layer = nn.Linear(512 *4*4, 1)


        self.c0 = nn.utils.spectral_norm(nn.Linear(512 *4*4, 128, bias=False))
        self.c1 = nn.utils.spectral_norm(nn.Linear(128, args['background_latent_size']))

        self.d0 = nn.utils.spectral_norm(nn.Linear(512 *4*4, 128, bias=False))
        self.d1 = nn.utils.spectral_norm(nn.Linear(128, args['salient_latent_size']))



        
    def forward(self, img):

        out = img 

        for block in self.blocks :
            #print(out.shape)

            out = block(self.GaussNoise(out))
            #print(out.shape)

        out = out.view(out.shape[0], -1)


        validity = self.sigmoid(self.adv_layer(self.GaussNoise(out)))

        classe = self.sigmoid(self.class_layer(self.GaussNoise(out)))

        c_pred = self.lrelu(self.c0(self.GaussNoise(out)))
        d_pred = self.lrelu(self.d0(self.GaussNoise(out)))

        c_pred = self.c1(self.GaussNoise(c_pred))
        d_pred = self.d1(self.GaussNoise(d_pred))

        return validity, classe, c_pred, d_pred

