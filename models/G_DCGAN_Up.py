## Generator model following DCGAN with upsampling layers followed by Conv2d instead of ConvTranspose

import torch.nn as nn
import torch
import functools
#from .utils import get_norm_layer

from .Attention import Attention

class G_Block_DCGAN_Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, upsample=False, stride=1, padding=1):
        super(G_Block_DCGAN_Up, self).__init__()

        # print('norm_layer = ', norm_layer)

        self.upsample = upsample

        self.up = nn.Upsample(scale_factor=2)

        # if spectral_norm :
        #     self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=use_bias))
        # else:
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()


    def forward(self, x):

        if self.upsample: 
            out = self.conv(self.up(x))
        else:
            out = self.conv(x)

        out = self.relu(self.bn(out))
        return out


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.args = args

        attention = args['generator']['attention']

        input_dim = args['background_latent_size'] + args['salient_latent_size']

        channels_list = self.args['generator']['channels']
        #sn = args['generator']['spectral_norm']

        self.init_ch = args['generator']['init_ch']

        self.l1 = nn.Linear(input_dim, channels_list[0]*self.init_ch**2, bias=False)

        self.blocks =  nn.ModuleList() 

        resolution = self.init_ch

        for index in range(len(channels_list)-1):
            resolution *= 2**args['generator']['upsample'][index]

            self.blocks.append(
            G_Block_DCGAN_Up(channels_list[index], channels_list[index+1], 
                                upsample=args['generator']['upsample'][index], stride=1, padding=1))

            print('resolution = ', resolution)

            if attention[resolution]: 
                print("attention module at resolution : ", resolution)
                self.blocks.append(Attention(channels_list[index+1]))

        self.last_conv = nn.Conv2d(channels_list[-1], self.args['in_channels'], kernel_size=3, stride=1, padding=1)

        if self.args["generator"]["act_fn"]=="sigmoid":
            self.act_fn = nn.Sigmoid()
        elif self.args["generator"]["act_fn"]=="tanh":
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError('Activation function [%s] for Generator is not found' % args['generator']['act_fn'])
   


    def forward(self,z):

        out = self.l1(z)

        out = out.view(out.shape[0],-1,self.init_ch, self.init_ch)
        #print(" ")

        for block in self.blocks :

            #print("in shape = ", out.shape)

            out = block(out)
            #print("out shape = ", out.shape)

        out = self.act_fn(self.last_conv(out))

        return out