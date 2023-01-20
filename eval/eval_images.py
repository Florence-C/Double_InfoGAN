import argparse
import os
import yaml

import numpy as np
import pytorch_lightning as pl
import torch
from data import SimpleDataset
from models.sRB_VAE import sRB_VAE, Conv_sRB_VAE
from models.cVAE import cVAE, Conv_cVAE
from models.MM_cVAE import MM_cVAE, Conv_MM_cVAE
from models.double_InfoGAN import Double_InfoGAN
from torch.utils.data import DataLoader
from data import load_aml, load_epithel

from data import CelebADataset, CelebADataset2, GridMnistDspriteDataset

from utils import set_seeds


import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from metrics import FactorVAEMetricDouble

#folder = "./results/double_infogan_dsprite_test/lightning_logs/version_1028285/"
#folder = "./results/double_infogan_dsprite_playingWeights_only/lightning_logs/"

# folder = "./results/double_infogan_celeba_DCGAN_variabilite/lightning_logs/"
# folder = "./results/double_infogan_celeba_DCGAN_noCR/lightning_logs/"

folder = "./results/double_infogan_brats/lightning_logs/"



#training_name = "version_1253204_inputN_wadv05_wbg05/"
# training_name = "version_1389158/"
# training_name = "version_232426/"
training_name = "version_2116075/"

ckpt = "checkpoints/epoch=108-step=29999.ckpt"
# ckpt = "checkpoints/epoch=216-step=59999.ckpt"
# ckpt = "checkpoints/epoch=505-step=139999.ckpt"


#ckpt = "checkpoints/epoch=130-step=149999.ckpt"
#ckpt = "checkpoints/epoch=95-step=14999.ckpt"
#ckpt = "checkpoints/epoch=127-step=19999.ckpt"
# ckpt = "checkpoints/epoch=509-step=79999.ckpt"
#ckpt = "checkpoints/epoch=191-step=29999.ckpt"
#ckpt = "checkpoints/epoch=350-step=54999.ckpt"





model = Double_InfoGAN.load_from_checkpoint(folder + training_name + ckpt)

# G = model.generator

num_img = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

z, s1 = model.sample_latent(num_img)
_, s2 = model.sample_latent(num_img)
_, s3 = model.sample_latent(num_img)
_, s4 = model.sample_latent(num_img)
_, s5 = model.sample_latent(num_img)
_, s6 = model.sample_latent(num_img)
_, s7 = model.sample_latent(num_img)
_, s8 = model.sample_latent(num_img)
_, s9 = model.sample_latent(num_img)
_, s10 = model.sample_latent(num_img)

s_zero = torch.zeros_like(s1)

image_bg = model.generator_step(torch.cat([z,s_zero],dim=1))
image_t1 = model.generator_step(torch.cat([z,s1],dim=1))
image_t2 = model.generator_step(torch.cat([z,s2],dim=1))
image_t3 = model.generator_step(torch.cat([z,s3],dim=1))
image_t4 = model.generator_step(torch.cat([z,s4],dim=1))
image_t5 = model.generator_step(torch.cat([z,s5],dim=1))
image_t6 = model.generator_step(torch.cat([z,s6],dim=1))
image_t7 = model.generator_step(torch.cat([z,s7],dim=1))
image_t8 = model.generator_step(torch.cat([z,s8],dim=1))
image_t9 = model.generator_step(torch.cat([z,s9],dim=1))
image_t10 = model.generator_step(torch.cat([z,s10],dim=1))



print('model training ? ', model.training)

# #G.training = False

# print('G training ? ', .training)

output = torch.cat((image_bg, image_t1, image_t2, image_t3, image_t4, image_t5, image_t6, image_t7, image_t8, image_t9, image_t10), 0)

img_name = folder + training_name + 'test_img.png'

save_image(output.data, img_name, nrow=num_img, normalize=True)


