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


folder = "./results/double_infogan_dsprite_vf/lightning_logs/"

trainings = [ 
                #"version_1266114/",
                #"version_1266171_wadv025/", 
                #"version_1266194_wiz05/", 
                # "version_1266252_bs256/", 
                # "version_1266259_bs512/", 
                # "version_1266283_bs1024/", 
                "version_1303407/", 
                "version_1303456/", 
                # "version_1306222/",
                # "version_1458924_salientDim10/", 
                # "version_1459366_salientDim10/",
                # "version_1459986_salientDim15/"

]

checkpoints = [
                "checkpoints/epoch=50-step=57999.ckpt",
                "checkpoints/epoch=100-step=114999.ckpt",
                "checkpoints/epoch=150-step=172999.ckpt",
                # "checkpoints/epoch=347-step=49999.ckpt", 
                # "checkpoints/epoch=436-step=249999.ckpt", 
                # "checkpoints/epoch=522-step=149999.ckpt",
                # "checkpoints/epoch=694-step=99999.ckpt"
                ]
# training_name = "version_1253204_inputN_wadv05_wbg05/"
# ckpt = "checkpoints/epoch=130-step=149999.ckpt"

for training_name in trainings : 
    for ckpt in checkpoints : 


        model = Double_InfoGAN.load_from_checkpoint(folder + training_name + ckpt)

        D = model.discriminator

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        D.to(device)


        print('D training ? ', D.training)

        #D.training = False

        print('D training ? ', D.training)

        metric_data_groups = np.load("datasets/metric_data_groups_gridmnist_dsprite.npy", allow_pickle=True)

        metric_data_eval_std = np.load("datasets/metric_data_eval_std.npy", allow_pickle=True) 

        fvaem = FactorVAEMetricDouble(metric_data_groups,metric_data_eval_std, True)

        metric = fvaem.evaluate(D)

        print(metric)

        txt_file = folder + training_name + 'results.txt'

        with open(txt_file, 'a') as f:
            f.write('ckpt : ' + ckpt + '\n')
            f.write('D training ? : ' + str(D.training))
            f.write('\n')
            f.write('fvae : ')
            f.write(str(metric['factorVAE_metric']))
            f.write('\n')
