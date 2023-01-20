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

#from celeb_utils import get_synthetic_images
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from metrics import FactorVAEMetricDouble
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


folder = "./results/double_infogan_celeba/lightning_logs/"


training_list = ["version_232307/",
                "version_232425/", 
                "version_232426/",
                "version_232427/",
                "version_232428/"]

checkpoints = ["checkpoints/epoch=95-step=14999.ckpt",
            "checkpoints/epoch=191-step=29999.ckpt",
            "checkpoints/epoch=318-step=49999.ckpt",
            "checkpoints/epoch=414-step=64999.ckpt",
            "checkpoints/epoch=509-step=79999.ckpt"]


for training_name in training_list : 
    for ckpt in checkpoints :

        model = Double_InfoGAN.load_from_checkpoint(folder + training_name + ckpt)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        model.to(device)

        D = model.discriminator

        total_ids = np.load("datasets/celeba_ids_eval.npy")
        total_labels = np.load("datasets/celeba_labels_eval.npy")


        transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5)
                                       ])


        dataset = CelebADataset2(
                total_ids,
                labels=total_labels, 
                transform=transform)

        loader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=True)


        Z = []
        S = []
        labels = []

        i = 0

        for img, label in loader : 
        	img = img.to(device)
        	_, _, pred_z, pred_s = D(img)
        	Z.extend(pred_z.detach().cpu().numpy())
        	labels.extend(label)
        	S.extend(pred_s.detach().cpu().numpy())


        S = np.array(S)
        Z = np.array(Z)
        labels = np.array(labels)


        clf = LogisticRegression()

        scores_s = cross_val_score(clf, S[labels > 0], labels[labels>0], cv=5)
        scores_z = cross_val_score(clf, Z[labels > 0], labels[labels>0], cv=5)
        print(scores_s, scores_z)
        print('accuracy target classif in S : cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
        print('accuracy target classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))


        txt_file = folder + training_name + 'results.txt'

        with open(txt_file, 'a') as f:
            f.write('ckpt : ' + ckpt + '\n')
            f.write('\n')
            f.write('cross validation \n')
            f.write(str(scores_s) + '\n')
            f.write('accuracy cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
            f.write('\n')
            f.write('accuracy target classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))
            f.write('\n')
            f.write('\n')