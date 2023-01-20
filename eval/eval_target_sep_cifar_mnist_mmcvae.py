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

from data import CelebADataset, CelebADataset2, GridMnistDspriteDataset, CifarMnistDataset

from utils import set_seeds

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from metrics import FactorVAEMetricDouble

from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

folder = "./results/mmcvae_cifar_latent/lightning_logs/"


trainings = [
        "version_240538/", 
        "version_240539/", 
        "version_240540/",
        "version_240541/",
        "version_240549/",
        ]

checkpoints = ["checkpoints/epoch=51-step=19999.ckpt",
        "checkpoints/epoch=102-step=39999.ckpt",
        "checkpoints/epoch=204-step=79999.ckpt",
        "checkpoints/epoch=306-step=119999.ckpt",
        "checkpoints/epoch=396-step=154999.ckpt",
        "checkpoints/epoch=498-step=194999.ckpt"
        ]

for training_name in trainings : 
        for ckpt in checkpoints : 

                model = Conv_MM_cVAE.load_from_checkpoint(folder + training_name + ckpt, background_disentanglement_penalty=10e3,
                        salient_disentanglement_penalty=10e2)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                model.to(device)

                cifar_mnist_labels_cifar_eval = np.load("datasets/cifar_mnist_labels_cifar_eval.npy", allow_pickle=True)
                cifar_mnist_labels_mnist_eval = np.load("datasets/cifar_mnist_labels_mnist_eval.npy", allow_pickle=True)
                cifar_mnist_labels_eval = np.load("datasets/cifar_mnist_labels_eval.npy", allow_pickle=True)
                cifar_mnist_img_eval = np.load("datasets/cifar_mnist_img_eval.npy", allow_pickle=True)


                transform=None

                in_channels = 3

                dataset = CifarMnistDataset(
                        cifar_mnist_img_eval,
                        labels=cifar_mnist_labels_eval, 
                        label_mnist = cifar_mnist_labels_mnist_eval,
                        label_cifar = cifar_mnist_labels_cifar_eval,
                        transform=transform,
                        in_channels=in_channels
                        )

                loader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=True)


                Z = []
                S = []
                labels = []
                labels_mnist = []
                labels_cifar = []


                i = 0

                for img, label, label_mnist, label_cifar in loader :  
                        img = img.to(device)
                        z_mu, _, s_mu, _ = model.encode(img)
                        Z.extend(z_mu.detach().cpu().numpy())
                        labels.extend(label)
                        labels_cifar.extend(label_cifar)
                        labels_mnist.extend(label_mnist)
                        S.extend(s_mu.detach().cpu().numpy())


                S = np.array(S)
                Z = np.array(Z)
                labels = np.array(labels)
                labels_mnist = np.array(labels_mnist)
                labels_cifar = np.array(labels_cifar)

                print(labels)
                print(S)


                clf = LogisticRegression(max_iter=1000)
                scores_s_mnist = cross_val_score(clf, S[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
                scores_z_mnist = cross_val_score(clf, Z[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
                scores_s_cifar = cross_val_score(clf, S, labels_cifar, cv=5)
                scores_z_cifar = cross_val_score(clf, Z, labels_cifar, cv=5)
                print('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_s_mnist.mean()) + ' and standart deviation : ' + str(scores_s_mnist.std()))
                print('accuracy mnist classif in Z : cross validation 5 folds : ' + str(scores_z_mnist.mean()) + ' and standart deviation : ' + str(scores_z_mnist.std()))
                print('accuracy cifar classif in S : cross validation 5 folds : ' + str(scores_s_cifar.mean()) + ' and standart deviation : ' + str(scores_s_cifar.std()))
                print('accuracy cifar classif in Z : cross validation 5 folds : ' + str(scores_z_cifar.mean()) + ' and standart deviation : ' + str(scores_z_cifar.std()))
             

                txt_file = folder + training_name + 'results.txt'

                with open(txt_file, 'a') as f:
                    f.write('ckpt : ' + ckpt + '\n')
                    f.write('\n')
                    f.write('cross validation \n')
                    f.write('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_s_mnist.mean()) + ' and standart deviation : ' + str(scores_s_mnist.std()) + '\n')
                    f.write('accuracy mnist classif in Z : cross validation 5 folds : ' + str(scores_z_mnist.mean()) + ' and standart deviation : ' + str(scores_z_mnist.std()) + '\n')
                    f.write('accuracy cifar classif in S : cross validation 5 folds : ' + str(scores_s_cifar.mean()) + ' and standart deviation : ' + str(scores_s_cifar.std()) + '\n')
                    f.write('accuracy cifar classif in Z : cross validation 5 folds : ' + str(scores_z_cifar.mean()) + ' and standart deviation : ' + str(scores_z_cifar.std()) + '\n')
                    f.write('\n')
                    f.write('\n')

