import numpy as np 

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib

import argparse
import os
import yaml

from data import SimpleDataset
from models.sRB_VAE import sRB_VAE, Conv_sRB_VAE
from models.cVAE import cVAE, Conv_cVAE
from models.MM_cVAE import MM_cVAE, Conv_MM_cVAE
from models.double_InfoGAN import Double_InfoGAN
from torch.utils.data import DataLoader
from data import load_aml, load_epithel
import helper
from data import CelebADataset, CelebADataset2, GridMnistDspriteDataset, CifarMnistDataset

from utils import set_seeds

from celeb_utils import get_synthetic_images
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from metrics import FactorVAEMetricDouble
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


#folder = "./results/mm_cvae_celeba/lightning_logs/"
folder = "./results/mm_cvae_dsprite/lightning_logs/"

#training_name = "version_0/"
training_name = "version_2/"

ckpt = "checkpoints/epoch=99-step=114600.ckpt"
# ckpt = "checkpoints/epoch=99-step=15700.ckpt"


model = Conv_MM_cVAE.load_from_checkpoint(folder + training_name + ckpt, background_disentanglement_penalty=10e3,
        salient_disentanglement_penalty=10e2, in_channels=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)


eval_gridmnist_dsprite_image = np.load("datasets/eval_gridmnist_dsprite_image.npy", allow_pickle=True)
eval_gridmnist_dsprite_label_shape = np.load("datasets/eval_gridmnist_dsprite_label_shape.npy", allow_pickle=True)



transform = None

dataset = GridMnistDspriteDataset(
        eval_gridmnist_dsprite_image,
        labels=eval_gridmnist_dsprite_label_shape, 
        transform=transform,
        in_channels=1
    )

loader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=True)


Z = []
S = []
labels = []

i = 0

for img, label in loader : 
    img = img.to(device)
    z_mu, _, s_mu, _ = model.encode(img)
    Z.extend(z_mu.detach().cpu().numpy())
    labels.extend(label)
    S.extend(s_mu.detach().cpu().numpy())


S = np.array(S)
Z = np.array(Z)
labels = np.array(labels)

print(labels)
print(S)


clf = LogisticRegression()
scores_s = cross_val_score(clf, S, labels, cv=5)
scores_z = cross_val_score(clf, Z, labels, cv=5)
print(scores_s, scores_z)
print('accuracy dsprite shape classif in S : cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
print('accuracy dsprite shape classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))

txt_file = folder + training_name + 'results.txt'

with open(txt_file, 'a') as f:
    f.write('ckpt : ' + ckpt + '\n')
    f.write('D training ? : ' + str(D.training))
    f.write('\n')
    f.write('cross validation \n')
    f.write(str(scores_s) + '\n')
    f.write('accuracy dsprite shape classif in S : cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
    f.write('\n')
    f.write('\n')