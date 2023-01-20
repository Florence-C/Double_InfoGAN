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


folder = "./results/double_infogan_dsprite_vf/lightning_logs/"

#training_name = "version_1266114/"

#training_name = "version_1458924_salientDim10/"
training_name = "version_1459986_salientDim15/"

ckpt = "checkpoints/epoch=100-step=114999.ckpt"

#ckpt = "checkpoints/epoch=610-step=699999.ckpt"

#ckpt = "checkpoints/epoch=101-step=15999.ckpt"
#ckpt = "checkpoints/epoch=509-step=79999.ckpt"
#ckpt = "checkpoints/epoch=999-step=156999.ckpt"
#ckpt = "checkpoints/epoch=1401-step=219999.ckpt"



model = Double_InfoGAN.load_from_checkpoint(folder + training_name + ckpt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

D = model.discriminator

eval_gridmnist_dsprite_image = np.load("datasets/eval_gridmnist_dsprite_image.npy", allow_pickle=True)
eval_gridmnist_dsprite_label_shape = np.load("datasets/eval_gridmnist_dsprite_label_shape.npy", allow_pickle=True)


transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)
                               ])

dataset = GridMnistDspriteDataset(
        eval_gridmnist_dsprite_image,
        labels=eval_gridmnist_dsprite_label_shape, 
        transform=transform,
    )

loader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=True)


Z = []
S = []
labels = []

i = 0

for img, label in loader : 
    img = img.to(device)
    #print(img)
    _, _, pred_z, pred_s = D(img)
    # if i < 20 :
    #     save_image(torch.tensor(img[0]), 'img_'+str(i) + '_shape'+ str(int(label[0]))+'.jpg', normalize=True)
    Z.extend(pred_z.detach().cpu().numpy())
    labels.extend(label)
    S.extend(pred_s.detach().cpu().numpy())
    i +=1


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