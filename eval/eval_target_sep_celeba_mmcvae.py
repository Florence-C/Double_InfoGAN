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
import helper
from data import CelebADataset, CelebADataset2, GridMnistDspriteDataset

from utils import set_seeds

from celeb_utils import get_synthetic_images
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

from metrics import FactorVAEMetricDouble

from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

#folder = "./results/mm_cvae_celeba/lightning_logs/"
#folder = "./results/mmcvae/lightning_logs/"
#folder = "./results/mmcvae_celeba_variabilite/lightning_logs/"
#folder = "./results/mmcvae_celeba_200epochs/lightning_logs/"
#folder = "./results/mmcvae_celeba_200epochs_latent_10_90/lightning_logs/"
folder = "./results/mmcvae_celeba_200epochs_latent_10_50/lightning_logs/"

#training_name = "version_0/"
#training_name = "version_1456336_dimS10_dimBG50/"
#training_name = "version_1383303/"
#training_name = "version_1777566/"
#training_name = "version_1793776/"
#training_name = "version_1793775/"
#training_name = "version_1794176/"
training_name = "version_1794177/"

#ckpt = "checkpoints/epoch=6-step=999.ckpt"
#ckpt = "checkpoints/epoch=25-step=3999.ckpt"
#ckpt = "checkpoints/epoch=50-step=7999.ckpt"
ckpt = "checkpoints/epoch=95-step=14999.ckpt"
#ckpt = "checkpoints/epoch=152-step=23999.ckpt"
#ckpt = "checkpoints/epoch=197-step=30999.ckpt"
# ckpt = "checkpoints/epoch=99-step=15700.ckpt"


model = Conv_MM_cVAE.load_from_checkpoint(folder + training_name + ckpt, background_disentanglement_penalty=10e3,
        salient_disentanglement_penalty=10e2, salient_latent_size=10, background_latent_size=50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# total_ids = np.load("datasets/celeba_ids.npy")
# total_labels = np.load("datasets/celeba_labels.npy")

total_ids = np.load("datasets/celeba_ids_eval.npy")
total_labels = np.load("datasets/celeba_labels_eval.npy")


transform=None

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
	# if i < 10 : 
	# 	img_name = folder + training_name + 'eval_' + str(i) + '.png'
	# 	img_0 = img[0]
	# 	np.reshape(64,64,3)
	# 	plt.imsave(img_dir + 'epochs_' + str(epochs) + '_qualitative_results_npreshape.png', img_to_save2)
	# 	save_image(img.data, img_name, normalize=True)
	# 	print(img)
	# 	print(label)
	# i+=1
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


# S_train, S_eval, label_train_s, label_eval_s = train_test_split(S, labels, test_size=0.25, random_state=42)

# log_reg = LogisticRegression().fit(S_train[label_train_s > 0], label_train_s[label_train_s > 0])
# log_reg_score = log_reg.score(S_eval[label_eval_s > 0], label_eval_s[label_eval_s > 0])
# print("Linear probe trained on sepating targets (hat ang glasses), specific latents : ", log_reg_score)


# clf = LogisticRegression()
# scores = cross_val_score(clf, S[labels > 0], labels[labels>0], cv=5)
# print(scores)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


clf = LogisticRegression()

scores_s = cross_val_score(clf, S[labels > 0], labels[labels>0], cv=5)
scores_z = cross_val_score(clf, Z[labels > 0], labels[labels>0], cv=5)

cross_val = cross_validate(clf, Z[labels > 0], labels[labels>0], cv=5, return_train_score=True)
print("cross_val : ", cross_val)
print(cross_val['train_score'], cross_val['test_score'])
print(scores_s, scores_z)
print('accuracy target classif in S : cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
print('accuracy target classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))

txt_file = folder + training_name + 'results.txt'

print(txt_file)

with open(txt_file, 'a') as f:
    f.write('ckpt : ' + ckpt + '\n')
    #f.write('D training ? : ' + str(D.training))
    #f.write('\n')
    #f.write('log reg hat/glasses on salient latent : ')
    #f.write(str(log_reg_score))
    #f.write('\n')
    f.write('cross validation \n')
    f.write(str(scores_s) + '\n')
    f.write('accuracy cross validation 5 folds : ' + str(scores_s.mean()) + ' and standart deviation : ' + str(scores_s.std()))
    f.write('\n')
    f.write('accuracy target classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))
    f.write('\n')
    f.write('\n')