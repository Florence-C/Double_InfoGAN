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


# folder = "./results/double_infogan_cifar_mnist/lightning_logs/"
# folder = "./results/double_infogan_cifar_mnist_latent_64_64/lightning_logs/"
# folder = "./results/double_infogan_cifar_mnist_latent_100_100/lightning_logs/"
folder = "./results/double_infogan_cifar_mnist_latent_64_64_noCR/lightning_logs/"


trainings = [
                "version_265653/",
                "version_265654/",
                "version_265655/",
                "version_265656/",
                "version_265657/"
]

# trainings = [
#             #"version_238583/",
#             #"version_240570/",
#             #"version_240571/",
#             #"version_240572/",
#             #"version_240573/",
#             #"version_240603/",
#             #"version_240605/",
#             #"version_240606/",
#             #"version_240607/",
#             #"version_240608/",
#             "version_240614/",
#             "version_240615/",
#             "version_240616/",
#             "version_240617/",
#             "version_240618/",
#             #"version_240645/",
#             #"version_240646/",
#             #"version_240647/",
#             #"version_240648/",
#             #"version_240649/"
#             ]

# trainings = [
#             "version_240684/",
#             "version_240685/",
#             "version_240686/",
#             "version_240687/",
#             "version_240688/"
#             # "version_240690/",
#             # "version_240691/",
#             # "version_240692/",
#             # "version_240693/",
#             # "version_240694/",
#             # "version_240702/",
#             # "version_240703/",
#             # "version_240704/",
#             # "version_240705/",
#             # "version_240706/",
#             # "version_240733/",
#             # "version_240734/",
#             # "version_240736/",
#             # "version_240737/",
#             # "version_240738/"
#             ]

checkpoints = [
        "checkpoints/epoch=51-step=19999.ckpt",
        "checkpoints/epoch=102-step=39999.ckpt",
        "checkpoints/epoch=204-step=79999.ckpt",
        "checkpoints/epoch=306-step=119999.ckpt",
        "checkpoints/epoch=396-step=154999.ckpt",
        "checkpoints/epoch=498-step=194999.ckpt"
        ]


for training_name in trainings: 

    for ckpt in checkpoints : 


        model = Double_InfoGAN.load_from_checkpoint(folder + training_name + ckpt)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        model.to(device)

        D = model.discriminator


        D.training = False

        cifar_mnist_labels_cifar_eval = np.load("datasets/cifar_mnist_labels_cifar_eval.npy", allow_pickle=True)
        cifar_mnist_labels_mnist_eval = np.load("datasets/cifar_mnist_labels_mnist_eval.npy", allow_pickle=True)
        cifar_mnist_labels_eval = np.load("datasets/cifar_mnist_labels_eval.npy", allow_pickle=True)
        cifar_mnist_img_eval = np.load("datasets/cifar_mnist_img_eval.npy", allow_pickle=True)



        transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5)
                                       ])
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
        I = []

        i = 0

        for img, label, label_mnist, label_cifar in loader : 
            img = img.to(device)
            _, _, pred_z, pred_s = D(img)
            Z.extend(pred_z.detach().cpu().numpy())
            labels.extend(label)
            labels_cifar.extend(label_cifar)
            labels_mnist.extend(label_mnist)
            S.extend(pred_s.detach().cpu().numpy())
            I.extend(img.cpu().numpy())


        # labels contains 0 for background and 1 for target
        # labels_mnist contains -1 for background and mnist label for target
        # labels_cifar contains cifar labels

        S = np.array(S)
        Z = np.array(Z)
        labels = np.array(labels)
        labels_mnist = np.array(labels_mnist)
        labels_cifar = np.array(labels_cifar)
        I = np.array(I)

        # print(labels)
        # print(S)


        clf = LogisticRegression()
        scores_s_mnist = cross_val_score(clf, S[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
        scores_z_mnist = cross_val_score(clf, Z[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
        scores_s_cifar = cross_val_score(clf, S, labels_cifar, cv=5)
        scores_z_cifar = cross_val_score(clf, Z, labels_cifar, cv=5)
        #scrores_i_mnist = cross_val_score(clf, I[labels>0], labels_mnist[labels>0])

        print('training : ', folder + training_name + ckpt)

        print('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_s_mnist.mean()) + ' and standart deviation : ' + str(scores_s_mnist.std()))
        print('accuracy mnist classif in Z : cross validation 5 folds : ' + str(scores_z_mnist.mean()) + ' and standart deviation : ' + str(scores_z_mnist.std()))
        print('accuracy cifar classif in S : cross validation 5 folds : ' + str(scores_s_cifar.mean()) + ' and standart deviation : ' + str(scores_s_cifar.std()))
        print('accuracy cifar classif in Z : cross validation 5 folds : ' + str(scores_z_cifar.mean()) + ' and standart deviation : ' + str(scores_z_cifar.std()))
        #print('accuracy cifar classif in image target : cross validation 5 folds : ' + str(scrores_i_mnist.mean()) + ' and standart deviation : ' + str(scrores_i_mnist.std()))


        txt_file = folder + training_name + 'results.txt'

        with open(txt_file, 'a') as f:
            f.write('ckpt : ' + ckpt + '\n')
            f.write('D training ? : ' + str(D.training))
            f.write('\n')
            f.write('cross validation \n')
            #f.write(str(scores_s) + '\n')
            f.write('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_s_mnist.mean()) + ' and standart deviation : ' + str(scores_s_mnist.std()) + '\n')
            f.write('accuracy mnist classif in Z : cross validation 5 folds : ' + str(scores_z_mnist.mean()) + ' and standart deviation : ' + str(scores_z_mnist.std()) + '\n')
            f.write('accuracy cifar classif in S : cross validation 5 folds : ' + str(scores_s_cifar.mean()) + ' and standart deviation : ' + str(scores_s_cifar.std()) + '\n')
            f.write('accuracy cifar classif in Z : cross validation 5 folds : ' + str(scores_z_cifar.mean()) + ' and standart deviation : ' + str(scores_z_cifar.std()) + '\n')
            f.write('\n')
            f.write('\n')