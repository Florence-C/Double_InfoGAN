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
from models.double_InfoGAN_cat import Double_InfoGAN_cat
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
import torch.nn.functional as F


#save_image(torch.Tensor(cifar_mnist_img_test[0]), 'img_0.png', normalize=True)

# matplotlib.pyplot.imsave('img_0_numpy.jpg', cifar_mnist_img_test[0])
# print(cifar_mnist_labels_cifar_test[0])

folder = "./results/double_infogan_cifar_mnist_cat2/lightning_logs/"

training_name = "version_1417709_noImgRecon_noCR_wcat05/"

ckpt = "checkpoints/epoch=306-step=119999.ckpt"

#ckpt = "checkpoints/epoch=101-step=15999.ckpt"
#ckpt = "checkpoints/epoch=509-step=79999.ckpt"
#ckpt = "checkpoints/epoch=999-step=156999.ckpt"
#ckpt = "checkpoints/epoch=1401-step=219999.ckpt"

model = Double_InfoGAN_cat.load_from_checkpoint(folder + training_name + ckpt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)



# def sample_latent(self, batch_size): 

#         if self.config['latent_distribution'] == 'normal' : 
#             z = torch.randn((batch_size, self.background_latent_size), device=self.device)
#             s = torch.randn((batch_size, self.salient_latent_size), device=self.device)
#         elif self.config['latent_distribution'] == 'uniform' : 
#             z = torch.rand((batch_size, self.background_latent_size), device=self.device)
#             s = torch.rand((batch_size, self.salient_latent_size), device=self.device)
#         elif self.config['latent_distribution'] == 'uniform5' : 
#             z = 5*torch.rand((batch_size, self.background_latent_size), device=self.device)
#             s = 5*torch.rand((batch_size, self.salient_latent_size), device=self.device)
#         elif self.config['latent_distribution'] == 'log_normal' : 
#             z = torch.zeros((batch_size, self.background_latent_size), device=self.device)
#             s = torch.zeros((batch_size, self.salient_latent_size), device=self.device)
#             z.log_normal_()
#             s.log_normal_()
#         elif self.config['latent_distribution'] == 'dual' : 
#             z = torch.randn((batch_size, self.background_latent_size), device=self.device)
#             s = torch.rand((batch_size, self.salient_latent_size), device=self.device)
#         else:
#             raise NotImplementedError(' Latent distribution [%s] is not found' % self.config['latent_distribution'])

#         cat = torch.randint(low=0, high=10, size = (batch_size,), device=self.device)
#         s_cat = F.one_hot(cat,num_classes=10)

#         return z, s_cat, s

num_img = 20

z, cat, s = model.sample_latent(num_img)

cat_zero = F.one_hot(cat,num_classes=10)

c0 = torch.zeros((num_img,), dtype=torch.int64, device=device)
c1= torch.ones((num_img,), dtype=torch.int64, device=device)
cat0 = F.one_hot(c0,num_classes=10)
cat1 = F.one_hot(c1,num_classes=10)
cat2 = F.one_hot(2*c1,num_classes=10)
cat3 = F.one_hot(3*c1,num_classes=10)
cat4 = F.one_hot(4*c1,num_classes=10)
cat5 = F.one_hot(5*c1,num_classes=10)

# cat = torch.randint(low=0, high=10, size = (batch_size,), device=self.device)
# #         s_cat = F.one_hot(cat,num_classes=10)

s_zero = torch.zeros_like(s)
cat_zero = torch.zeros_like(cat_zero)

image_bg = model.generator_step(torch.cat([z,cat_zero, s_zero],dim=1))
image_t0 = model.generator_step(torch.cat([z,cat0, s],dim=1))
image_t1 = model.generator_step(torch.cat([z,cat1, s],dim=1))
image_t2 = model.generator_step(torch.cat([z,cat2, s],dim=1))
image_t3 = model.generator_step(torch.cat([z,cat3, s],dim=1))
image_t4 = model.generator_step(torch.cat([z,cat4, s],dim=1))
image_t5 = model.generator_step(torch.cat([z,cat5, s],dim=1))


output = torch.cat((image_bg, image_t0, image_t1, image_t2, image_t3, image_t4, image_t5), 0)

img_name = folder + training_name + 'test_img_cat0-5.png'

save_image(output.data, img_name, nrow=num_img, normalize=True)



D = model.discriminator

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


# Z = []
# S = []
# Cat = []
# labels = []
# labels_mnist = []
# labels_cifar = []

# i = 0

# for img, label, label_mnist, label_cifar in loader : 
#     img = img.to(device)
#     _, _, pred_z, pred_cat, pred_s = D(img)
#     print('label = ', label)
#     print('labels_mnist = ', labels_mnist)
#     print('pred_cat = ', pred_cat)
#     Cat.extend(pred_cat.detach().cpu().numpy())
#     Z.extend(pred_z.detach().cpu().numpy())
#     labels.extend(label)
#     labels_cifar.extend(label_cifar)
#     labels_mnist.extend(label_mnist)
#     S.extend(pred_s.detach().cpu().numpy())


# S = np.array(S)
# Z = np.array(Z)
# Cat = np.array(Cat)
# labels = np.array(labels)
# labels_mnist = np.array(labels_mnist)
# labels_cifar = np.array(labels_cifar)

# print(labels)
# print(S)


# clf = LogisticRegression()
# scores_cat = cross_val_score(clf, Cat[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
# scores_salient = cross_val_score(clf, S[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
# scores_z = cross_val_score(clf, Z[labels_mnist > -1], labels_mnist[labels_mnist>-1], cv=5)
# print(scores_cat, scores_salient)
# print('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_salient.mean()) + ' and standart deviation : ' + str(scores_salient.std()))
# print('accuracy mnist classif in Cat : cross validation 5 folds : ' + str(scores_cat.mean()) + ' and standart deviation : ' + str(scores_cat.std()))
# print('accuracy mnist classif in Z : cross validation 5 folds : ' + str(scores_z.mean()) + ' and standart deviation : ' + str(scores_z.std()))


# txt_file = folder + training_name + 'results.txt'

# with open(txt_file, 'a') as f:
#     f.write('ckpt : ' + ckpt + '\n')
#     f.write('D training ? : ' + str(D.training))
#     f.write('\n')
#     f.write('cross validation \n')
#     f.write(str(scores_salient) + '\n')
#     f.write('accuracy mnist classif in S : cross validation 5 folds : ' + str(scores_salient.mean()) + ' and standart deviation : ' + str(scores_salient.std()))
#     f.write('\n')
#     f.write('\n')