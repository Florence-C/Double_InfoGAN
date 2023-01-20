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
from models.MM_cVAE_Robin import Conv_MM_cVAE_Robin
from models.double_InfoGAN import Double_InfoGAN
from models.double_InfoGAN_cat import Double_InfoGAN_cat
from models.double_Info_VAE_GAN import Double_Info_VAE_GAN
from torch.utils.data import DataLoader
from data import load_aml, load_epithel
from data import CelebADataset, CelebADataset2, GridMnistDspriteDataset, CifarMnistDataset, BratsDataset
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import set_seeds

import random

from celeb_utils import get_synthetic_images
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
#parser.add_argument('dataset', type=str, choices=['Epithel', 'CelebA', 'AML', 'Dsprite'])
#parser.add_argument('model', type=str, choices=['mm_cvae', 'cvae', 'srb_vae', 'double_infogan'])
#parser.add_argument('n_epoch', type=int, default=10)
parser.add_argument('--config_file', type=str)
parser.add_argument('--logdir', type=str)
parser.add_argument('--load_from_ckpt', action='store_true')
parser.add_argument('--ckpt', type=str, default='none')

args = parser.parse_args()

print('args = ', args)

yaml_file = open(args.config_file, 'r')
config = yaml.safe_load(yaml_file)

print('config = ', config)

if config["seed"] == "None":
    seed = random.randint(1,1000)
    config["seed"] = seed
else:
    seed = config["seed"]

set_seeds(seed)

print('seed = ', seed)

epochs = config['n_epoch']

transform = None

print(config['dataset'])

# if config['dataset'] == 'Epithel':
#     (data1, labels1), (data2, labels2), (data3, labels3) = load_epithel()

#     data_total = np.concatenate([data1, data2, data3])
#     labels_total = np.concatenate([labels1, labels2, labels3])

#     dataset = SimpleDataset(X=data_total, y=labels_total)
    
#     trainer = pl.Trainer(max_epochs=epochs, gpus=[0])
#     loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

#     if config['model'] == 'srb_vae':
#         model = sRB_VAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5
#         )
#     elif config['model'] == 'cvae':
#         model = cVAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5,
#         )

#     elif config['model'] == 'mm_cvae':
#         model = MM_cVAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5,
#             background_disentanglement_penalty=10e3,
#             salient_disentanglement_penalty=10e2
#         )

#     trainer.fit(model, loader)

# elif config['dataset'] == 'AML':
#     (data1, labels1), (data2, labels2), (data3, labels3) = load_aml()

#     data_total = np.concatenate([data1, data2, data3])
#     labels_total = np.concatenate([labels1, labels2, labels3])

#     dataset = SimpleDataset(X=data_total, y=labels_total)
#     #epochs = 100
#     trainer = pl.Trainer(max_epochs=epochs, gpus=[0])
#     loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

#     if config['model'] == 'srb_vae':
#         model = sRB_VAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5
#         )
#     elif config['model'] == 'cvae':
#         model = cVAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5
#         )

#     elif config['model'] == 'mm_cvae':
#         model = MM_cVAE(
#             input_dim=data1.shape[1],
#             background_latent_size=10,
#             salient_latent_size=5,
#             background_disentanglement_penalty=10e3,
#             salient_disentanglement_penalty=10e2
#         )


#     trainer.fit(model, loader)

if config['dataset'] == 'celeba':

    total_ids = np.load("datasets/celeba_ids.npy")
    total_labels = np.load("datasets/celeba_labels.npy")

    total_ids_test = np.load("datasets/celeba_ids_test.npy")
    total_labels_test = np.load("datasets/celeba_labels_test.npy")

    if config['model'] == "mm_cvae" or config['model'] == 'cvae' : 
        transform = None
    elif config['generator']['act_fn'] == 'sigmoid' : 
        transform = transforms.ToTensor()
    else: 
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)
                               ])


    dataset = CelebADataset2(
        total_ids,
        labels=total_labels, 
        transform=transform
    )

    testset = CelebADataset2(
        total_ids_test,
        labels=total_labels_test, 
        transform=transform
    )

    in_channels = 3

    metric_data_groups=None
    metric_data_eval_std=None

elif config['dataset'] == 'celeba_smile_glasses':

    total_ids = np.load("datasets/celeba_smile_glasses_ids_train.npy")
    total_labels = np.load("datasets/celeba_smile_glasses_labels_train.npy")

    total_ids_test = np.load("datasets/celeba_smile_glasses_ids_test.npy")
    total_labels_test = np.load("datasets/celeba_smile_glasses_labels_test.npy")

    if config['model'] == "mm_cvae": 
        transform = None
    elif config['generator']['act_fn'] == 'sigmoid' : 
        transform = transforms.ToTensor()
    else: 
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)
                               ])


    dataset = CelebADataset2(
        total_ids,
        labels=total_labels, 
        transform=transform
    )

    testset = CelebADataset2(
        total_ids_test,
        labels=total_labels_test, 
        transform=transform
    )

    in_channels = 3

    metric_data_groups=None
    metric_data_eval_std=None

elif config['dataset'] == 'cifar_mnist':

    total_ids = np.load("datasets/cifar_mnist_img_train.npy")
    total_labels = np.load("datasets/cifar_mnist_labels_train.npy")

    total_ids_test = np.load("datasets/cifar_mnist_img_test.npy")
    total_labels_test = np.load("datasets/cifar_mnist_labels_test.npy")

    if "mm_cvae" in config['model']: 
        transform = None
    elif config['generator']['act_fn'] == 'sigmoid' : 
        transform = transforms.ToTensor()
    else: 
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)
                               ])

    in_channels = 3

    dataset = CifarMnistDataset(
        total_ids,
        labels=total_labels, 
        transform=transform, 
        in_channels = in_channels
    )

    testset = CifarMnistDataset(
        total_ids_test,
        labels=total_labels_test, 
        transform=transform, 
        in_channels=in_channels
    )

    

    metric_data_groups=None
    metric_data_eval_std=None
   

elif config['dataset'] == 'dsprite':

    total_ids_train = np.load("datasets/gridmnist_dsprite_train_ids.npy")
    total_labels_train = np.load("datasets/gridmnist_dsprite_train_labels.npy")

    total_ids_test = np.load("datasets/gridmnist_dsprite_test_ids.npy")
    total_labels_test = np.load("datasets/gridmnist_dsprite_test_labels.npy")

    if config['model'] == "mm_cvae": 
        transform = None
    elif config['generator']['act_fn'] == 'sigmoid' : 
        transform = transforms.ToTensor()
    else: 
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)
                               ])

    dataset = GridMnistDspriteDataset(
        total_ids_train,
        labels=total_labels_train, 
        transform=transform
    )

    testset = GridMnistDspriteDataset(
        total_ids_test,
        labels=total_labels_test, 
        transform=transform
    )

    in_channels = 1


elif config['dataset'] == 'brats_t1ce' or config['dataset'] == 'brats_t1':

    if config['dataset'] == 'brats_t1ce' : 

        total_ids_train = np.load("datasets/brats_t1ce_img_train.npy")
        total_labels_train = np.load("datasets/brats_t1ce_labels_train.npy")

        total_ids_test = np.load("datasets/brats_t1ce_img_eval.npy")
        total_labels_test = np.load("datasets/brats_t1ce_labels_eval.npy")

    elif config['dataset'] == 'brats_t1' :

        total_ids_train = np.load("datasets/brats_t1_img_train.npy")
        total_labels_train = np.load("datasets/brats_t1_labels_train.npy")

        total_ids_test = np.load("datasets/brats_t1_img_eval.npy")
        total_labels_test = np.load("datasets/brats_t1_labels_eval.npy") 



    if config['model'] == "mm_cvae": 
        transform = None
    elif config['generator']['act_fn'] == 'sigmoid' : 
        #transform = transforms.ToTensor()
        transform=transforms.Compose([
                               transforms.ToTensor(), 
                               transforms.Resize((128, 128))
                               #transforms.Resize((64, 64))
                               ])
    else: 
        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5), 
                               transforms.Resize((128, 128))
                               #transforms.Resize((64, 64))
                               ])

    dataset = BratsDataset(
        total_ids_train,
        labels=total_labels_train, 
        transform=transform
    )

    testset = BratsDataset(
        total_ids_test,
        labels=total_labels_test, 
        transform=transform
    )

    in_channels = 1



#checkpoint_callback = ModelCheckpoint(every_n_epochs=config['save_ckpt'], save_top_k=-1)

#checkpoint_callback = ModelCheckpoint(every_n_train_steps=50000, save_top_k=-1)

checkpoint_callback = ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1)


trainer = pl.Trainer(max_epochs=epochs, gpus=1, default_root_dir=args.logdir, callbacks=[checkpoint_callback]) #, enable_progress_bar=False) ## Jean Zay

save_path = args.logdir + 'lightning_logs/version_' + str(trainer.logger.version) +  "/"

loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=12, shuffle=True)

#testloader = DataLoader(testset, batch_size=config['batch_size'], num_workers=12, shuffle=False)
testloader = DataLoader(testset, batch_size=128, num_workers=12, shuffle=False)

batch_test = next(iter(testloader))


print(config['model'])

if config['model'] == 'srb_vae':
    model = Conv_sRB_VAE()
elif config['model'] == 'cvae':
    model = Conv_cVAE()
elif config['model'] == 'mm_cvae':
    model = Conv_MM_cVAE(
        background_disentanglement_penalty=config['background_disentanglement_penalty'],
        salient_disentanglement_penalty=config['salient_disentanglement_penalty'], 
        in_channels=in_channels, 
        save_path=save_path,
        salient_latent_size = config['salient_latent_size'],
        background_latent_size = config['background_latent_size'], 
        batch_test = batch_test,
        save_img_epoch = config['save_img_epoch']
    )
elif config['model'] == 'mm_cvae_robin':
    model = Conv_MM_cVAE_Robin(
        background_disentanglement_penalty=config['background_disentanglement_penalty'],
        salient_disentanglement_penalty=config['salient_disentanglement_penalty'], 
        in_channels=in_channels, 
        save_path=save_path,
        salient_latent_size = config['salient_latent_size'],
        background_latent_size = config['background_latent_size'], 
        batch_test = batch_test,
        save_img_epoch = config['save_img_epoch'], 
        config=config
    )
elif config['model'] == 'double_infogan': 
    if args.load_from_ckpt: 
        print('load from ckpt !!')
        model = Double_InfoGAN.load_from_checkpoint(args.ckpt)
        model.setup(batch_test=batch_test)
        model.config = config
    else:
        model = Double_InfoGAN(config=config, save_path=save_path, batch_test=batch_test)

elif config['model'] == 'double_infogan_w': 
    model = Double_InfoGAN_W(config=config, save_path=save_path, batch_test=batch_test)

elif config['model'] == 'double_infovaegan': 
    model = Double_Info_VAE_GAN(config=config, save_path=save_path, batch_test=batch_test)

elif config['model'] == 'double_infogan_cat': 
    model = Double_InfoGAN_cat(config=config, save_path=save_path, batch_test=batch_test)

trainer.fit(model, loader)
