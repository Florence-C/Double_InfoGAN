from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scanpy as sc
import os
from PIL import Image
from os.path import join as pjoin
from utils import load_idx
import torch

import matplotlib.pyplot as plt

def load_aml():
    DATA_DIR = "./data/bmmc/"
    pretransplant1 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "pretransplant1.csv"), index_col=0
    )
    pretransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "pretransplant2.csv"), index_col=0
    )

    posttransplant1 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "posttransplant1.csv"), index_col=0
    )
    posttransplant2 = pd.read_csv(
        pjoin(DATA_DIR, "clean", "posttransplant2.csv"), index_col=0
    )

    healthy1 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy1.csv"), index_col=0)
    healthy2 = pd.read_csv(pjoin(DATA_DIR, "clean", "healthy2.csv"), index_col=0)

    data1 = np.concatenate([healthy1.values, healthy2.values]).astype('float32')
    data2 = np.concatenate([pretransplant1.values, pretransplant2.values]).astype('float32')
    data3 = np.concatenate([posttransplant1.values, posttransplant2.values]).astype('float32')

    return (data1, np.zeros(data1.shape[0])), (data2, np.ones(data2.shape[0])), (data3, np.ones(data3.shape[0]) * 2)

def load_epithel():
    data1 = pd.read_csv("./data/epithel/data/Control.csv", index_col=0).T.values
    labels1 = np.zeros(data1.shape[0])
    data2 = pd.read_csv("./data/epithel/data/Salmonella.csv", index_col=0).T.values
    labels2 = np.ones(data2.shape[0])
    data3 = pd.read_csv("./data/epithel/data/Hpoly.Day10.csv", index_col=0).T.values
    labels3 = np.ones(data3.shape[0]) * 2  # Array of 2's

    return (data1.astype('float32'), labels1), (data2.astype('float32'), labels2), (data3.astype('float32'), labels3)


class SimpleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class CelebADataset(Dataset):
  def __init__(self, image_files, labels=None, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.transform = transform
    self.image_names = image_files
    self.labels = labels

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    #print('idx = ', idx)
    # Get the path to the image
    img_path = os.path.join(self.image_names[idx])
    # Load image and convert it to RGB
    image = Image.open(img_path).convert('RGB')

    face_width = face_height = 108
    j = (image.size[0] - face_width) // 2
    i = (image.size[1] - face_height) // 2
    image = image.crop([j, i, j + face_width, i + face_height])
    image = image.resize([64, 64], Image.BILINEAR)
    # Apply transformations to the image

    #image.save(str(idx) + "testPIL.jpg")

    image = np.array(image.convert('RGB')) / 255

    #plt.imsave(str(idx)+ 'test.png',image)

    if self.labels is not None:
        return image.reshape(3, 64, 64).astype('float32'), self.labels[idx]
    else:
        return image.reshape(3, 64, 64).astype('float32')


class CelebADataset2(Dataset):
  def __init__(self, image_files, labels=None, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.transform = transform
    self.image_names = image_files
    self.labels = labels

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    #print('idx = ', idx)
    # Get the path to the image
    img_path = os.path.join(self.image_names[idx])
    # Load image and convert it to RGB
    image = Image.open(img_path).convert('RGB')

    face_width = face_height = 108
    j = (image.size[0] - face_width) // 2
    i = (image.size[1] - face_height) // 2
    image = image.crop([j, i, j + face_width, i + face_height])
    image = image.resize([64, 64], Image.BILINEAR)
    # Apply transformations to the image

    #image.save(str(idx) + "testPIL.jpg")

    image = np.array(image.convert('RGB')) / 255
    image = image.astype('float32')

    #plt.imsave(str(idx)+ 'test.png',image)

    if self.transform is not None:
        image = self.transform(image)
    else: 
        image = image.reshape(3, 64, 64)


    if self.labels is not None:
        return image , self.labels[idx]
    else:
        return image


class GridMnistDspriteDataset(Dataset):
  def __init__(self, images, labels=None, transform=None, in_channels=1):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.transform = transform
    self.images = images
    self.labels = labels
    self.in_channels = in_channels

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    image = self.images[idx]

    if self.transform is not None:
        image = self.transform(image)
    else: 
        image = image.reshape(self.in_channels, 64, 64)


    if self.labels is not None:
        return image , self.labels[idx]
    else:
        return image

class BratsDataset(Dataset):
  def __init__(self, images, labels=None, transform=None, in_channels=1):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.transform = transform
    self.images = images
    self.labels = labels
    self.in_channels = in_channels

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    #image = torch.from_numpy(np.transpose(np.load(self.images[idx]))).type(torch.FloatTensor).unsqueeze(0)
    image = np.transpose(np.load(self.images[idx])).astype('float32')
    # print(self.images[idx])
    # print(image.shape)

    if self.transform is not None:
        image = self.transform(image)
    else: 
        image = image.reshape(self.in_channels, 64, 64)


    if self.labels is not None:
        return image , self.labels[idx]
    else:
        return image


class CifarMnistDataset(Dataset):
  def __init__(self, images, labels=None, transform=None, in_channels=1, label_mnist=None, label_cifar=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    self.transform = transform
    self.images = images
    self.labels = labels
    self.in_channels = in_channels
    self.label_mnist = label_mnist
    self.label_cifar = label_cifar


  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    image = self.images[idx]

    if self.transform is not None:
        image = self.transform(image)
    else: 
        image = image.reshape(self.in_channels, 64, 64)


    if self.labels is not None:
        if self.label_mnist is not None and self.label_cifar is not None : 
            return image, self.labels[idx], self.label_mnist[idx], self.label_cifar[idx]
        else: 
            return image , self.labels[idx]
    else:
        return image
