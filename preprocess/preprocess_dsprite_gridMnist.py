
from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torchvision
import matplotlib.pyplot

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def create_gridMnist_dsprite_bg_target(dir, c1=1, c2=2, c3=3, c4=4): 

    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    filepath = f'{dir}/{filename}'
    #dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')
    dataset_zip = np.load(filepath, allow_pickle=True, encoding='latin1')

    # print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']
    imgs = imgs.astype(np.float32)

    #### Load MNIST

    transform_mnist =  transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    ### Load MNIST Dataset

    trainsetc1_1 = torchvision.datasets.MNIST(
        root=dir, train=True, download=True, transform=transform_mnist)
    indices_train_c1_1 = trainsetc1_1.targets == c1
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[indices_train_c1_1], trainsetc1_1.targets[indices_train_c1_1]


    trainsetc2_1 = torchvision.datasets.MNIST(
        root=dir, train=True, download=True, transform=transform_mnist)
    indices_train_c2_1 = trainsetc2_1.targets == c2
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[indices_train_c2_1], trainsetc2_1.targets[indices_train_c2_1]


    trainsetc3_1 = torchvision.datasets.MNIST(
        root=dir, train=True, download=True, transform=transform_mnist)
    indices_train_c3_1 = trainsetc3_1.targets == c3
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[indices_train_c3_1], trainsetc3_1.targets[indices_train_c3_1]


    trainsetc4_1 = torchvision.datasets.MNIST(
        root=dir, train=True, download=True, transform=transform_mnist)
    indices_train_c4_1 = trainsetc4_1.targets == c4
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[indices_train_c4_1], trainsetc4_1.targets[indices_train_c4_1]

    # Shuffle data
    N = len(trainsetc1_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[index], trainsetc1_1.targets[index]

    data_loader_1 = torch.utils.data.DataLoader(trainsetc1_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc2_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[index], trainsetc2_1.targets[index]

    data_loader_2 = torch.utils.data.DataLoader(trainsetc2_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc3_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[index], trainsetc3_1.targets[index]

    data_loader_3 = torch.utils.data.DataLoader(trainsetc3_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc4_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[index], trainsetc4_1.targets[index]

    data_loader_4 = torch.utils.data.DataLoader(trainsetc4_1,
                                      batch_size=1,
                                      shuffle=True)


    #### ADD MNIST TO DSPRITE


    target = []
    for i in range(imgs.shape[0]): 

        r = np.random.rand()

        if r < 0.1: 

            imgi = imgs[i].astype(np.float32)
            c1 = next(iter(data_loader_1))[0].numpy()
            c2 = next(iter(data_loader_2))[0].numpy()
            c3 = next(iter(data_loader_3))[0].numpy()
            c4 = next(iter(data_loader_4))[0].numpy()

            c1 = np.squeeze(c1)
            c2 = np.squeeze(c2)
            c3 = np.squeeze(c3)
            c4 = np.squeeze(c4)

            cat1 = np.concatenate((c1,c3), axis=0)
            cat2 = np.concatenate((c2,c4), axis=0)
            cat3 = np.concatenate((cat1, cat2), axis=1)
            # cat = np.zeros_like(imgi)
            
            add = 1.0*imgi + 0.5*cat3

            one = np.ones_like(add)
            out = np.minimum(add, one)

            out = out.astype(np.float32)

            out = np.expand_dims(out, axis=0)

            target.append(out)

        if i % 10000 ==0 : 
            print(i)


    #### Gridmnist alone

    bg = []
    for i in range(len(target)): 


        c1 = next(iter(data_loader_1))[0].numpy()
        c2 = next(iter(data_loader_2))[0].numpy()
        c3 = next(iter(data_loader_3))[0].numpy()
        c4 = next(iter(data_loader_4))[0].numpy()

        c1 = np.squeeze(c1)
        c2 = np.squeeze(c2)
        c3 = np.squeeze(c3)
        c4 = np.squeeze(c4)

        cat1 = np.concatenate((c1,c3), axis=0)
        cat2 = np.concatenate((c2,c4), axis=0)
        cat3 = np.concatenate((cat1, cat2), axis=1)
        # cat = np.zeros_like(imgi)
        
        out = 0.5*cat3

        out = np.expand_dims(out, axis=0)

        bg.append(out)

        if i % 10000 ==0 : 
            print(i)

    target = np.concatenate(target, axis=0)
    bg = np.concatenate(bg, axis=0)

    np.random.shuffle(target)

    num_test = 64

    target_test = target[:num_test]
    target_train = target[num_test:]
    bg_test = bg[:num_test]
    bg_train = bg[num_test:]

    print('target : ', target.shape)
    print('train : ', target_train.shape)
    print('test : ', target_test.shape)



    total_ids_train = np.concatenate([bg_train,target_train],axis=0)

    total_labels_train = np.concatenate([
    np.zeros(len(bg_train)),
    np.ones(len(target_train))])


    total_ids_test = np.concatenate([bg_test,target_test],axis=0)

    total_labels_test = np.concatenate([
    np.zeros(len(bg_test)),
    np.ones(len(target_test))])


    np.save("gridmnist_dsprite_train_ids.npy", total_ids_train)
    np.save("gridmnist_dsprite_train_labels.npy", total_labels_train)


    np.save("gridmnist_dsprite_test_ids.npy", total_ids_test)
    np.save("gridmnist_dsprite_test_labels.npy", total_labels_test)


    # return target, bg




def create_metric_data(dir): 
    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    filepath = f'{dir}/{filename}'

    dataset_zip = np.load(filepath, allow_pickle=True, encoding='latin1')

    imgs = dataset_zip['imgs']
    imgs = imgs.astype(np.float32)

    metadata = dataset_zip['metadata'][()]

    latents_names = metadata["latents_names"]
    latents_sizes = metadata["latents_sizes"]
    latents_possible_values = metadata["latents_possible_values"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    #### Load MNIST

    transform_mnist =  transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    c1=1
    c2=2
    c3=3 
    c4=4

    ### Load MNIST Dataset

    trainsetc1_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c1_1 = trainsetc1_1.targets == c1
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[indices_train_c1_1], trainsetc1_1.targets[indices_train_c1_1]


    trainsetc2_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c2_1 = trainsetc2_1.targets == c2
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[indices_train_c2_1], trainsetc2_1.targets[indices_train_c2_1]


    trainsetc3_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c3_1 = trainsetc3_1.targets == c3
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[indices_train_c3_1], trainsetc3_1.targets[indices_train_c3_1]


    trainsetc4_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c4_1 = trainsetc4_1.targets == c4
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[indices_train_c4_1], trainsetc4_1.targets[indices_train_c4_1]

    # Shuffle data
    N = len(trainsetc1_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[index], trainsetc1_1.targets[index]

    data_loader_1 = torch.utils.data.DataLoader(trainsetc1_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc2_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[index], trainsetc2_1.targets[index]

    data_loader_2 = torch.utils.data.DataLoader(trainsetc2_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc3_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[index], trainsetc3_1.targets[index]

    data_loader_3 = torch.utils.data.DataLoader(trainsetc3_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc4_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[index], trainsetc4_1.targets[index]

    data_loader_4 = torch.utils.data.DataLoader(trainsetc4_1,
                                      batch_size=1,
                                      shuffle=True)

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    def sample_latent(size=1):
        samples = np.zeros((size, latents_sizes.size))
        for lat_i, lat_size in enumerate(latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    metric_data_groups = []
    L = 100
    M = 500

    for i in range(M):
        fixed_latent_id = i % 5 + 1
        latents_sampled = sample_latent(size=L)
        latents_sampled[:, fixed_latent_id] = \
            np.random.randint(latents_sizes[fixed_latent_id], size=1)
        # print(latents_sampled[0:10])
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled].reshape(L, 1, 64,64).astype(np.float)

        for j in range(len(imgs_sampled)): 
            print(i,j)

            imgi = imgs_sampled[j].astype(np.float32)
            c1 = next(iter(data_loader_1))[0].numpy()
            c2 = next(iter(data_loader_2))[0].numpy()
            c3 = next(iter(data_loader_3))[0].numpy()
            c4 = next(iter(data_loader_4))[0].numpy()

            c1 = np.squeeze(c1)
            c2 = np.squeeze(c2)
            c3 = np.squeeze(c3)
            c4 = np.squeeze(c4)

            cat1 = np.concatenate((c1,c3), axis=0)
            cat2 = np.concatenate((c2,c4), axis=0)
            cat3 = np.concatenate((cat1, cat2), axis=1)
            # cat = np.zeros_like(imgi)
            
            add = 1.0*imgi + 0.5*cat3

            one = np.ones_like(add)
            out = np.minimum(add, one)

            out = out.astype(np.float32)

            out = np.expand_dims(out, axis=0)

            #target.append(out)
            imgs_sampled[j] = out

        metric_data_groups.append(
            {"img": imgs_sampled,
             "label": fixed_latent_id - 1})

    selected_ids = np.random.permutation(range(imgs.shape[0]))
    # num_img_eval_std = int(self.imgs.shape[0] / 100)
    #num_img_eval_std = int(imgs.shape[0] / 50) # 14745
    num_img_eval_std = 7000
    # num_img_eval_std = int(self.imgs.shape[0] / 10)
    print("num_img_eval_std = ", num_img_eval_std)
    print("total num img = ", imgs.shape[0])
    selected_ids = selected_ids[0: num_img_eval_std]
    metric_data_eval_std = imgs[selected_ids].reshape(num_img_eval_std, 1, 64,64).astype(np.float)

    for k in range(len(metric_data_eval_std)): 

            imgi = metric_data_eval_std[k].astype(np.float32)
            c1 = next(iter(data_loader_1))[0].numpy()
            c2 = next(iter(data_loader_2))[0].numpy()
            c3 = next(iter(data_loader_3))[0].numpy()
            c4 = next(iter(data_loader_4))[0].numpy()

            c1 = np.squeeze(c1)
            c2 = np.squeeze(c2)
            c3 = np.squeeze(c3)
            c4 = np.squeeze(c4)

            cat1 = np.concatenate((c1,c3), axis=0)
            cat2 = np.concatenate((c2,c4), axis=0)
            cat3 = np.concatenate((cat1, cat2), axis=1)
            # cat = np.zeros_like(imgi)
            
            add = 1.0*imgi + 0.5*cat3

            one = np.ones_like(add)
            out = np.minimum(add, one)

            out = out.astype(np.float32)

            out = np.expand_dims(out, axis=0)

            #target.append(out)
            metric_data_eval_std[j] = out

    metric_data = {
    "groups": metric_data_groups,
    "img_eval_std": metric_data_eval_std}

    np.save("metric_data_groups_gridmnist_dsprite.npy", metric_data_groups)
    np.save("metric_data_eval_std.npy", metric_data_eval_std)

    return metric_data



def create_eval_data(dir): 
    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    filepath = f'{dir}/{filename}'

    dataset_zip = np.load(filepath, allow_pickle=True, encoding='latin1')

    imgs = dataset_zip['imgs']
    imgs = imgs.astype(np.float32)

    metadata = dataset_zip['metadata'][()]

    latents_values = dataset_zip['latents_values']

    print("latents_values : ", latents_values)

    print("metadata = ", metadata)

    latents_names = metadata["latents_names"]
    latents_sizes = metadata["latents_sizes"]
    latents_possible_values = metadata["latents_possible_values"]

    print('latents_possible_values : ', latents_possible_values)
    print('latents_names : ', latents_names)
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    #### Load MNIST

    transform_mnist =  transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    c1=1
    c2=2
    c3=3 
    c4=4

    # ### Load MNIST Dataset

    trainsetc1_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c1_1 = trainsetc1_1.targets == c1
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[indices_train_c1_1], trainsetc1_1.targets[indices_train_c1_1]


    trainsetc2_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c2_1 = trainsetc2_1.targets == c2
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[indices_train_c2_1], trainsetc2_1.targets[indices_train_c2_1]


    trainsetc3_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c3_1 = trainsetc3_1.targets == c3
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[indices_train_c3_1], trainsetc3_1.targets[indices_train_c3_1]


    trainsetc4_1 = torchvision.datasets.MNIST(
        root=dir, train=False, download=True, transform=transform_mnist)
    indices_train_c4_1 = trainsetc4_1.targets == c4
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[indices_train_c4_1], trainsetc4_1.targets[indices_train_c4_1]

    # Shuffle data
    N = len(trainsetc1_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc1_1.data, trainsetc1_1.targets = trainsetc1_1.data[index], trainsetc1_1.targets[index]

    data_loader_1 = torch.utils.data.DataLoader(trainsetc1_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc2_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc2_1.data, trainsetc2_1.targets = trainsetc2_1.data[index], trainsetc2_1.targets[index]

    data_loader_2 = torch.utils.data.DataLoader(trainsetc2_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc3_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc3_1.data, trainsetc3_1.targets = trainsetc3_1.data[index], trainsetc3_1.targets[index]

    data_loader_3 = torch.utils.data.DataLoader(trainsetc3_1,
                                      batch_size=1,
                                      shuffle=True)

    N = len(trainsetc4_1.data)
    print(N)
    index=np.arange(N) # Integers from 0 to N-1
    np.random.shuffle(index)
    trainsetc4_1.data, trainsetc4_1.targets = trainsetc4_1.data[index], trainsetc4_1.targets[index]

    data_loader_4 = torch.utils.data.DataLoader(trainsetc4_1,
                                      batch_size=1,
                                      shuffle=True)

    num_img = len(imgs)

    num_img_eval = 7000

    eval_gridmnist_dsprite_image = []
    eval_gridmnist_dsprite_label_shape = []

    for i in range(num_img_eval): 
        idx = np.random.randint(0, num_img)

        imgi = imgs[idx].reshape(1,64,64).astype(np.float32)

        latent_val = latents_values[idx]

        shape = latents_values[idx][1]

        c1 = next(iter(data_loader_1))[0].numpy()
        c2 = next(iter(data_loader_2))[0].numpy()
        c3 = next(iter(data_loader_3))[0].numpy()
        c4 = next(iter(data_loader_4))[0].numpy()

        c1 = np.squeeze(c1)
        c2 = np.squeeze(c2)
        c3 = np.squeeze(c3)
        c4 = np.squeeze(c4)

        cat1 = np.concatenate((c1,c3), axis=0)
        cat2 = np.concatenate((c2,c4), axis=0)
        cat3 = np.concatenate((cat1, cat2), axis=1)
        # cat = np.zeros_like(imgi)
        
        add = 1.0*imgi + 0.5*cat3

        one = np.ones_like(add)
        out = np.minimum(add, one)

        out = out.astype(np.float32)
        out = np.squeeze(out)

        print(out.shape)

        eval_gridmnist_dsprite_image.append(out)

        eval_gridmnist_dsprite_label_shape.append(shape)


    eval_gridmnist_dsprite_label_shape = np.array(eval_gridmnist_dsprite_label_shape)
    eval_gridmnist_dsprite_image = np.array(eval_gridmnist_dsprite_image)

    np.save("eval_gridmnist_dsprite_image.npy", eval_gridmnist_dsprite_image)
    np.save("eval_gridmnist_dsprite_label_shape.npy", eval_gridmnist_dsprite_label_shape)



if __name__ == "__main__":


    create_gridMnist_dsprite_bg_target('./dsprite_data')

    metric_data = create_eval_data('./dsprite_data')
