import numpy as np
# import tensorflow.compat.v2 as tf
# from keras.datasets import cifar10

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


transform_cifar = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((32,32))])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_cifar)
trainloader_cifar = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_cifar)
testloader_cifar = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print('len trainset cifar : ', len(trainset))
print('len testset cifar : ', len(testset))


transform_mnist =  transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    ### Load MNIST Dataset

trainset_mnist = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_mnist)

trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

testset_mnist = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_mnist)

testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

print('len trainset mnist :' , len(trainset_mnist))
print('len testset mnist :' , len(testset_mnist))

print("TRAIN ")

img_train = []
label_train = []
label_cifar_train = []
label_mnist_train = []

for i, data in enumerate(trainloader_cifar): 

    
    img_cifar, label_cifar = data
    img_cifar = img_cifar.numpy()
    label_cifar = int(label_cifar.numpy())
    img_cifar = np.squeeze(img_cifar, axis = 0)

    img_cifar_list = np.split(img_cifar, 3, axis=0)
    # img_cifar_numpy = np.concatenate((np.reshape(img_cifar_list[0], (64,64,1)),np.reshape(img_cifar_list[1], (64,64,1)),np.reshape(img_cifar_list[2], (64,64,1))), axis=2)
    img_cifar_numpy = np.concatenate((np.reshape(img_cifar_list[0], (32,32,1)),np.reshape(img_cifar_list[1], (32,32,1)),np.reshape(img_cifar_list[2], (32,32,1))), axis=2)


    if i % 100 == 0 : 
        print(i)

    if i < 25000:

        if i == 0 : 
            save_image(torch.Tensor(img_cifar), 'img_0.png', normalize=True)
            matplotlib.pyplot.imsave('img_numpy.jpg', img_cifar_numpy)
        #print(label_cifar)
        img_train.append(0.5*img_cifar_numpy)
        label_train.append(0)
        label_cifar_train.append(label_cifar)
        label_mnist_train.append(-1)

    else  :

        img_mnist, label_mnist = next(iter(trainloader_mnist))
        img_mnist = img_mnist.numpy()
        label_mnist = int(label_mnist.numpy())
        img_mnist = np.squeeze(img_mnist, axis=0)
        # img_mnist = np.reshape(img_mnist,(64,64,1))
        img_mnist = np.reshape(img_mnist,(32,32,1))
        img_mnist = np.repeat(img_mnist, 3, axis=2)

        target_img = 0.5 * img_cifar_numpy + 0.5 * img_mnist

        one = np.ones_like(target_img)
        target_img = np.minimum(target_img, one)

        img_train.append(target_img)
        label_train.append(1)
        label_cifar_train.append(label_cifar)
        label_mnist_train.append(label_mnist)

        if i ==25000: 
            matplotlib.pyplot.imsave('target_train_numpy.jpg', target_img) 


print(img_train)
print(label_cifar_train)

img_train = np.array(img_train)
label_train = np.array(label_train)
label_cifar_train = np.array(label_cifar_train)
label_mnist_train = np.array(label_mnist_train)


np.save("cifar_mnist_32_img_train.npy", img_train)
np.save("cifar_mnist_32_labels_train.npy", label_train)
np.save("cifar_mnist_32_labels_cifar_train.npy", label_cifar_train)
np.save("cifar_mnist_32_labels_mnist_train.npy", label_mnist_train)


print("TEST & EVAL")

img_test = []
label_test = []
label_cifar_test = []
label_mnist_test = []

img_eval = []
label_eval = []
label_cifar_eval = []
label_mnist_eval = []

for i, data in enumerate(testloader_cifar): 

    
    img_cifar, label_cifar = data
    img_cifar = img_cifar.numpy()
    label_cifar = int(label_cifar.numpy())
    img_cifar = np.squeeze(img_cifar, axis = 0)
    img_cifar_list = np.split(img_cifar, 3, axis=0)
    #img_cifar_numpy = np.concatenate((np.reshape(img_cifar_list[0], (64,64,1)),np.reshape(img_cifar_list[1], (64,64,1)),np.reshape(img_cifar_list[2], (64,64,1))), axis=2)
    img_cifar_numpy = np.concatenate((np.reshape(img_cifar_list[0], (32,32,1)),np.reshape(img_cifar_list[1], (32,32,1)),np.reshape(img_cifar_list[2], (32,32,1))), axis=2)


    if i % 100 == 0 : 
        print(i)

    if i < 5000:

        if i < 64 :

            img_test.append(0.5*img_cifar_numpy)
            label_test.append(0)
            label_cifar_test.append(label_cifar)
            label_mnist_test.append(-1)

        img_eval.append(0.5* img_cifar_numpy)
        label_eval.append(0)
        label_cifar_eval.append(label_cifar)
        label_mnist_eval.append(-1)

    else  :

        img_mnist, label_mnist = next(iter(trainloader_mnist))
        img_mnist = img_mnist.numpy()
        label_mnist = int(label_mnist.numpy())

        img_mnist = np.squeeze(img_mnist, axis=0)
        #img_mnist = np.reshape(img_mnist,(64,64,1))
        img_mnist = np.reshape(img_mnist,(32,32,1))
        img_mnist = np.repeat(img_mnist, 3, axis=2)
        target_img = 0.5* img_cifar_numpy + 0.5 * img_mnist

        one = np.ones_like(target_img)
        target_img = np.minimum(target_img, one)


        if i < 5064 : 
            img_test.append(target_img)
            label_test.append(1)
            label_cifar_test.append(label_cifar)
            label_mnist_test.append(label_mnist)


        img_eval.append(target_img)
        label_eval.append(1)
        label_cifar_eval.append(label_cifar)
        label_mnist_eval.append(label_mnist)

        if i ==5000: 
            matplotlib.pyplot.imsave('target_test_numpy.jpg', target_img)
       

img_test = np.array(img_test)
label_test = np.array(label_test)
label_cifar_test = np.array(label_cifar_test)
label_mnist_test = np.array(label_mnist_test)

np.save("cifar_mnist_32_img_test.npy", img_test)
np.save("cifar_mnist_32_labels_test.npy", label_test)
np.save("cifar_mnist_32_labels_cifar_test.npy", label_cifar_test)
np.save("cifar_mnist_32_labels_mnist_test.npy", label_mnist_test)

img_eval = np.array(img_eval)
label_eval = np.array(label_eval)
label_cifar_eval = np.array(label_cifar_eval)
label_mnist_eval = np.array(label_mnist_eval)


np.save("cifar_mnist_32_img_eval.npy", img_eval)
np.save("cifar_mnist_32_labels_eval.npy", label_eval)
np.save("cifar_mnist_32_labels_cifar_eval.npy", label_cifar_eval)
np.save("cifar_mnist_32_labels_mnist_eval.npy", label_mnist_eval)


