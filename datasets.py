import random
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import seaborn as sns
import pandas as pd
import numpy
import math
from backbone.resnet import *
import torchvision.transforms as transforms, torchvision.datasets as datasets
from torchvision.datasets import ImageFolder,CIFAR100,ImageNet,CIFAR10
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, Sampler, Dataset,Subset
# from cutout import Cutout
def get_transforms(isDense):
    if isDense:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose([
           transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        return transform_train,transform_test
    else:
        transform_train = transforms.Compose([
           #transforms.RandomCrop(32,padding=4,fill=128),
           transforms.RandomResizedCrop(32),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           #Cutout(n_holes=1, length=16),
           transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
           ])
        transform_test = transforms.Compose([
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
           ])
        return transform_train,transform_test

def get_val_transforms():
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform_val


#set = ImageFolder(root="./datasets/stanforddogs/images",transform=get_transforms(isDense)[0])

def get_trainloader(params='CIFAR10',isDense=False,bs=128):
    #if params not in  ["CUB200","standford_dogs","TinyImageNet","CIFAR100","CIFAR10"]:
    #    raise NameError('training on this datasets has not been implemented')

    trainset = None
    if params == "standford_dogs":
        trainset = ImageFolder(root="./datasets/stanforddogs/train", transform=get_transforms(isDense)[0])
        #n_train = int(0.8*len(set))
        #trainset = Subset(set,range(n_train)) 
    elif params == "CUB200":
        set = ImageFolder(root=".datasets/CUB_200_2011/images", transform=get_transforms(isDense)[0])
        n_train = int(0.8*len(set))
        trainset = Subset(set,range(n_train))
    elif params == "CIFAR100":
        trainset = CIFAR100("./datasets/CIFAR100", train=True, download=True, transform=get_transforms(isDense)[0])
    elif params == "CIFAR10":
        trainset = CIFAR10("./datasets/CIFAR10", train=True, download=True, transform=get_transforms(isDense)[0])
    elif params == "TinyImageNet":
        trainset = ImageFolder(root="./datasets/TinyImageNet/train", transform=get_transforms(isDense)[0])
    elif params == "ImageNet":
        trainset = ImageNet("./datasets/ImageNet", train=True, download=True, transform=get_transforms(isDense)[0])
    elif params == "cal":
        trainset = ImageFolder("./datasets/cal/train/images",transform=get_transforms(isDense)[0])
    elif params == "CUB":
        trainset = ImageFolder("./datasets/CUB/train",transform=get_transforms(isDense)[0])

    return DataLoader(trainset,num_workers=4,batch_size=bs,shuffle=True,drop_last=True)

def get_testloader(params='CIFAR10',isDense=False,bs=128):
    # if params not in  ["CUB200","standford_dogs","TinyImageNet","CIFAR100","CIFAR10","cal"]:
    #     raise NameError('testing on this datasets has not been implemented')

    trainset = None
    if params == "standford_dogs":
        trainset = ImageFolder(root="./datasets/stanforddogs/val", transform=get_transforms(isDense)[1])
        #n_train = int(0.8*len(set))
        #trainset = Subset(set,range(n_train,len(set)))
    elif params == "CUB200":
        set = ImageFolder(root=".datasets/CUB_200_2011/images", transform=get_transforms(isDense)[1])
        n_train = int(0.8*len(set))
        trainset = Subset(set,range(n_train,len(set)))
    elif params == "CIFAR100":
        trainset = CIFAR100("./datasets/CIFAR100", train=False, download=True, transform=get_transforms(isDense)[1])
    elif params == "CIFAR10":
        trainset = CIFAR10("./datasets/CIFAR10", train=False, download=True, transform=get_transforms(isDense)[1])
    elif params == "TinyImageNet":
        trainset = ImageFolder(root="./datasets/TinyImageNet/val", transform=get_transforms(isDense)[1])
    elif params == "ImageNet":
        trainset = ImageNet("./datasets/ImageNet", train=False, download=True, transform=get_transforms(isDense)[1])
    elif params == "cal":
        trainset = ImageFolder("./datasets/cal/val/images", transform=get_transforms(isDense)[1])
    elif params == "CUB":
        trainset = ImageFolder("./datasets/CUB/val/", transform=get_transforms(isDense)[1])


    return DataLoader(trainset,num_workers=4,batch_size=bs,shuffle=True,drop_last=True)


def get_valset(params='TinyImageNet', data_path='./datasets/'):
    if params == 'TinyImageNet':
        return ImageFolder(root=r"./datasets/TinyImageNet/val", transform=get_val_transforms())
    elif params == "standford_dogs":
        return ImageFolder(root=r"./datasets/stanforddogs/val",transform=get_val_transforms())
    data = None
    if params == 'CIFAR10':
        data = CIFAR10
    elif params == 'CIFAR100':
        data = datasets.CIFAR100
    elif params == 'ImageNet':
        data = datasets.ImageNet
    elif params == 'CUB':
        data = ImageFolder("./datasets/CUB/val",transform=get_val_transforms())
        return data
    else:
        raise NameError('validation on this datasets has not been implemented')
    return data(data_path, train=False, download=True, transform=get_val_transforms())

def get_single_val_loader():
    single_data = ImageFolder(root=r"./datasets/single_val",transform=get_val_transforms())
    return DataLoader(single_data, num_workers=0, batch_size=1)

def single_test(net):
    net.eval()
    # f = open("./logs/single-logs.txt", "w")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(get_single_val_loader()):
            inputs = inputs.to("cpu")
            outputs,_ = net(inputs)
            # f.write(str(outputs[0])[7:-1])
            # f.write("\n")
            return outputs.numpy().tolist()[0]
        # print("single test finished.")
        # f.close()

def get_model():
    return ResNet18(200)


if __name__ == '__main__':
    net = get_model()
    checkpoint = torch.load("./save/samB.pth",map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    n1 = single_test(net)


    checkpoint = torch.load("./save/samA.pth",map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    n2 = single_test(net)

    checkpoint = torch.load("./save/sam1.pth",map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    n3 = single_test(net)


    checkpoint = torch.load("./save/sam2.pth",map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    n4 =  single_test(net)


    loggerbase = n1

    loggerF = n2

    loggeronlyone = n3

    loggeronlytwo = n4

    cons = 0
    K = 1
    for i in loggerbase:
        cons += math.exp(i / K)
    erf = []
    for i in loggerbase:
        erf.append(math.exp(i / K) / cons)

    cons2 = 0
    for i in loggerF:
        cons2 += math.exp(i / K)
    erf1 = []
    for i in loggerF:
        erf1.append(math.exp(i / K) / cons2)

    cons3 = 0
    for i in loggeronlyone:
        cons3 += math.exp(i / K)
    erf2 = []
    for i in loggeronlyone:
        erf2.append(math.exp(i / K) / cons3)

    cons4 = 0
    for i in loggeronlytwo:
        cons4 += math.exp(i / K)
    erf3 = []
    for i in loggeronlytwo:
        erf3.append(math.exp(i / K) / cons4)

    sd = np.linspace(0, 30, 31)
    plt.yticks([0, 5, 10, 15, 20, 25, 30])
    # plt.yticks(sd)
    plt.xlim(0, 0.25)
    plt.barh(sd, sorted(erf, reverse=True)[:31], color="blue", label="baseline")
    plt.savefig("./b1.svg")
    plt.show()
    plt.xlim(0, 0.25)
    plt.yticks([0, 5, 10, 15, 20])

    plt.barh(sd, sorted(erf1, reverse=True)[:31], alpha=0.5, color="red", label="DRI+DWR")
    plt.savefig("./a1.svg")
    plt.show()
    plt.xlim(0, 0.25)
    plt.yticks([0, 5, 10, 15, 20])

    plt.barh(sd, sorted(erf2, reverse=True)[:31], alpha=0.5, color="red", label="DRI")
    plt.savefig("./11.svg")
    plt.show()
    plt.xlim(0, 0.25)
    plt.yticks([0, 5, 10, 15, 20])

    plt.barh(sd, sorted(erf3, reverse=True)[:31], alpha=0.5, color="red", label="DSR")
    plt.savefig("./21.svg")
    plt.show()

    print(sorted(zip(loggerbase, np.linspace(0, 99, 100)), reverse=True)[:7])
    print(sorted(zip(loggerF, np.linspace(0, 99, 100)), reverse=True)[:7])
    print(sorted(zip(loggeronlyone, np.linspace(0, 99, 100)), reverse=True)[:7])
    print(sorted(zip(loggeronlytwo, np.linspace(0, 99, 100)), reverse=True)[:7])

    # val_data = iter(get_valset("standford_dogs"))
    # for i in range(1):
    #     (inputs,targets) = next(val_data)
    #     print(targets)
    #     plt.imshow(inputs.permute(1,2,0).cpu().numpy())
    #     plt.savefig("./i.png")
    #     plt.show()


