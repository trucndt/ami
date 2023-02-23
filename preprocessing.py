from PIL import Image
from img2vec_pytorch import Img2Vec
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path

img2vec = Img2Vec(cuda=True)

# CelebA
IMG2VEC_PATH = 'data_celebA/celeba_img2vec_resnet/'
IMG_PATH='data_celebA/celeba/img_align_celeba/'

Path(IMG2VEC_PATH).mkdir(parents=True, exist_ok=True)

data_name = sorted(os.listdir(IMG_PATH))

for name in tqdm(data_name):
    img_loc = IMG_PATH + name
    img = Image.open(img_loc)
    img_tensor = img2vec.get_vec(img, tensor=True)
    img_tensor = torch.squeeze(img_tensor)
    torch.save(img_tensor, IMG2VEC_PATH + name)


# CIFAR-10
train_data = datasets.CIFAR10('data_cifar10', train=True, download=True)
valid_data = datasets.CIFAR10('data_cifar10', train=False, download=True)
IMG2VEC_PATH = 'cifar10_vec/'

data_t = list(train_data)
for i in tqdm(range(len(data_t))):
    img_tensor = img2vec.get_vec(data_t[i][0], tensor=True)
    img_tensor = torch.squeeze(img_tensor)
    # img_tensor = torch.cat((img_tensor, torch.tensor([data_t[i][1]])))
    torch.save(img_tensor, IMG2VEC_PATH + str(i).zfill(5) + '.vec')

data_t = list(valid_data)
for i in tqdm(range(len(data_t))):
    img_tensor = img2vec.get_vec(data_t[i][0], tensor=True)
    img_tensor = torch.squeeze(img_tensor)
    # img_tensor = torch.cat((img_tensor, torch.tensor([data_t[i][1]])))
    torch.save(img_tensor, IMG2VEC_PATH + str(i+50000).zfill(5) + '.vec')


# ImageNet
IMG2VEC_PATH = 'imagenette2/imagenette_img2vec_resnet/'
IMG_PATH='imagenette2/'

import glob
data_name = []
start = len(IMG_PATH)
for f in glob.glob(IMG_PATH + 'train/*/*.JPEG', recursive=True):
    data_name.append(f[start:])
for f in glob.glob(IMG_PATH + 'val/*/*.JPEG', recursive=True):
    data_name.append(f[start:])

for name in tqdm(data_name):
    img_loc = IMG_PATH + name
    img = Image.open(img_loc).convert('RGB')
    img_tensor = img2vec.get_vec(img, tensor=True)
    img_tensor = torch.squeeze(img_tensor)
    Path(IMG2VEC_PATH + name).parent.mkdir(parents=True, exist_ok=True)
    torch.save(img_tensor, IMG2VEC_PATH + name)