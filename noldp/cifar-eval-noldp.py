import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from img2vec_pytorch import Img2Vec
import os
import argparse

parser = argparse.ArgumentParser(description='CIFAR-10 AMI experiments - noldp')
parser.add_argument('--D', type=int, default=100)
parser.add_argument('--times', type=int, default=5000)
parser.add_argument('--numproc', type=int, default=13)
parser.add_argument('-r', '--numneurons', type=int, default=2000)
parser.add_argument('-o', '--output_path', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, default=4)
args = parser.parse_args()



VEC_PATH='../cifar10_vec/'

import multiprocessing

import numpy as np


def tpr_tnr(prediction, truth):
    confusion_vector = prediction / truth

    true_negatives = torch.sum(confusion_vector == 1).item()
    false_negatives = torch.sum(confusion_vector == float('inf')).item()
    true_positives = torch.sum(torch.isnan(confusion_vector)).item()
    false_positives = torch.sum(confusion_vector == 0).item()

    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)    



device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark=True


import sys
print(sys.executable)

class AMIADatasetCifar10(Dataset):
    def __init__(self, target, transform, dataroot, train=True, imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        if train:
            # self.valid_data = np.arange(50000, 60000)
            self.valid_data = np.arange(50000)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            # self.train_data = np.arange(50000)
            self.train_data = np.arange(50001, 60000)
            # mask = np.ones(50000, dtype=bool)
            # mask[target] = False
            # self.train_data = self.train_data[mask, ...]
            self.length = len(self.train_data) + len(target) * multiplier
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.train = train
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.train == False:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[ int(idx / self.target_multiplier) ]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))                
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
                
        else:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[ int(idx / self.target_multiplier) ]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))                
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
            
        if self.imgroot:
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])
        
        # img_tensor = img2vec.get_vec(img, tensor=True)
        # img_tensor = torch.squeeze(img_tensor)
        img_tensor = torch.load(self.dataroot + filename)
        
        # img_tensor = img_tensor + s1.astype(np.float32)
        
        return img_tensor, class_id, img


# eps=1: 2000 neurons
# eps=2: 1000 neurons
class Classifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, args.numneurons)
        self.fc2 = nn.Linear(args.numneurons, n_outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        fc2 = self.fc2(x)
        x = torch.sigmoid(fc2)
        probs = F.softmax(x, dim=1)
        return x, probs, fc2


num_target = 1
target = [50000]

SAVE_NAME = f'{args.output_path}/Cifar10_embed_{args.numneurons}_single_{target[0]}.pth'

print(SAVE_NAME)

print('Loading data...')

test_loader = torch.utils.data.DataLoader(AMIADatasetCifar10(target, None, VEC_PATH, False, imgroot=None, multiplier=1000), shuffle=False, num_workers=0, batch_size=200000)


x_test, y_test, _ = next(iter(test_loader))
x_test = x_test.to(device)
y_test = y_test.to(device)

# print(torch.unique(y_train, return_counts=True))
print(torch.unique(y_test, return_counts=True))

print('Done.')

# for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = Classifier(512, num_target + 1)
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)


model.load_state_dict(torch.load(SAVE_NAME)['net'])
print('Train: ', torch.load(SAVE_NAME)['train'])
print('Test: ', torch.load(SAVE_NAME)['test'])
print('Acc: ', torch.load(SAVE_NAME)['acc'])
print('Epoch: ', torch.load(SAVE_NAME)['epoch'])


D = args.D
times = args.times
NUM_PROCESS = args.numproc
from tqdm import tqdm

tpr = 0
for _ in tqdm(range(times)):
    x = torch.cat((x_test[:1], x_test[np.random.randint(1000, x_test.shape[0], D-1)]))
    _, _, fc2 = model(x)
    if torch.sum(fc2[:,0] > 0) > 0:
        tpr += 1

tnr = 0
for _ in tqdm(range(times)):
    x = x_test[np.random.randint(1000, x_test.shape[0], D)]
    _, _, fc2 = model(x)
    if torch.sum(fc2[:,0] > 0) == 0:
        tnr += 1

tpr /= times
print(f'tpr = {tpr}')
tnr /= times
print(f'tnr = {tnr}')

print(f'adv = {tpr/2 + tnr/2}')
    
    
# tpr_a = []
# tnr_a = []
# adv_a = []


# for i in range(50):
#     model.load_state_dict(torch.load(SAVE_NAME + '-epoch' + str(i+1))['net'])
# #     print('Train: ', torch.load(SAVE_NAME)['train'])
#     print('Test: ', torch.load(SAVE_NAME + '-epoch' + str(i+1))['test'])
#     print('Acc: ', torch.load(SAVE_NAME + '-epoch' + str(i+1))['acc'])
# #     print('Epoch: ', torch.load(SAVE_NAME)['epoch'])


#     D = args.D
#     times = args.times
#     NUM_PROCESS = args.numproc
#     from tqdm import tqdm

#     tpr = 0
#     for _ in tqdm(range(times)):
#         x = torch.cat((x_test[:1], x_test[np.random.randint(1000, x_test.shape[0], D-1)]))
#         _, _, fc2 = model(x)
#         if torch.sum(fc2[:,0] > 0) > 0:
#             tpr += 1

#     tnr = 0
#     for _ in tqdm(range(times)):
#         x = x_test[np.random.randint(1000, x_test.shape[0], D)]
#         _, _, fc2 = model(x)
#         if torch.sum(fc2[:,0] > 0) == 0:
#             tnr += 1

#     tpr /= times
#     print(f'tpr = {tpr}')
#     tnr /= times
#     print(f'tnr = {tnr}')

#     print(f'adv = {tpr/2 + tnr/2}')
    
#     tpr_a.append(tpr)
#     tnr_a.append(tnr)
#     adv_a.append(tpr/2 + tnr/2)
    

# print(tpr_a)
# print(tnr_a)
# print(adv_a)