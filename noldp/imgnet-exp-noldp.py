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
import glob

parser = argparse.ArgumentParser(description='ImageNet AMI experiments - noldp')
parser.add_argument('-r', '--numneurons', type=int, default=1000)
parser.add_argument('-o', '--output_path', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()



VEC_PATH = '../imagenette2/imagenette_img2vec_resnet/'
IMG_PATH='../imagenette2/'

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

class AMIADatasetImagenet(Dataset):
    
    def __init__(self, target, transform, dataroot, train=True, imgroot=None, multiplier=100):
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.target = self.__get_target_full_path__(target)
        self.target_multiplier = multiplier
        self.transform = transform
        self.train = train
        if train:
            self.valid_data = glob.glob(dataroot + 'val/*/*.JPEG', recursive=True)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            self.train_data = glob.glob(dataroot + 'train/*/*.JPEG', recursive=True)
            # self.train_data_0 = np.array(self.train_data)
            mask = np.ones(len(self.train_data), dtype=bool)
#             mask[target] = False
            for t in target:
                for i in range(len(self.train_data)):
                    if t in self.train_data[i]:
                        mask[i] = False
                        break
            self.train_data = np.array(self.train_data)[mask, ...]
            self.length = len(self.train_data) + len(target) * multiplier
            
    def __get_target_full_path__(self, target):
        train_data = glob.glob(self.dataroot + 'train/*/*.JPEG', recursive=True)
        for i in range(len(target)):
            for j in range(len(train_data)):
                if target[i] in train_data[j]:
                    target[i] = train_data[j]
                    break
                    
        return target
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.train == False:
            if idx / self.target_multiplier < len(self.target):
                filename = self.target[int(idx / self.target_multiplier)]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))                
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.train_data[idx]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
                
        else:
            if idx / self.target_multiplier < len(self.target):
                filename = self.target[ int(idx / self.target_multiplier) ]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))                
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.valid_data[idx]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
            
        if self.imgroot:
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])
        
        # img_tensor = img2vec.get_vec(img, tensor=True)
        # img_tensor = torch.squeeze(img_tensor)
        img_tensor = torch.load(filename)
        
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
target = ['n01440764_10026']

SAVE_NAME = f'{args.output_path}/ImageNet_embed_{args.numneurons}_single_{target[0]}.pth'

print(SAVE_NAME)

print('Loading data...')

train_loader = torch.utils.data.DataLoader(AMIADatasetImagenet(target, None, VEC_PATH, True, imgroot=None, multiplier=1), shuffle=False, num_workers=0, batch_size=200000)
test_loader = torch.utils.data.DataLoader(AMIADatasetImagenet(target, None, VEC_PATH, False, imgroot=None, multiplier=1000), shuffle=False, num_workers=0, batch_size=200000)


x_train, y_train, _ = next(iter(train_loader))
x_train = x_train.to(device)
y_train = y_train.to(device)

x_test, y_test, _ = next(iter(test_loader))
x_test = x_test.to(device)
y_test = y_test.to(device)
# x_train = torch.load(f'{args.output_path}/ImgNet_x_train.pt').to(device)
# y_train = torch.load(f'{args.output_path}/ImgNet_y_train.pt').to(device)

# x_test = torch.load('ImgNet-res/ImgNet_x_test.pt').to(device)
# y_test = torch.load('ImgNet-res/ImgNet_y_test.pt').to(device)

print(torch.unique(y_train, return_counts=True))
print(torch.unique(y_test, return_counts=True))

print('Done.')

# for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)


import sklearn.utils.class_weight

model = Classifier(x_train.shape[1], num_target + 1)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)

custom_weight = np.array([500, 0.1])
criterion = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))

min_loss = 100000000000
max_correct = 0
max_tpr = 0.0
max_tnr = 0.0
max_acc = 0.0
epoch = 0

lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

for i in range(1000000):
    num_correct = 0
    num_samples = 0
    loss_value = 0
    epoch += 1

    # for imgs, labels in iter(train_loader):
    model.train()

    out, probs, fc2 = model(x_train)
    loss = criterion(out, y_train)
    
    loss_value += loss
    
    loss.backward() 
    optimizer.step()              # make the updates for each parameter
    optimizer.zero_grad()         # a clean up step for PyTorch

    predictions = fc2[:, 0] < 0
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)
    
    
    # Test acc
    model.eval()
    out, probs, fc2 = model(x_test)
    predictions = fc2[:, 0] < 0
    tpr, tnr, _ = tpr_tnr(predictions, y_test)
    acc = (tpr + tnr)/2
    
   
    if acc >= max_acc:
        
        state = {
            'net': model.state_dict(),
            'test': (tpr, tnr),
            'train': (tpr_train, tnr_train),
            'acc' : acc,
            'lr' : lr,
            'epoch' : epoch
        }
        
        max_acc = acc
        torch.save(state, SAVE_NAME)
        if acc == 1.0:
            break

    
    if i % 1 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')
        
#     if epoch % 1 == 0:
#         state = {
#             'net': model.state_dict(),
#             'test': (tpr, tnr),
#             'train': (tpr_train, tnr_train),
#             'acc' : acc,
#             'lr' : lr,
#             'epoch' : epoch
#         }
        
# #         max_tpr = (tpr + tnr)/2
#         torch.save(state, SAVE_NAME + '-epoch' + str(epoch))
    

print('Train: ', torch.load(SAVE_NAME)['train'])
print('Test: ', torch.load(SAVE_NAME)['test'])
print('Acc: ', torch.load(SAVE_NAME)['acc'])
print('Epoch: ', torch.load(SAVE_NAME)['epoch'])
