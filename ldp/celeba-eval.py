import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='CELEBA AMI experiment evaluation')
parser.add_argument('-e', '--eps', type=float, required=True)
parser.add_argument('--D', type=int, default=20)
parser.add_argument('--times', type=int, default=5000)
parser.add_argument('-p', '--numproc', type=int, default=8)
parser.add_argument('-r', '--numneurons', type=int, default=1000)
parser.add_argument('-m', '--mech', type=str, required=True)
parser.add_argument('-o', '--output_path', type=str, required=True)
parser.add_argument('-s', '--seed', type=int, default=1)
args = parser.parse_args()

# for reproducibility
torch.manual_seed(args.seed)
VEC_PATH='../data_celebA/celeba_img2vec_resnet/'
IMG_PATH='../data_celebA/celeba/img_align_celeba/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark=True


import sys
print(sys.executable)

l = 10
m = 5
r = 512

def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    sign = bstr[0]
    bs = bstr[1:]
    res = int(bs, 2) / 2 ** n
    if int(sign) == 1:
        res = -1 * res
    return res

def string_to_int(a):
    bit_str = "".join(x for x in a)
    return np.array(list(bit_str)).astype(int)


def join_string(a, num_bit=l, num_feat=r):
    res = np.empty(num_feat, dtype="S10")
    # res = []
    for i in range(num_feat):
        # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
        res[i] = "".join(str(x) for x in a[i*l:(i+1)*l])
    return res


def float_bin(x):
    return float_to_binary(x, m, l-m-1)
    

def bin_float(x):
    return binary_to_float(x, m, l-m-1)


def BitRand_1(sample_feature_arr, eps, l=10, m=5, r=512):
    float_bin_2 = lambda x: float_to_binary(x, m, l-m-1)
    float_to_binary_vec_2 = np.vectorize(float_bin_2)
    bin_float_2 = lambda x: binary_to_float(x, m, l-m-1)
    binary_to_float_vec_2 = np.vectorize(bin_float_2)

    feat_tmp = float_to_binary_vec_2(sample_feature_arr)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat_tmp)
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps*k /l)
    alpha = np.sqrt((eps + r*l) /( 2*r *sum_ ))

    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (1, r))
    p =  1/(1+alpha * np.exp(index_matrix*eps/l) )
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    perturb_feat = (perturb + feat)%2
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)
    perturb_feat = binary_to_float_vec_2(perturb_feat)
    return torch.squeeze(torch.tensor(perturb_feat, dtype=torch.float))#.cuda()


def OME_1(sample_feature_arr, eps=10.0, l=10, m=5):
    
    float_bin_2 = lambda x: float_to_binary(x, m, l-m-1)
    float_to_binary_vec_2 = np.vectorize(float_bin_2)
    bin_float_2 = lambda x: binary_to_float(x, m, l-m-1)
    binary_to_float_vec_2 = np.vectorize(bin_float_2)

    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = float_to_binary_vec_2(sample_feature_arr)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    alpha_ome = 100
    index_matrix_1 = np.array([alpha_ome / (1+ alpha_ome), 1/ (1+alpha_ome**3)]*int(l/2)) # np.array(range(l))
    index_matrix_0 = np.array([ (alpha_ome * np.exp(eps/rl)) /(1 + alpha_ome* np.exp(eps/rl))]*int(l) )
    p_1 = np.tile(index_matrix_1, (sample_feature_arr.shape[0], r))
    p_0 = np.tile(index_matrix_0, (sample_feature_arr.shape[0], r))

    p_temp = np.random.rand(p_0.shape[0], p_0.shape[1])
    perturb_0 = (p_temp > p_0).astype(int)
    perturb_1 = (p_temp > p_1).astype(int)

    perturb_feat = np.array(torch.where(torch.tensor(feat)>0, torch.tensor((perturb_1 + feat)%2), torch.tensor((perturb_0 + feat)%2)) )
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)

    perturb_feat = binary_to_float_vec_2(perturb_feat)
    return torch.squeeze(torch.tensor(perturb_feat, dtype=torch.float))#.cuda()


class AMIADatasetCelebA(Dataset):
    def __init__(self, target, transform, dataroot, train=True, imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        if train:
            self.valid_data = np.arange(162770, 182637)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            self.train_data = np.arange(62770)
            mask = np.ones(62770, dtype=bool)
            mask[target] = False
            self.train_data = self.train_data[mask, ...]
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
target = [0]

if args.mech == 'BitRand':
    mech_1 = BitRand_1
elif args.mech == 'OME':
    mech_1 = OME_1
else:
    print('Error mech')
    exit()

eps = args.eps
SAVE_NAME = f'{args.output_path}/CELEBA_embed_{args.numneurons}_{args.mech}_single_{target[0]}_{eps}.pth'

print(SAVE_NAME)

print('Loading data...')

np.random.seed(args.seed)
test_loader = torch.utils.data.DataLoader(AMIADatasetCelebA(target, None, VEC_PATH, False, imgroot=None, multiplier=1000), shuffle=False, num_workers=0, batch_size=200000)
x_test, y_test, _ = next(iter(test_loader))
# x_test = torch.load('Celeba-res/Celeba_x_test.pt')
# y_test = torch.load('Celeba-res/Celeba_y_test.pt')

print(x_test.shape[0])

print(torch.unique(y_test, return_counts=True))
print('Done.')

model = Classifier(512, num_target + 1)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)


print('Train: ', torch.load(SAVE_NAME)['train'])
print('Test: ', torch.load(SAVE_NAME)['test'])
print('Acc: ', torch.load(SAVE_NAME)['acc'])
print('Epoch: ', torch.load(SAVE_NAME)['epoch'])
model.load_state_dict(torch.load(SAVE_NAME)['net'])


D = args.D
times = args.times
NUM_PROCESS = args.numproc
from tqdm import tqdm

def task_tpr(i):
    x_test_threat = torch.cat((x_test[:1], x_test[np.random.randint(1000, x_test.shape[0], D-1)]))
    x_test_threat = mech_1(x_test_threat, eps)
    return x_test_threat

def task_tnr(i):
    x_test_threat = x_test[np.random.randint(1000, x_test.shape[0], D)]
    x_test_threat = mech_1(x_test_threat, eps)
    return x_test_threat


with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
    x_tpr = list(tqdm(pool.imap_unordered(task_tpr, range(times), chunksize=5), total=times))
    x_tnr = list(tqdm(pool.imap_unordered(task_tnr, range(times), chunksize=5), total=times))

tpr = 0
for x in tqdm(x_tpr):
    _, _, fc2 = model(x)
    if torch.sum(fc2[:,0] > 0) > 0:
        tpr += 1
                 
tnr = 0
for x in tqdm(x_tnr):
    _, _, fc2 = model(x)
    if torch.sum(fc2[:,0] > 0) == 0:
        tnr += 1
                 
tpr /= times
print(f'tpr = {tpr}')
tnr /= times
print(f'tnr = {tnr}')

print(f'adv = {tpr/2 + tnr/2}')