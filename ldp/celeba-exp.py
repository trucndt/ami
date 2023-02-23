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

parser = argparse.ArgumentParser(description='CELEBA AMI experiments')
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

import multiprocessing

import numpy as np

NUM_PROCESS = args.numproc

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, NUM_PROCESS)]

    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)



def parallel_matrix_operation(func, arr):
    chunks = np.array_split(arr, NUM_PROCESS)
    
    
    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(func, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


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


def OME(sample_feature_arr, eps=10.0, l=10, m=5):
    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

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
    perturb_feat = parallel_apply_along_axis(join_string, axis=1, arr=perturb_feat)

    return torch.tensor(parallel_matrix_operation(binary_to_float_vec, perturb_feat), dtype=torch.float)


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


def BitRand(sample_feature_arr, eps=10.0, l=10, m=5):

    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps*k /l)
    alpha = np.sqrt((eps + rl) /( 2*r *sum_ ))
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (sample_feature_arr.shape[0], r))
    p =  1/(1+alpha * np.exp(index_matrix*eps/l) )
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)

    perturb_feat = (perturb + feat)%2
    perturb_feat = parallel_apply_along_axis(join_string, axis=1, arr=perturb_feat)
    # print(perturb_feat)
    return torch.tensor(parallel_matrix_operation(binary_to_float_vec, perturb_feat), dtype=torch.float)



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
target = [0]

if args.mech == 'BitRand':
    mech = BitRand
    mech_1 = BitRand_1
elif args.mech == 'OME':
    mech = OME
    mech_1 = OME_1
else:
    print('Error mech')
    exit()

eps = args.eps
SAVE_NAME = f'{args.output_path}/CELEBA_embed_{args.numneurons}_{args.mech}_single_{target[0]}_{eps}.pth'

print(SAVE_NAME)

print('Loading data...')
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

train_loader = torch.utils.data.DataLoader(AMIADatasetCelebA(target, transform, VEC_PATH, True, imgroot=None, multiplier=4000), shuffle=False, num_workers=0, batch_size=200000)
test_loader = torch.utils.data.DataLoader(AMIADatasetCelebA(target, transform, VEC_PATH, False, imgroot=None, multiplier=1000), shuffle=False, num_workers=0, batch_size=200000)


np.random.seed(args.seed)

x_train, y_train, _ = next(iter(train_loader))
x_train[1:] = mech(x_train[1:], eps)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test, y_test, _ = next(iter(test_loader))
y_test_threat = y_test
x_test_threat = x_test
x_test_threat = mech(x_test_threat, eps)

x_test_threat = x_test_threat.to(device)
y_test_threat = y_test_threat.to(device)
print(torch.unique(y_train, return_counts=True))

torch.save(x_train, f'{args.output_path}/Celeba_x_train-{args.mech}-{int(eps)}.pt')
torch.save(y_train, f'{args.output_path}/Celeba_y_train-{args.mech}-{int(eps)}.pt')

print('Done.')


import sklearn.utils.class_weight

weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(num_target + 1), y=y_train.cpu().detach().numpy())
model = Classifier(x_train.shape[1], num_target + 1)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)

custom_weight = np.array([1600.0, 200.0])
# custom_weight = np.array([25.8845    ,  2.50796572])
criterion = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))

min_loss = 100000000000
max_correct = 0
max_tpr = 0.0
max_tnr = 0.0
max_acc = 0.0
epoch = 0

lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

for i in range(100000):
    num_correct = 0
    num_samples = 0
    loss_value = 0
    epoch += 1

    # for imgs, labels in iter(train_loader):
    model.train()

    out, probs, fc2 = model(x_train)
    loss = criterion(out, y_train)
    
    loss_value += loss
    
    predictions = fc2[:, 0] < 0
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)
    
    loss.backward() 
    optimizer.step()              # make the updates for each parameter
    optimizer.zero_grad()         # a clean up step for PyTorch
    
    
    # Test acc
    out, probs, fc2 = model(x_test_threat)
    predictions = fc2[:, 0] < 0
    tpr, tnr, _ = tpr_tnr(predictions, y_test_threat)
    acc = (tpr + tnr)/2
    
   
    if acc > max_acc:
        
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

    
    if i % 1000 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')
        
    if epoch % 20000 == 0:
        state = {
            'net': model.state_dict(),
            'test': (tpr, tnr),
            'train': (tpr_train, tnr_train),
            'acc' : acc,
            'lr' : lr,
            'epoch' : epoch
        }
        
#         max_tpr = (tpr + tnr)/2
        torch.save(state, SAVE_NAME + '-epoch' + str(epoch))
    

print('Train: ', torch.load(SAVE_NAME)['train'])
print('Test: ', torch.load(SAVE_NAME)['test'])
print('Acc: ', torch.load(SAVE_NAME)['acc'])
print('Epoch: ', torch.load(SAVE_NAME)['epoch'])
# model.load_state_dict(torch.load(SAVE_NAME)['net'])

# D = args.D
# times = args.times
# NUM_PROCESS = args.numproc
# from tqdm import tqdm

# def task_tpr(i):
#     x_test_threat = torch.cat((x_test[:1], x_test[np.random.randint(1000, 63769, D-1)]))
#     x_test_threat = mech_1(x_test_threat, eps)
#     return x_test_threat

# def task_tnr(i):
#     x_test_threat = x_test[np.random.randint(1000, 63769, D)]
#     x_test_threat = mech_1(x_test_threat, eps)
#     return x_test_threat


# with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
#     x_tpr = list(tqdm(pool.imap_unordered(task_tpr, range(times), chunksize=5), total=times))
#     x_tnr = list(tqdm(pool.imap_unordered(task_tnr, range(times), chunksize=5), total=times))

# tpr = 0
# for x in tqdm(x_tpr):
#     _, _, fc2 = model(x)
#     if torch.sum(fc2[:,0] > 0) > 0:
#         tpr += 1
                 
# tnr = 0
# for x in tqdm(x_tnr):
#     _, _, fc2 = model(x)
#     if torch.sum(fc2[:,0] > 0) == 0:
#         tnr += 1
                 
# tpr /= times
# print(f'tpr = {tpr}')
# tnr /= times
# print(f'tnr = {tnr}')

# print(f'adv = {tpr/2 + tnr/2}')


# from tqdm import tqdm
# D = args.D
# times = args.times
# tpr = 0
# for i in tqdm(range(times)):
#     x_test_threat = torch.cat((x_test[i:i+1], x_test[np.random.randint(1000, 10999, D-1)]))
#     x_test_threat = BitRand_1(x_test_threat, eps)
#     out, probs, fc2 = model(x_test_threat)
#     if torch.sum(fc2[:,0] > 0) > 0:
#         tpr += 1
        
# tpr /= times
# print(f'tpr = {tpr}')

# tnr = 0
# for i in tqdm(range(times)):
#     x_test_threat = x_test[np.random.randint(1000, 10999, D)]
#     x_test_threat = BitRand_1(x_test_threat, eps)
#     out, probs, fc2 = model(x_test_threat)
#     if torch.sum(fc2[:,0] > 0) == 0:
#         tnr += 1

# tnr /= times

# print(f'tnr = {tnr}')

# print(f'adv = {tpr/2 + tnr/2}')