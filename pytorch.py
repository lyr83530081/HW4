# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:09:47 2021

@author: user
"""


import torch 
import torch.nn as nn #神經網絡庫
import torch.nn.functional as F #內置函數庫
from torch import optim
from torchvision import transforms
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from MyDataset import MyDataset
import time
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
                        transforms.Resize((128,128)), # 缩放
                        transforms.ToTensor(), # 图片转为张量,同时归一化像素值从[0,255]到[0,1]
                    ])

train_data=MyDataset(txt='train.txt', transform=transform)
val_data=MyDataset(txt='val.txt', transform=transform)
data_loader = DataLoader(train_data, batch_size=100,shuffle=True)
data_loader_val = DataLoader(val_data, batch_size=100,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.features = nn.Sequential(
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        #)
        self.fc1 = nn.Linear(13456, 5000)
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500, 50)
        
    def _forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        #aaa=x.cpu()
        #bb=aaa.detach().numpy()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

    


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

with open('train.txt', 'r') as f:
    index = f.readlines() 
    state=np.random.get_state()
    np.random.shuffle(index)
    lable_list = [0]*len(index)
    k=0
    for line in index:
        img_lable = line.split()
        lable_list[k] = int(img_lable[1])
        k+=1
    lable_list = np.array(lable_list)
    
with open('val.txt', 'r') as f:
    index_val = f.readlines() 
    #state=np.random.get_state()
    np.random.shuffle(index_val)
    lable_list_val = [0]*len(index_val)
    k=0
    for line in index_val:
        img_lable = line.split()
        lable_list_val[k] = int(img_lable[1])
        k+=1
    lable_list_val = np.array(lable_list_val)
    
 

output_size=50    
Y_list = MakeOneHot(lable_list, output_size)
#Y_list_val = MakeOneHot(lable_list_val, output_size)

batch_size = 64
epoches = 50
learning_rate = 0.0001
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
use_gpu = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss(size_average=False)

loss_train = []
loss_val = []
acc_list_val = []
for k in range(epoches):
    loss_t = 0
    print("-------------------------------------")
    print("epoch"+str(k))
    for i, traindata in enumerate(data_loader):
        x_train, y_train = traindata
        if use_gpu:
            x_train, y_train = Variable(x_train.cuda()),Variable(y_train.cuda())
            model = model.cuda()
        else:
            x_train, y_train = Variable(x_train),Variable(y_train)
        optimizer.zero_grad()
        y_pre = model._forward(x_train)
        prob = F.softmax(y_pre,dim=1)
        loss = criterion(y_pre, y_train)
        loss.backward()
        optimizer.step()
        aaa=loss.cpu().detach().numpy()
        loss_t+=aaa
        #print(aaa)
    loss_train.append(loss_t/len(data_loader))
    print(" Loss_val= {:.5}".format(loss_t/len(data_loader)))
    
    loss_v = 0
    correct_val = 0
    total = 0
    for j, valdata in enumerate(data_loader_val):
        x_val, y_val = valdata
        if use_gpu:
            x_val, y_val = Variable(x_val.cuda()),Variable(y_val.cuda())
            model = model.cuda()
        else:
            x_val, y_val = Variable(x_val),Variable(y_val)
        y_pre = model._forward(x_val)
        prob = F.softmax(y_pre,dim=1)
        loss = criterion(y_pre, y_val)
        aaa=loss.cpu().detach().numpy()
        loss_v += aaa
        _, label_pre = torch.max(y_pre.data, 1)
        correct_val += (label_pre == y_val.data).sum()
        total += y_val.size(0)
    loss_val.append(loss_v/len(data_loader_val))
    b = correct_val.cpu().detach().numpy()
    acc_list_val.append(b/(len(data_loader_val)*batch_size))
    print(" Loss_val= {:.5},acc_val = {:.5}%".format(loss_v/len(data_loader_val),
                                                     100*b/(len(data_loader_val)*batch_size)))
    torch.save(model.state_dict(), './net%.2f_%03d.pth' % (loss_v/len(data_loader_val),k))
#prob = torch.nn.functional.softmax(output,dim=1)
#optimizer = optim.Adam(Net.parameters(), lr=learning_rate)


        
    

        