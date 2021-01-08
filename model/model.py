from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import math
import numpy as np
from torchaudio.transforms import Spectrogram

import torch.nn.functional as F
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation='relu'):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]/[B,C,N,1]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    #print(points.shape,idx.shape)
    if len(points.shape) == 4:
        points = points.squeeze()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    points = points.permute(0,2,1) #(B,N,C)
    new_points = points[batch_indices, idx, :]
    if len(new_points.shape)==3:
        new_points = new_points.permute(0,2,1)
    elif len(new_points.shape) == 4:
        new_points = new_points.permute(0,3,1,2)
    return new_points
        
def get_rec_window(n):
    a=np.ones(shape=[n])
    return torch.from_numpy(a)
def get_gaussian_window(n):
    a=np.arange(n)+0.0
    a-=(n/2.0)
    a=np.abs(a)/(n/2.0)
    a=-math.pi*np.power(a, 2)
    a=np.exp(a)
    return torch.from_numpy(a)
def get_triangular_window(n):
    a=np.arange(n)+0.0
    a-=(n/2.0)
    a=np.abs(a)/(n/2.0)
    a=np.ones(shape=[n])-a
    return torch.from_numpy(a)

class NET(nn.Module):
  def __init__(self):
    super(NET, self).__init__()
    self.conv1 = nn.Conv1d(1,32,kernel_size=25,stride=10)
    self.conv2 = nn.Conv1d(32,64,kernel_size=20,stride=5)
    self.conv3 = nn.Conv1d(64,128,kernel_size=10,stride=4)
    self.mp1 = torch.nn.MaxPool1d(kernel_size=2)
    self.mp2 = torch.nn.MaxPool1d(kernel_size=2)
    self.mp3 = torch.nn.MaxPool1d(kernel_size=2)
    #self.bn1 = nn.BatchNorm1d(32)
    #self.bn2 = nn.BatchNorm1d(64)
    #self.bn3 = nn.BatchNorm1d(128)
    #self.bn4 = nn.BatchNorm1d(512)
    #self.bn5 = nn.BatchNorm1d(32)
    self.fc1= nn.Linear(4992, 512)
    self.fc2= nn.Linear(512, 32)
    self.fc3= nn.Linear(32, 10)
    
  def forward(self,x):
    
    [B,L]=x.shape
    x=x.view(B,1,L)
    x=F.relu(self.conv1(x))
    x=self.mp1(x)
    x=F.relu(self.conv2(x))
    x=self.mp2(x)
    x=F.relu(self.conv3(x))
    x=self.mp3(x)
    x=x.view(B,-1)
    #print(123,x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x,dim=1)
class NET2(nn.Module):
  def __init__(self):
    super(NET2, self).__init__()
    self.conv1 = nn.Conv2d(1,32,kernel_size=[10,5],stride=[4,2])
    self.conv2 = nn.Conv2d(32,64,kernel_size=[10,5],stride=[4,2])
    self.conv3 = nn.Conv2d(64,128,kernel_size=[10,5],stride=[4,2])
    self.mp1 = torch.nn.MaxPool2d(kernel_size=[4,2])
    self.mp2 = torch.nn.MaxPool2d(kernel_size=[4,2])
    self.mp3 = torch.nn.MaxPool2d(kernel_size=[4,4])
    #self.bn1 = nn.BatchNorm2d(32)
    #self.bn2 = nn.BatchNorm2d(64)
    #self.bn3 = nn.BatchNorm2d(128)
    #self.bn4 = nn.BatchNorm1d(512)
    #self.bn5 = nn.BatchNorm1d(32)
    self.fc1= nn.Linear(3072, 512)
    self.fc2= nn.Linear(512, 32)
    self.fc3= nn.Linear(32, 10)
    self.window= get_triangular_window
    self.stft = Spectrogram(n_fft=16000,hop_length=20, window_fn=self.window,win_length=320,normalized=True)
  def forward(self,x):
    x1=self.stft(x)
    [B,H,W]=x1.shape
    #print(222,x.shape)
    x=x1.view(B,1,H,W)
    x=F.relu(self.conv1(x))
    x=self.mp1(x)
    x=F.relu(self.conv2(x))
    x=self.mp2(x)
    x=F.relu(self.conv3(x))
    x=self.mp3(x)
    x=x.view(B,-1)
    print(123,x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x,dim=1),x1
class NET3(nn.Module):
  def __init__(self):
    super(NET3, self).__init__()
    self.conv1 = nn.Conv2d(1,32,kernel_size=[10,5],stride=[4,2])
    self.conv2 = nn.Conv2d(32,64,kernel_size=[10,5],stride=[4,2])
    self.conv3 = nn.Conv2d(64,128,kernel_size=[10,5],stride=[4,2])
    self.mp1 = torch.nn.MaxPool2d(kernel_size=[4,2])
    self.mp2 = torch.nn.MaxPool2d(kernel_size=[4,2])
    self.mp3 = torch.nn.MaxPool2d(kernel_size=[4,4])
    #self.bn1 = nn.BatchNorm2d(32)
    #self.bn2 = nn.BatchNorm2d(64)
    #self.bn3 = nn.BatchNorm2d(128)
    #self.bn4 = nn.BatchNorm1d(512)
    #self.bn5 = nn.BatchNorm1d(32)
    #self.att=Self_Attn(128)
    self.fc1= nn.Linear(3072, 512)
    self.fc2= nn.Linear(512, 32)
    self.fc3= nn.Linear(32, 10)
    #self.window= get_rec_window
    self.stft1 = Spectrogram(n_fft=16000,hop_length=20, window_fn=get_rec_window,win_length=320,normalized=True)
    #self.stft2 = Spectrogram(n_fft=16000,hop_length=20, window_fn=get_gaussian_window,win_length=320,normalized=True)
    #self.stft3 = Spectrogram(n_fft=16000,hop_length=20, window_fn=get_triangular_window,win_length=320,normalized=True)
  def forward(self,x):
    x1=self.stft1(x)
    #x2=self.stft2(x)
    #x3=self.stft3(x)
    [B,H,W]=x1.shape
    #print(222,x.shape)
    x=x1.view(B,1,H,W)
    x=F.relu(self.conv1(x))
    x=self.mp1(x)
    x=F.relu(self.conv2(x))
    x=self.mp2(x)
    x=F.relu(self.conv3(x))
    x=self.mp3(x)
    #x,_=self.att(x)
    x=x.view(B,-1)
    #print(123,x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x,dim=1)
if __name__ == '__main__':
  model= NET()
  print(model)