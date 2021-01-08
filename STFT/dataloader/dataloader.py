from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch

import numpy as np
import sys

class dataloader:
  def __init__(self,train=True,npoints=2048,ndata=9840):  
    self.npoints = npoints
    self.train = train
    '''
		if train:
			prefix='/media/bofan/linux/BOFAN/pointnetdata/voxel_oreinted_traindata16/traindata'
			labeldir='/media/bofan/linux/BOFAN/downloads/pointnet2.pytorch-master/misc/csvdata_train/trainlabel.csv'
		else:
			prefix='/media/bofan/linux/BOFAN/pointnetdata/voxel_oreinted_testdata16/testdata'
			labeldir='/media/bofan/linux/BOFAN/downloads/pointnet2.pytorch-master/misc/csvdata_test/testlabel.csv'
    '''
    prefix='./UrbanSound8K/fold'
    if train:
        for i in range(9):
            if i==0:
                self.data=np.load(prefix+str(i+1)+'_data.npy')
                self.label=np.load(prefix+str(i+1)+'_ans.npy')
            else:
                self.data=np.concatenate([self.data,np.load(prefix+str(i+1)+'_data.npy')],axis=0)
                self.label=np.concatenate([self.label,np.load(prefix+str(i+1)+'_ans.npy')],axis=0)
    else:
        self.data=np.load(prefix+str(10)+'_data.npy')
        self.label=np.load(prefix+str(10)+'_ans.npy')
    print('len: ',len(self.label))

  def __getitem__(self, index):
		
    return  self.data[index], self.label[index]
  def __len__(self):
		
    return self.data.shape[0]

if __name__ == '__main__':
  #a=dataloader(train=True)
  b=dataloader(train=False)
  print(b[1][0].shape,b[1][1])