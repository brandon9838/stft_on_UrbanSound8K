from __future__ import print_function 
import argparse
import random
import time
import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel # for multi-GPU training 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable

import model.model as MD
import util.util as utils
import dataloader.dataloader as DL
import cv2


train_dataset = DL.dataloader(train=True)
test_dataset = DL.dataloader(train=False)
#print('number of train samples is: ', len(train_dataset))
#print('number of test samples is: ', len(test_dataset))
print('finished loading data')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ngpu=1

max_epoch=150



def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = utils.AverageMeter() 
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
	model.train()
	end = time.time()
	[N,C,H,W]=[8,130,6,10]
	temp1=np.arange(0.0,1.0,1.0/H).reshape(H,1)
	temp1=np.repeat(temp1, W, 1)
	temp2=np.arange(0.0,1.0,1.0/W).reshape(1,W)
	temp2=np.repeat(temp2, H, 0)
	temp=np.concatenate((temp1.reshape(1,1,H,W),temp2.reshape(1,1,H,W)),axis=1)
	temp=np.repeat(temp,N,0)
	temp=torch.from_numpy(temp).float().cuda()
	tempVar=Variable(temp)
	for i, (input_points, labels) in enumerate(train_loader):
		input_points = Variable(input_points)
		labels = Variable(labels)
		input_points = input_points.cuda() 
		labels = labels.long().cuda() 
		output= model(input_points)
		#print(output)
		#print(labels)
		loss = criterion(output, labels)
		prec1 = utils.accuracy(output.data, labels.data, topk=(1,))[0]
		losses.update(loss.data, input_points.size(0))
		top1.update(prec1, input_points.size(0))
		optimizer.zero_grad()
		loss.backward() 
		utils.clip_gradient(optimizer, 0.01)
		optimizer.step()
		batch_time.update(time.time() - end) 
		end = time.time() 
		if i % 100== 0:
			#print(x1.shape)
			#x1=x1.cpu().numpy()
			#x1[0]/=np.max(x1[0])
			#x1[0]*=255.0
			#cv2.imwrite(str(labels.cpu().numpy()[0])+"i.png",x1[0])
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1)) 
	return  top1.avg

def validate(test_loader, model, criterion, epoch):
	batch_time = utils.AverageMeter() 
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
	model.eval()
	end = time.time()
	[N,C,H,W]=[8,130,6,10]
	temp1=np.arange(0.0,1.0,1.0/H).reshape(H,1)
	temp1=np.repeat(temp1, W, 1)
	temp2=np.arange(0.0,1.0,1.0/W).reshape(1,W)
	temp2=np.repeat(temp2, H, 0)
	temp=np.concatenate((temp1.reshape(1,1,H,W),temp2.reshape(1,1,H,W)),axis=1)
	temp=np.repeat(temp,N,0)
	temp=torch.from_numpy(temp).float().cuda()
	tempVar=Variable(temp)
	with torch.no_grad():
		for i, (input_points, labels) in enumerate(test_loader):
			input_points = input_points.cuda()
			labels = labels.long().cuda(async=True)
			input_var = Variable(input_points).reshape(1,-1)
			target_var =  Variable(labels)
			output= model(input_var)
			loss = criterion(output, target_var)
			prec1 = utils.accuracy(output.data, target_var.data, topk=(1,))[0]
			losses.update(loss.data, input_points.size(0))
			top1.update(prec1, input_points.size(0))
			batch_time.update(time.time() - end)
			end = time.time()
			if i % 80 == 0:
				print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					i, len(test_loader), batch_time=batch_time, loss=losses,
					top1=top1))
	print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
	return top1.avg


def main():
  best_prec1 = 0
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=False, num_workers=4)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=True, num_workers=4,drop_last=True)
  model = MD.NET3()
  train_acc=[]
  test_acc=[]
  criterion = nn.CrossEntropyLoss()
  model = model.cuda() 
  criterion = criterion.cuda()
  optimizer = optim.SGD(model.parameters(), 0.001,momentum=0.9,weight_decay=1e-4)
  for epoch in range(150):
    print('start training at: '+str(epoch))
    temp=train(train_loader, model, criterion, optimizer, epoch)
    train_acc.append(temp)
    prec1 = validate(test_loader, model, criterion, epoch)
    test_acc.append(prec1)
    temp=np.concatenate([np.array(train_acc).reshape(-1,1),np.array(test_acc).reshape(-1,1)],axis=-1)
    np.savetxt('acc.csv', temp, delimiter=',')
    if best_prec1 < prec1: 
      best_prec1 = prec1
      ''' 
      utils.save_checkpoint(model.state_dict(), '/media/bofan/linux/BOFAN/myfirstNN/checkpoint/model_best.pth')
			optim_state = {} 
			optim_state['epoch'] = epoch + 1 
			optim_state['best_prec1'] = best_prec1 
			optim_state['optim_state_best'] = optimizer.state_dict() 
			utils.save_checkpoint(optim_state,'/media/bofan/linux/BOFAN/myfirstNN/checkpoint/optim_state_best.pth')
      '''
    print('best accuracy: ', best_prec1)

if __name__ == '__main__':
	main() 
