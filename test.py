import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from scipy.ndimage import imread
from imageio import imread, imwrite
# from scipy.misc import imsave
import numpy as np
import os, random
import cv2
import torch.nn.functional as F
import AWnet
import math
import time
import warnings
warnings.filterwarnings('ignore')
# network
awnet = AWnet.pwc_residual().cuda()
# load model
awnet.load_state_dict(torch.load('model-no-noise.pkl'))

def test(ref,sr,result):
	# load image
	sr = imread(sr).astype(np.float32)/255.
	ref = imread(ref).astype(np.float32)/255.
	# input image size needs to be a multiple of 64
	[h,w,c] = sr.shape
	hx = (int((h-1)/64)+1)*64
	wx = (int((w-1)/64)+1)*64
	padding = torch.nn.ReplicationPad2d([int((wx-w)/2),wx-w-int((wx-w)/2),int((hx-h)/2),hx-h-int((hx-h)/2)])
	# from numpy to tensor
	sr = torch.from_numpy(sr.transpose(2,0,1)).unsqueeze(0)
	ref = torch.from_numpy(ref.transpose(2,0,1)).unsqueeze(0)
	# load image from cpu to gpu
	sr, ref = Variable(sr).cuda(), Variable(ref).cuda()
	# padding
	sr, ref = padding(sr), padding(ref)
	# inference
	output,warp,mask = awnet(ref, sr)
	# save result
	output = (output[:,:,int((hx-h)/2):h+int((hx-h)/2),int((wx-w)/2):w+int((wx-w)/2)]*255.0).clamp(0,255.0).detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8).astype(np.float32)/255.
	imwrite(result,(output[0]*255).astype(np.uint8))

def test_image():
	sr = 'l4.png'
	ref = 'ref.png'
	result = 'result.png'
	test(ref,sr,result)

if __name__ == '__main__':
	with torch.no_grad():
		test_image()
