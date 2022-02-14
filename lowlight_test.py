import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import lowlight_model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time



def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)  # 读RGB图像 C*H*W
	data_lowlight = (np.asarray(data_lowlight)/255.0)  # 转化为ndarray

	data_lowlight = torch.from_numpy(data_lowlight).float()  # 转化为tensor
	data_lowlight = data_lowlight.permute(2, 0, 1)   # 维度转置 H*W*C
	data_lowlight = data_lowlight.cuda().unsqueeze(0)  # 在第0维插入1个维度

	SCL_LLE_net = lowlight_model.enhance_net_nopool().cuda()
	SCL_LLE_net.load_state_dict(torch.load('checkpoints/Epoch38.0.pth'))
	start = time.time()
	_,enhanced_image,_ = SCL_LLE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	with torch.no_grad():
		filePath = 'datasets/test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name)
			for image in test_list:
				print(image)
				lowlight(image)

		

