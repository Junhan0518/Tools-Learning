import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from model import Origin

def process(img):
	trans = transforms.Compose([
	transforms.Resize((128, 128)),
	transforms.ToTensor(),
	])
	img = torch.reshape(trans(img), (1, 3, 128, 128))
    
	return img

def predict(image):
	model = torch.load('model.pb', map_location=torch.device('cpu'))
	label = ['Defect', 'OK']

	logits = model(image)

	logits = F.softmax(logits, dim = -1)

	pred = logits.argmax(dim=-1)

	return label[pred], math.floor(logits[0][pred].item() * 10000) / 100

if __name__ == '__main__':
	img = Image.open('static/img1.jpeg')
	img = process(img)
	label, prob = predict(img)
	print(label)
	print("%.2f" %prob)