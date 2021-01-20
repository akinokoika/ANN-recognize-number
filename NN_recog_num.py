import numpy as np
import math

def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()

dimensions = [28*28,10]
activation = [tanh,softmax]
distribution = [
    {'b':[0,0]},
    {'b':[0,0],
    'w':[-math.sqrt(6/dimensions[0]+dimensions[1]),
        math.sqrt(6/dimensions[0]+dimensions[1])]},
]

def init_parameter_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameter_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameter():
    paremeter = []
    for i in range(len(dimensions)):
        layer_paremeter = {}
        for j in distribution[i].keys():
            if j == 'b':
                layer_paremeter['b'] = init_parameter_b(i)
                continue
            if j == 'w':
                layer_paremeter['w'] = init_parameter_w(i)
                continue
        paremeter.append(layer_paremeter)
    return paremeter

parameters = init_parameter()

def predict(img,parameters):
    l0_in = img+parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out = activation[1](l1_in)
    return l1_out
    
#print(init_parameter())
#print(predict(np.random.rand(784),parameters))
#print(predict(np.random.rand(784),parameters).argmax())

from pathlib import Path

train_img_path = './mnist/train-images.idx3-ubyte'
train_label_path = './mnist/train-labels.idx1-ubyte'
test_img_path = './mnist/t10k-images.idx3-ubyte'
test_label_path = './mnist/t10k-labels.idx1-ubyte'

import struct
import sys
from io import BufferedReader

train_num = 50000
valid_num = 10000
test_num = 10000

def file_trans_np(file_path,sel):
    f = open(file_path,'rb')
    if sel == 'img':
        struct.unpack('>4i',f.read(16))
    if sel == 'lab':
        struct.unpack('>2i',f.read(8))
    if sys.platform == "linux":
        fb = BufferedReader.read(f)
        if sel == 'img':
            rs = np.frombuffer(fb,dtype=np.uint8).reshape(-1,28*28)
        if sel == 'lab':
            rs = np.frombuffer(fb,dtype=np.uint8)
    else:
        if sel == 'img':
            rs = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)
        if sel == 'lab':
            rs = np.fromfile(f,dtype=np.uint8)
    return rs

train_img = file_trans_np(train_img_path,'img')[:train_num]
train_lab = file_trans_np(train_label_path,'lab')[:train_num]
valid_img = file_trans_np(train_img_path,'img')[train_num:]
valid_lab = file_trans_np(train_label_path,'lab')[train_num:]
test_img = file_trans_np(test_img_path,'img')
test_lab = file_trans_np(test_label_path,'lab')

import numpy as np
from matplotlib import pyplot as plt 

def show_train(index,types):
    if types == 'train':
        img = train_img[index].reshape(28,28)
        lab = train_lab[index]
    if types == 'valid':
        img = valid_img[index].reshape(28,28)
        lab = valid_lab[index]
    if types == 'test':
        img = test_img[index].reshape(28,28)
        lab = test_lab[index]
    plt.title(f"label : {str(lab)}")
    plt.imshow(img,cmap="gray")
    plt.show()
#show_train(0,'train')
show_train(np.random.randint(train_num),'train')
#show_train(np.random.randint(test_num),'test')
#show_train(np.random.randint(valid_num),'valid')
