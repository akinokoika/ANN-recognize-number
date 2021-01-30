import numpy as np
import math
from pathlib import Path
import struct
from io import BufferedReader
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

from dearpygui.core import *
from dearpygui.simple import *

train_img_path = './mnist/train-images.idx3-ubyte'
train_label_path = './mnist/train-labels.idx1-ubyte'
test_img_path = './mnist/t10k-images.idx3-ubyte'
test_label_path = './mnist/t10k-labels.idx1-ubyte'

train_num = 50000
valid_num = 10000
test_num = 10000

def file_trans_np(file_path,sel):
    with open(file_path,'rb') as f:
        if sel == 'img':
            struct.unpack('>4i',f.read(16))
            rs = np.fromfile(f,dtype = np.uint8).reshape(-1,28*28)/255
        if sel == 'lab':
            struct.unpack('>2i',f.read(8))
            rs = np.fromfile(f,dtype = np.uint8)
    return rs

train_img = file_trans_np(train_img_path,'img')[:train_num]
train_lab = file_trans_np(train_label_path,'lab')[:train_num]
valid_img = file_trans_np(train_img_path,'img')[train_num:]
valid_lab = file_trans_np(train_label_path,'lab')[train_num:]
test_img = file_trans_np(test_img_path,'img')
test_lab = file_trans_np(test_label_path,'lab')

def bypass(x):
    return x

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp/exp.sum()

def d_bypass(data):
    return 1

def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm,sm)
    
def d_tanh(data):
    return 1/(np.cosh(data))**2

differential = {softmax:d_softmax,tanh:d_tanh,bypass:d_bypass}
d_type = {bypass:'times',softmax:'dot',tanh:'times'}

dimensions = [28*28,100,10]
activation = [bypass,tanh,softmax]
distribution = [
    {
        #'b':[0,0]
    },
    {
        'b':[0,0],
        'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),math.sqrt(6/(dimensions[0]+dimensions[1]))]
    },
    {
        'b':[0,0],
        'w':[-math.sqrt(6/(dimensions[1]+dimensions[2])),math.sqrt(6/(dimensions[1]+dimensions[2]))]
    }
]

def init_parameter_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameter_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameters():
    paremeter = []
    for i in range(len(distribution)):
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

global parameters
parameters = init_parameters()

def predict(img,parameters):
    l_in = img
    l_out = activation[0](l_in)
    for layer in range(1,len(dimensions)):
        l_in = np.dot(l_out,parameters[layer]['w'])+parameters[layer]['b']
        l_out = activation[layer](l_in)
    return l_out

onehot = np.identity(dimensions[-1])

def sqr_loss(img,lab,parameters):
    y_pred = predict(img,parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff,diff)

def grad_parameters(img,lab,parameters):
    l_in_list = [img]
    l_out_list = [activation[0](l_in_list[0])]

    for layer in range(1,len(dimensions)):
        l_in = np.dot(l_out_list[layer-1],parameters[layer]['w'])+parameters[layer]['b']
        l_out = activation[layer](l_in)
        l_in_list.append(l_in)
        l_out_list.append(l_out)

    d_layer = -2*(onehot[lab] - l_out_list[-1])
    grad_result = [None] * len(dimensions)

    for layer in range(len(dimensions)-1,0,-1):
        if d_type[activation[layer]] == 'times':
            d_layer = differential[activation[layer]](l_in_list[layer])*d_layer
        if d_type[activation[layer]] == 'dot':
            d_layer = np.dot(differential[activation[layer]](l_in_list[layer]),d_layer)
        grad_result[layer] = {}
        grad_result[layer]['b'] = d_layer
        grad_result[layer]['w'] = np.outer(l_out_list[layer-1],d_layer)
        d_layer = np.dot(parameters[layer]['w'],d_layer)

    return grad_result

def loss(parameters,types):
    loss_accu = 0
    if types == "train":
        for img_i in range(train_num):
            loss_accu += sqr_loss(train_img[img_i],train_lab[img_i],parameters)
        loss_accu = loss_accu/(train_num/10000)
    if types == "valid":
        for img_i in range(valid_num):
            loss_accu += sqr_loss(valid_img[img_i],valid_lab[img_i],parameters)
        loss_accu = loss_accu/(valid_num/10000)
    return loss_accu

def accuracy(parameters,types):
    if types == "train":
        correct = [predict(train_img[img_i],parameters).argmax() == train_lab[img_i] for img_i in range(train_num)]
    if types == "valid":
        correct = [predict(valid_img[img_i],parameters).argmax() == valid_lab[img_i] for img_i in range(valid_num)]
    if types == "test":
        correct = [predict(test_img[img_i],parameters).argmax() == test_lab[img_i] for img_i in range(test_num)]
    return correct.count(True)/len(correct)

batch_size = 100

def grad_count(grad1,grad2,types):
    for layer in range(1,len(grad1)):
        for pname in grad1[layer].keys():
            if types == "add":
                grad1[layer][pname] += grad2[layer][pname]
            if types == "divide":
                grad1[layer][pname] /= grad2
    return grad1

def train_batch(current_batch,parameters):
    grad_accu = grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_tmp = grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters)
        grad_count(grad_accu,grad_tmp,"add")
    grad_count(grad_accu,batch_size,"divide")
    return grad_accu

def combine_parameters(parameters,grad,learn_rate):
    parameters_tmp = copy.deepcopy(parameters)
    for layer in range(1,len(parameters_tmp)):
        for pname in parameters_tmp[layer].keys():
            parameters_tmp[layer][pname] -= learn_rate * grad[layer][pname]
    return parameters_tmp

def train(parameters,learn_rate,epoch_num,statistics=False):
    for epoch in range(epoch_num):
        for i in range(train_num//batch_size):
            if i%100 == 99:
                with window("retrieve_train"):
                    add_text("running batch {}/{}".format(i+1,train_num//batch_size))
            grad_tmp = train_batch(i,parameters)
            parameters = combine_parameters(parameters,grad_tmp,learn_rate)
    with window("retrieve_train"):
        add_text("train complete!")
    return parameters

def retrieve_train(sender, data):
    delete_item("retrieve_train")
    with window("retrieve_train"):
        add_text("learn rate: " + str(round(get_value("learn rate"),4)))
        add_text("epoch_num: " + str(get_value("epoch_num")))
        add_text("please wait...")
        global parameters
        parameters = train(parameters,round(get_value("learn rate"),4),get_value("epoch_num"))
        
global accuracy_list
accuracy_list = []

def retrieve_accuracy(sender, data):
    delete_item("retrieve_accuracy")
    with window("retrieve_accuracy"):
        accu = accuracy(parameters,"valid")
        add_text("current accuracy : {}".format(str(accu)))
        global accuracy_list
        accuracy_list.append(accu)

def retrieve_show(sender, data):
    delete_item("retrieve_show")
    with window("retrieve_show"):
        add_text("current accuracy_list : {}".format(str(accuracy_list)))
        add_text("click accuracy button to add list")
        add_simple_plot("", value=accuracy_list, height=300)

with window("test"):
    add_input_float("learn rate",default_value=0.6)
    add_input_int("epoch_num",default_value=1)
    add_button("accuracy", callback=retrieve_accuracy)
    add_button("train", callback=retrieve_train)
    add_button("show", callback=retrieve_show)

set_main_window_size(500, 500)

set_primary_window("test",True)

start_dearpygui()