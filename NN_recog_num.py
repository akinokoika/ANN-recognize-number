import numpy as np
import math
from pathlib import Path
import struct
from io import BufferedReader
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

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

def show_img(index,types):
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
#show_img(np.random.randint(train_num),'train')
#show_img(np.random.randint(test_num),'test')
#show_img(np.random.randint(valid_num),'valid')

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

#
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

#
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
'''
parameters = init_parameters()
grad_list = []
h = 0.001
layer = 1
pname = 'w'
for i in tqdm(range(len(parameters[layer][pname]))):
    for j in range(len(parameters[layer][pname][0])):
        img_i = np.random.randint(train_num)
        test_parameters = init_parameters()
        derivative = grad_parameters(train_img[img_i],train_lab[img_i],test_parameters)[layer][pname]
        value1 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
        test_parameters[layer][pname][i][j]+=h
        value2 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
        grad_list.append(derivative[i][j]-(value2-value1)/h)
print(np.abs(grad_list).max())
'''

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

def learn_rate_show(parameters,lower=0,upper=1,step=0.1,batch=np.random.randint(train_num//batch_size)):
    grad_lr = train_batch(batch,parameters)
    lr_list = []
    for learn_rate in tqdm(np.linspace(lower,upper,num=int((upper-lower)/step+1))):
        parameters_tmp = combine_parameters(parameters,grad_lr,learn_rate)
        loss_tmp = loss(parameters_tmp,"train")
        lr_list.append([learn_rate,loss_tmp])
    lr_list = np.array(lr_list)
    plt.plot(lr_list[:,0],lr_list[:,1],color = "deepskyblue")
    plt.show()

parameters = init_parameters()
print(accuracy(parameters,"valid"))

train_loss_list = []
train_accu_list = []
valid_loss_list = []
valid_accu_list = []

learn_rate = 0.6
epoch_num = 5
for epoch in tqdm(range(epoch_num)):
    for i in tqdm(range(train_num//batch_size)):
        grad_tmp = train_batch(i,parameters)
        parameters = combine_parameters(parameters,grad_tmp,learn_rate)
    
    train_loss_list.append(loss(parameters,"train"))
    train_accu_list.append(accuracy(parameters,"train"))
    valid_loss_list.append(loss(parameters,"valid"))
    valid_accu_list.append(accuracy(parameters,"valid"))

print(accuracy(parameters,"valid"))

x = np.random.randint(test_num)
print("predict : {}".format(predict(test_img[x],parameters).argmax()))
show_img(x,'test')

lower = 0
plt.title("loss list")
plt.plot(train_loss_list[lower:],color = "black", label="train")
plt.plot(valid_loss_list[lower:],color = "red", label="valid")
plt.show()
plt.title("accu list")
plt.plot(train_accu_list[lower:],color = "black", label="train")
plt.plot(valid_accu_list[lower:],color = "red", label="valid")
plt.show()