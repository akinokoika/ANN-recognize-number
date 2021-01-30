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

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp/exp.sum()

dimensions = [28*28,10]
activation = [tanh,softmax]
distribution = [
    {
        'b':[0,0]
    },
    {
        'b':[0,0],
        'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),math.sqrt(6/(dimensions[0]+dimensions[1]))]
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
    l0_in = img+parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out = activation[1](l1_in)
    return l1_out

def d_softmax(data):
    sm = softmax(data)
    return np.diag(sm) - np.outer(sm,sm)
    
def d_tanh(data):
    return 1/(np.cosh(data))**2

differential = {softmax:d_softmax,tanh:d_tanh}

onehot = np.identity(dimensions[-1])

def sqr_loss(img,lab,parameters):
    y_pred = predict(img,parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff,diff)

#
def grad_parameters(img,lab,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out = activation[1](l1_in)

    diff = onehot[lab] - l1_out
    act1 = np.dot(differential[activation[1]](l1_in),diff)

    grad_b1 = -2*act1
    grad_w1 = -2*np.outer(l0_out,act1)
    grad_b0 = -2*differential[activation[0]](l0_in)*np.dot(parameters[1]['w'],act1)

    return {'b0':grad_b0,'b1':grad_b1,'w1':grad_w1}

def loss(parameters,types):
    loss_accu = 0
    if types == "valid":
        for img_i in range(valid_num):
            loss_accu += sqr_loss(valid_img[img_i],valid_lab[img_i],parameters)
        loss_accu = loss_accu/(valid_num/10000)
    if types == "train":
        for img_i in range(train_num):
            loss_accu += sqr_loss(train_img[img_i],train_lab[img_i],parameters)
        loss_accu = loss_accu/(train_num/10000)
    return loss_accu

def accuracy(parameters,types):
    if types == "valid":
        correct = [predict(valid_img[img_i],parameters).argmax() == valid_lab[img_i] for img_i in range(valid_num)]
    if types == "train":
        correct = [predict(train_img[img_i],parameters).argmax() == train_lab[img_i] for img_i in range(train_num)]
    return correct.count(True)/len(correct)

batch_size = 100

def train_batch(current_batch,parameters):
    grad_accu = grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_tmp = grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters)
        for key in grad_accu.keys():
            grad_accu[key] += grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key] /= batch_size
    return grad_accu

def combine_parameters(parameters,grad,learn_rate):
    parameters_tmp = copy.deepcopy(parameters)
    parameters_tmp[0]['b'] -= learn_rate*grad['b0']
    parameters_tmp[1]['b'] -= learn_rate*grad['b1']
    parameters_tmp[1]['w'] -= learn_rate*grad['w1']
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