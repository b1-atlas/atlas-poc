
# coding: utf-8

# In[254]:


import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt


# In[255]:


def scalar(data):
    maxg = max(data)
    scalar_data = [i / maxg for i in data]
    return scalar_data, maxg


# In[256]:


def fill(data):

    fill_data = data.copy()

    for index, value in enumerate(data):

        prefix_index = index
        suffix_index = index

        if value == 0:

            while (0 < prefix_index) and (data[prefix_index] == 0):
                prefix_index -= 1
            while (suffix_index < len(data) - 1) and (data[suffix_index] == 0):
                suffix_index += 1

            fill_data[index] = data[prefix_index] * (suffix_index - index) + data[suffix_index] * (index - prefix_index)
            fill_data[index] /= (suffix_index - prefix_index)
            
    return fill_data


# In[257]:


def smooth(data, width):
    
    smooth_data = data.copy()
    
    for i in range(len(data)):
        
        prefix = max(0, i - width)
        suffix = min(len(data) - 1, i + width)
        smooth_data[i] = sum(data[prefix: suffix + 1]) / (suffix - prefix + 1)
        
    return smooth_data


# In[258]:


def create_sequence(dataset, prefix_size = 1, suffix_size = 1, shuffle = True):
    
    dataX, dataY = [], []
    
    for i in range(len(dataset) - prefix_size - suffix_size + 1):
        
        if shuffle:
            index = random.randint(0, i)
        else:
            index = i
        
        dataX.insert(index, dataset[i: i + prefix_size])
        dataY.insert(index, dataset[i + prefix_size: i + prefix_size + suffix_size])
        
    return dataX, dataY


# In[259]:


class Rnn(nn.Module):
    
    def __init__(self, hidden = 5):
        
        super(Rnn, self).__init__()
        
        self.hidden = hidden
        self.encoder = nn.LSTMCell(1, hidden)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.dense = nn.Linear(hidden, 1)
        self.optim = torch.optim.Adam(self.parameters())
        self.mode = 'cpu'
    
    def cuda(self, device_id = None):

        nn.Module.cuda(self, device_id)
        self.mode = 'gpu'

    def cpu(self):

        nn.Module.cpu(self)
        self.mode = 'cpu'
    
    def forward(self, x, predict_length = 1):
        
        if self.mode == 'gpu':
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 2).cuda())
            h = Variable(torch.zeros((x.size()[0], self.hidden)).cuda())
            c = Variable(torch.zeros((x.size()[0], self.hidden)).cuda())
        elif self.mode == 'cpu':
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 2))
            h = Variable(torch.zeros((x.size()[0], self.hidden)))
            c = Variable(torch.zeros((x.size()[0], self.hidden)))
        
        for itr in range(x.size()[1]):
            h, c = self.encoder(x[:, itr, :], (h, c))
        
        result = None
        
        # X, H, C (B, H)
        x, nh, nc = h, h, c
        
        for itr in range(predict_length):
            nh, nc = self.decoder(x, (nh, nc))
            # X (B, 1)
            x = self.dense(x)
            
            result = torch.cat((result, x), 1) if not (result is None) else x
            x = self.encoder(x, (h, c))[0]
            
        return result
    
    def neg(self, x, y):
        
        if self.mode == 'gpu':
            std = Variable(torch.FloatTensor(y).cuda())
        elif self.mode == 'cpu':
            std = Variable(torch.FloatTensor(y))
        
        output = self.forward(x, std.size()[1])
        loss = torch.mean(torch.abs(output - std))
        
        return loss
    
    def predict(self, dataX, predict_length = 1):
    
        return self.forward(dataX, predict_length).data.tolist()
    
    def train(self, trainX, trainY, testX, testY, epoch = 100):
        
        for i in range(epoch):
            
            loss = self.neg(trainX, trainY)
            self.zero_grad()
            loss.backward()
            self.optim.step()
            
            loss_test = self.neg(testX, testY)
            
            if (i == 0) or (i == epoch - 1):
                print('Epoch %d: Train Loss: %.2f  Test Loss: %.2f' % (i + 1, loss.data.tolist()[0], loss_test.data.tolist()[0]))


# In[274]:


def loaddata(filepath, rng = None):
    
    origin_time = []
    origin_data = []
    file = open(filepath, 'r')
    for line in file:
        line = line[:-1].split(' ')
        origin_time.append(line[0])
        origin_data.append(float(line[1]))
    
    if not (rng is None):
        datasets = origin_data.copy()[-rng:]
    else:
        datasets = origin_data.copy()

    datasets, maxg = scalar(datasets)
    datasets = fill(datasets)
    datasets = smooth(datasets, 5)
    
    return datasets


# In[275]:


def train(datasets, epoch = 1000):

    prefix_size = 28
    suffix_size = 7
    train_ratio = 0.7
    validate_ratio = 0.2
    test_ratio = 1 - train_ratio - validate_ratio

    train_size = int(len(datasets) * train_ratio)
    validate_size = int(len(datasets) * validate_ratio)
    test_size = len(datasets) - train_size - validate_size

    dataX, dataY = create_sequence(datasets[:-test_size], prefix_size, suffix_size, True)

    trainX, testX = dataX[:train_size], dataX[train_size:]
    trainY, testY = dataY[:train_size], dataY[train_size:]
    
    rnn = Rnn(5)
    rnn.cuda()
    
    rnn.train(trainX, trainY, testX, testY, epoch)
    
    return rnn


# In[276]:


def rectify(model, dataX, predict_length, preview_time = 4, alpha = 0.4, period = 90):

    answer_sequence = None
    
    time = predict_length // period if (predict_length % period == 0) else predict_length // period + 1
    
    for itr in range(time):
        
        dataY = torch.FloatTensor(model.predict(dataX))
        dataX = torch.FloatTensor(dataX)

        temp = []
        for i in range(min(preview_time, dataX.size()[1] // period)):
            temp.append(dataX[:, -(i + 1) * period:])
        
        result = dataY * alpha + temp[0][:, :period] * (1 - alpha)
#         for i in reversed(temp):
#             result = result * alpha + i[:, :period] * (1 - alpha)
#         result = result * alpha + dataY * (1 - alpha)
        dataX = torch.cat((dataX, result), 1)[:, period:]
        
        answer_sequence = torch.cat((answer_sequence, result), 1) if not (answer_sequence is None) else result

    return answer_sequence[:, :predict_length].tolist()


# In[296]:


# datasets = loaddata('./data_sale_quantity.txt', 1000)
# rnn = train(datasets, 2000)
# torch.save((datasets, rnn), './data_sale_quantity_last_year.bin')


# In[306]:


def _predict(binfile, days=90):
    obj = torch.load(binfile)
    result = rectify(obj[1], [obj[0][-days:]], days)[0]
    result = [i * obj[2] for i in result]
    return result


def predict(mode='full'):
    folder = './hack/mlalgo'
    print("predict by {}".format(mode))
    if mode == 'full':

        purchase_price = _predict('{}/data_purchase_price_full_range.bin'.format(folder))
        sales_quantity = _predict('{}/data_sale_quantity_full_range.bin'.format(folder))
    else:
        purchase_price = _predict('{}/data_purchase_price_last_year.bin'.format(folder))
        sales_quantity = _predict('{}/data_sale_quantity_last_year.bin'.format(folder))

    print('type of purchase_price in predict : {}'.format(type(purchase_price)))
    print('type of sales_quantity in predict : {}'.format(type(sales_quantity)))

    price = list()
    quantity = list()
    price.extend(purchase_price)
    quantity.extend(sales_quantity)

    return price, quantity


