
# coding: utf-8

# In[10]:


import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import sys


# In[2]:


def scalar(data):
    maxg = max(data)
    scalar_data = [i / maxg for i in data]
    return scalar_data, maxg


# In[3]:


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


# In[4]:


def smooth(data, width):
    
    smooth_data = data.copy()
    
    for i in range(len(data)):
        
        prefix = max(0, i - width)
        suffix = min(len(data) - 1, i + width)
        smooth_data[i] = sum(data[prefix: suffix + 1]) / (suffix - prefix + 1)
        
    return smooth_data


# In[5]:


def create_sequence(dataset, windows_size = 1, shuffle = True):
    
    dataX, dataY = [], []
    
    for i in range(len(dataset) - windows_size):
        
        if shuffle:
            index = random.randint(0, i)
        else:
            index = i
        
        dataX.insert(index, dataset[i: i + windows_size])
        dataY.insert(index, dataset[i + windows_size])
        
    return dataX, dataY


# In[213]:


class Rnn(nn.Module):
    
    def __init__(self, hidden = 5):
        
        super(Rnn, self).__init__()
        
        self.rnn = nn.LSTM(1, hidden, batch_first = True)
        self.ann = nn.Linear(hidden, 1)
        self.optim = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        
        output = Variable(torch.unsqueeze(torch.FloatTensor(x), 2))
        output = self.rnn(output)[0][:,-1,:]
        output = self.ann(output)
        
        return output
    
    def neg(self, x, y):
        
        output = self.forward(x)
        std = Variable(torch.unsqueeze(torch.FloatTensor(y), 1))
        loss = torch.mean(torch.abs(output - std))
        
        return loss
    
    def predict(self, dataX):
    
        return self.forward(dataX)
    
    def train(self, trainX, trainY, testX, testY, epoch = 100):
        
        for i in range(epoch):
            
            loss = self.neg(trainX, trainY)
            self.zero_grad()
            loss.backward()
            self.optim.step()
            
            loss_test = self.neg(testX, testY)
            
            if (i == 0) or (i == epoch - 1):
                print('Epoch %d: Train Loss: %.2f  Test Loss: %.2f' % (i + 1, loss.data.tolist()[0], loss_test.data.tolist()[0]))


# In[ ]:


filename_input = sys.argv[1]
filename_output = sys.argv[2]


# In[391]:


origin_time = []
origin_data = []
file = open(filename_input, 'r')
for line in file:
    line = line[:-1].split(' ')
    origin_time.append(line[0])
    origin_data.append(float(line[1]))


# In[392]:


datasets = origin_data.copy()[-1000:]

datasets, maxg = scalar(datasets)
datasets = fill(datasets)
datasets = smooth(datasets, 5)


# In[404]:


windows_size = 7
train_ratio = 0.7
validate_ratio = 0.2
test_ratio = 1 - train_ratio - validate_ratio


# In[405]:


train_size = int(len(datasets) * train_ratio)
validate_size = int(len(datasets) * validate_ratio)
test_size = len(datasets) - train_size - validate_size

dataX, dataY = create_sequence(datasets[:-test_size], windows_size)

trainX, testX = dataX[:train_size], dataX[train_size:]
trainY, testY = dataY[:train_size], dataY[train_size:]


# In[406]:


rnn = Rnn(10)


# In[423]:


rnn.train(trainX, trainY, testX, testY)


# In[424]:


cts_predict = datasets[:-test_size][-windows_size:]

for i in range(test_size):
    
#     NdataX, NdataY = create_sequence(cts_predict, windows_size, False)
    result = rnn.predict([cts_predict[-windows_size:]])
    cts_predict += result.data.view(-1).tolist()

output = open(filename_output, 'w')
for i in cts_predict[windows_size:]:
    output.write('%.4f\n' % i)
output.flush()
output.close()
    
# NdataX, NdataY = create_sequence(datasets, windows_size, False)
# y = rnn.predict(NdataX)
# y = y.data.view(-1).tolist()
    
    
# plt.figure(figsize=(20,10))

# plt.plot(datasets[windows_size:][-test_size:])
# plt.plot(cts_predict[windows_size:])
# # plt.plot(list(range(len(datasets) - test_size, len(datasets))), cts_predict[windows_size:])
# plt.plot(y[-test_size:])
# # plt.plot([len(datasets) - test_size, len(datasets) + 1 - test_size],[0,1], 'r')

