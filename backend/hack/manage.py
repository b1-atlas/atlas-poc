#!/usr/bin/env python
import os
import sys
from torch import nn
import torch
from torch.autograd.variable import Variable


class Rnn(nn.Module):
    def __init__(self, hidden=5):

        super(Rnn, self).__init__()

        self.hidden = hidden
        self.encoder = nn.LSTMCell(1, hidden)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.dense = nn.Linear(hidden, 1)
        self.optim = torch.optim.Adam(self.parameters())
        self.mode = 'cpu'

    def cuda(self, device_id=None):

        nn.Module.cuda(self, device_id)
        self.mode = 'gpu'

    def cpu(self):

        nn.Module.cpu(self)
        self.mode = 'cpu'

    def forward(self, x, predict_length=1):

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

    def predict(self, dataX, predict_length=1):

        return self.forward(dataX, predict_length).data.tolist()

    def train(self, trainX, trainY, testX, testY, epoch=100):

        for i in range(epoch):

            loss = self.neg(trainX, trainY)
            self.zero_grad()
            loss.backward()
            self.optim.step()

            loss_test = self.neg(testX, testY)

            if (i == 0) or (i == epoch - 1):
                print('Epoch %d: Train Loss: %.2f  Test Loss: %.2f' % (
                i + 1, loss.data.tolist(), loss_test.data.tolist()))


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hack.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
