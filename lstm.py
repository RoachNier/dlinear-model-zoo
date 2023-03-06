# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:27:52 2022

@author: 14935
"""

import torch
import torch.nn as nn
import math

#lstm用来做效果对比
class Model(nn.Module):
    def __init__(self,num_layers=2,input_size=9,hidden_size=18,output_size=9,batch_first=True):
        super(Model,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=batch_first)
        self.fc = nn.Linear(hidden_size,output_size)
        self.prelu = nn.PReLU()
    def forward(self,x):
        #seq_last= x[:,-1:,:].detach()
        #x = x - seq_last
        output, (hidden,cell) = self.lstm(x)
        output = self.fc(output)
        output = self.prelu(output)
        #output = output + seq_last
        return output

    

        




























