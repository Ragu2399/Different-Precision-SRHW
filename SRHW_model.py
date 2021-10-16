#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file contains functions for training a PyTorch MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import os

from random import randint

# Network



   
class SRHW(nn.Module):
    def __init__(self,upscale=2):
        super(SRHW,self).__init__()
        self.Conv1=nn.Conv2d(1,32,3,padding=(1,1),bias=False)
        nn.init.uniform_(self.Conv1.weight)
        self.DWConv1=nn.Conv2d(32,32,(1,5),padding=(0,2),groups=32,bias=False)
        nn.init.uniform_(self.DWConv1.weight)
        self.PWConv1=nn.Conv2d(32,16,1,bias=False)
        nn.init.uniform_(self.PWConv1.weight)
        self.DWConv2=nn.Conv2d(16,16,(1,5),padding=(0,2),groups=16,bias=False)
        nn.init.uniform_(self.DWConv2.weight)
        self.PWConv2=nn.Conv2d(16,32,1,bias=False)
        nn.init.uniform_(self.PWConv2.weight)
        self.DWConv3=nn.Conv2d(32,32,3,padding=(1,1),groups=32,bias=False)
        nn.init.uniform_(self.DWConv3.weight)
        self.PWConv3=nn.Conv2d(32,16,1,bias=False)
        nn.init.uniform_(self.PWConv3.weight)
        self.DWConv4=nn.Conv2d(16,16,3,padding=(1,1),groups=16,bias=False)
        nn.init.uniform_(self.DWConv4.weight)
        self.PWConv4=nn.Conv2d(16,upscale**2,1,bias=False)
        nn.init.uniform_(self.PWConv4.weight)
        self.PS=nn.PixelShuffle(upscale)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x=self.Conv1(x)
        res=self.relu(x)
        res=self.relu(self.PWConv1(self.DWConv1(res)))
        res=self.PWConv2(self.DWConv2(res))
        x=x+res
        x=self.relu(x)
        x=self.relu(self.PWConv3(self.DWConv3(x)))
        x=self.PWConv4(self.DWConv4(x))
        x=self.PS(x)
        return x


class Model(object):
    def __init__(self):
        self.network = SRHW()
        print(self.network.eval())   

    def get_weights(self):
        # loaddict
#         self.network.load_state_dict(torch.load('/home/raguhtic/Codes/FSRCNN/fsrcnn_x4.pth'))
        path="/home/raguhtic/Codes/SRHW_super_resolution/"
#         model= SRHW().to(device)
        self.network.load_state_dict(torch.load(os.path.join(path,'checkpoint1_1.8.pt')))
        return self.network.state_dict()

    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = randint(0, len(data) - 1)
        test_case = data.numpy()[case_num].ravel().astype(np.float32)
        test_name = target.numpy()[case_num]
        return test_case, test_name

