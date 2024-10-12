import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import DataLoader  # lets us load data in batches
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  # for evaluating results
import matplotlib.pyplot as plt

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120,84]):
        """
        in_sz: flatten the incoming tensors 28x28 to vec of 784
        out_sz: it will be the number of neurons on output
        """
        super().__init__()
        ## input layer
        self.fc1 = nn.Linear(in_sz,layers[0])
        ## hidden layers
        self.fc2 = nn.Linear(layers[0],layers[1])
        ## output layer
        self.fc3 = nn.Linear(layers[1],out_sz)
    
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
model = MultilayerPerceptron()
print(model)
# (out) MultilayerPerceptron(
# (out)   (fc1): Linear(in_features=784, out_features=120, bias=True)
# (out)   (fc2): Linear(in_features=120, out_features=84, bias=True)
# (out)   (fc3): Linear(in_features=84, out_features=10, bias=True)
# (out) )

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

count_parameters(model)

# (out)  94080
# (out)    120
# (out)  10080
# (out)     84
# (out)    840
# (out)     10
# (out) ______
# (out) 105214
