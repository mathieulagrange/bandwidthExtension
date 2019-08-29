import os
import os.path
import time
from audio_data import *
from tqdm import tqdm
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, kernel_size=5):
        super(CNNModel, self).__init__()

        padding_size = int((kernel_size-1)/2)

        layers = nn.ModuleList()
        layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers.append(nn.Conv2d(1, 64, (3, kernel_size), stride=1))
        layers.append(nn.ReLU())
        layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers.append(nn.Conv2d(64, 64, (3, kernel_size), stride=1))
        layers.append(nn.ReLU())
        layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers.append(nn.Conv2d(64, 1, (3, kernel_size), stride=1))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def parameter_print(self):
        par = list(self.parameters())
        for d in par:
            print(d)

def load_latest_model_from(location, model_name, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    files = [f for f in files if model_name in f]
    newest_file = max(files, key=os.path.getctime)
    print('Loading last saved model: ' + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model

def load_model_from(model_name, use_cuda=True):
    print('Loading model: ' + model_name)

    if use_cuda:
        model = torch.load(model_name)
    else:
        model = load_to_cpu(model_name)

    return model

def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.module.cpu()
    return model
