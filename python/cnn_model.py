import os
import os.path
import time
from audio_data import *
from tqdm import tqdm
import torch.nn as nn

class View(nn.Module):
	def __init__(self, *shape):
		super(View, self).__init__()
		self.shape = shape
	def forward(self, input):
		return input.view(input.shape[0], *self.shape)

class CNNModel(nn.Module):
    def __init__(self, kernel_size=5, nb_channels=64, nb_layers=3, dilation=0):
        super(CNNModel, self).__init__()

        padding_size = int((kernel_size-1)/2)

        layers = nn.ModuleList()
        layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers.append(nn.Conv2d(1, nb_channels, (3, kernel_size), stride=1))
        layers.append(nn.ReLU())
        dil=1
        for l in range(nb_layers-2) :
            if dilation>1:
                dil = dilation
                padding_size = int(dil*(kernel_size-1)/2)
            layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
            layers.append(nn.Conv2d(nb_channels, nb_channels, (3, kernel_size), stride=1, dilation=(1, dil)))
            layers.append(nn.ReLU())
        padding_size = int((kernel_size-1)/2)
        layers.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers.append(nn.Conv2d(nb_channels, 1, (3, kernel_size), stride=1))
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

class AutoDense(nn.Module):
    def __init__(self, N=175, T=10, F=129):
        super(AutoDense, self).__init__()

        layers = nn.ModuleList()
        # Forward
        layers.append(nn.Conv2d(1, N, (1, F), stride=1, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(N, N, (T, 1), stride=1, padding=0))
        layers.append(nn.ReLU())
        layers.append(View(-1)) # Flatten before Linear
        layers.append(nn.Linear(N, 64, bias=True))
        layers.append(nn.ReLU())
        # Inverse
        layers.append(nn.Linear(64, N, bias=True))
        layers.append(nn.ReLU())
        layers.append(View(N, 1, 1)) # Reshape before Conv
        layers.append(nn.ConvTranspose2d(N, N, (T, 1), stride=1, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(N, 1, (1, F), stride=1, padding=0))
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

class AutoStride(nn.Module):
    def __init__(self, N=40, K1=(1, 5), K2=(3, 4), S=(1, 2)):
        super(AutoStride, self).__init__()

        layers = nn.ModuleList()
        # Original implem.
        '''
        # Forward
        layers.append(nn.Conv2d(1, N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(N, 2*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(2*N, 3*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(3*N, 4*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(4*N, 5*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(5*N, 6*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        # Inverse
        layers.append(nn.ConvTranspose2d(6*N, 5*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(5*N, 4*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(4*N, 3*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(3*N, 2*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(2*N, N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(N, 1, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        '''
        # F0 = 127
        # Forward
        layers.append(nn.Conv2d(1, N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(N, 2*N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(2*N, 3*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(3*N, 4*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        # Inverse
        layers.append(nn.ConvTranspose2d(4*N, 3*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(3*N, 2*N, K2, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(2*N, N, K1, stride=S, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(N, 1, K1, stride=S, padding=0))
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
