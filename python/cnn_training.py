import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import os
import librosa as lr
from third_octave import griffin_lim
import numpy as np

class CNNTrainer:
    def __init__(self,
                 model,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 dtype=torch.FloatTensor):
        self.model = model
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype

    def train(self,
              dataset,
              dataset_validation=None,
              batch_size=32,
              target='spec',
              epochs=10,
              q = None):
        self.model.module.mode = 0 # Training mode
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=False)
        store={}
        obs={}
        # DC component
        dc = torch.zeros(1, 10, 1)

        # loss_weight = torch.cat((torch.linspace(10, 1, steps=513), torch.ones(512))) # 10 at 0Hz, 1 at 4kHz
        # loss_weight = Variable(loss_weight.type(self.dtype))
        # mel_filters = torch.from_numpy(lr.filters.mel(16000, 1024, n_mels=q))

        # if not os.path.exists(self.snapshot_path):
        #     os.makedirs(self.snapshot_path)
        # if not os.path.exists('logs'):
        #     os.makedirs('logs')

        loss = 0
        step = 0
        losses = np.zeros(epochs)
        lossValidations = np.zeros(epochs)
        fileNames = []
        tic = time.time()
        for current_epoch in range(epochs):
            loss_batch = np.zeros(len(self.dataloader))
            for current_batch, (x) in enumerate(self.dataloader):
                dc_b = Variable(dc.repeat(x.size(0), 1, 1).type(self.dtype))
                # x = x/100
                t = x[:, :, 257:513]
                x = x[:, :, 1:257]

                x = Variable(x.type(self.dtype))
                t = Variable(t.type(self.dtype))

                o = self.model(x.unsqueeze(1))

                reference = torch.cat((dc_b, x, t), 2)
                prediction = torch.cat((dc_b, x, o.squeeze(dim=1)), 2)

                # if self.target=='spec':
                #     loss = F.mse_loss(o.squeeze(), t.squeeze())
                # elif self.target=='wspec':
                #     loss = torch.mean(loss_weight*(o.squeeze()-t.squeeze())**2)
                # elif self.target=='cqt':
                #     mel_filters_b = Variable(mel_filters.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
                #     loss = F.mse_loss(torch.bmm(o.squeeze()**2, mel_filters_b), torch.bmm(t.squeeze()**2, mel_filters_b))
                # else:
                #     print('Unrecognized target representation.')

                self.optimizer.zero_grad()
                loss = F.mse_loss(prediction, reference) # .squeeze()
                loss.backward()
                loss_batch[current_batch] =  float(loss.data)*x.size(0)

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

                self.optimizer.step()
                step += 1
                # time step duration
                if step == 100:
                    toc = time.time()
                    #tqdm.write('One training step takes approximately {:.6f} seconds.'.format((toc - tic) * 0.01))
            #    self.model.module.parameter_print()
            losses[current_epoch] = np.sum(loss_batch)/len(dataset)
            print(losses)
            if dataset_validation is not None:
                storeTest, obsTest = self.test(dataset_validation, batch_size)
                lossValidations[current_epoch] = obsTest['loss_spec']
                print(lossValidations)

            fileName = self.snapshot_path + '_Epoch' + str(current_epoch+1)+'.torch'
            fileNames.append(fileName)
            torch.save(self.model, fileName)
        print('Finished training at epoch {}. Final loss is {:.6f}.'.format(epochs, loss))
        # toc = time.time()
        # with open('logs/' + self.snapshot_name + '.txt', 'a') as log_file:
        #     log_file.write('Loss: {:.6f}, Elapsed: {:.3f}, Iters: {}, It/s: {:.3f}\n'.format(loss, tLD_LIBRARY_PATH="" /usr/bin/python3 ../../main.py --expLanes=/home/lagrange/data/experiments/bandwithExtension/train/e117e4422e7d22279eefc82cde97fe4boc - tic, len(self.dataloader), len(self.dataloader)/(toc - tic)))
        obs['loss'] = losses
        obs['lossValidation'] = lossValidations
        store['modelPath'] = fileNames
        return store, obs

    def validate(self):
        self.model.eval()
        dataset.train = False

        # loss_weight = torch.cat((torch.linspace(10, 1, steps=513), torch.ones(512))) # 10 at 0Hz, 1 at 4kHz
        # loss_weight = Variable(loss_weight.type(self.dtype))
        # mel_filters = torch.from_numpy(lr.filters.mel(16000, 1024, n_mels=27))
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        total_loss = 0
        for x in iter(self.dataloader):
            t = x[:, :, 257:513]
            x = x[:, :, 1:257]

            x = Variable(x.type(self.dtype))
            t = Variable(t.type(self.dtype))

            o = self.model(x.unsqueeze(1))
            # if self.target=='spec':
            #     loss = F.mse_loss(o.squeeze(), t.squeeze())
            # elif self.target=='wspec':
            #     loss = torch.mean(loss_weight*(o.squeeze()-t.squeeze())**2)
            # elif self.target=='cpredictionqt':
            #     mel_filters_b = Variable(mel_filters.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
            #     loss = F.mse_loss(torch.bmm(o.squeeze()**2, mel_filters_b), torch.bmm(t.squeeze()**2, mel_filters_b))
            # else:
            #     print('LD_LIBRARY_PATH="" /usr/bin/python3 ../../main.py --expLanes=/home/lagrange/data/experiments/bandwithExtension/train/e117e4422e7d22279eefc82cde97fe4bUnrecognized target representation.')

            loss = F.mse_loss(o.squeeze(dim=1), t) # .squeeze()
            total_loss += loss.data

        avg_loss = total_loss / len(self.dataloader)
        dataset.train = True
        self.model.train()
        return avg_loss

    def test(self, dataset, batch_size=1):
        self.model.eval()
        dataset.train = False
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8,
                                                 pin_memory=False)

        # DC component    print(obs['loss_spec'])
        dc = torch.zeros(1, 10, 1)
        # Mel fft filterbanks
        mel27 = torch.from_numpy(lr.filters.mel(5000, 1024, n_mels=27))
        mel40 = torch.from_numpy(lr.filters.mel(5000, 1024, n_mels=40))
        # Initialize losses
        loss_spec = 0
        loss_cqt27 = 0
        loss_cqt40 = 0

        store = {}
        obs = {}
        predictions = np.zeros((len(dataset), 10, 256))
        # Iterate
        tic = time.time()
        for i, x in enumerate(dataloader):
            dc_b = Variable(dc.repeat(x.size(0), 1, 1).type(self.dtype))
            mel27_b = Variable(mel27.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
            mel40_b = Variable(mel40.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))

            t = x[:, :, 257:513]
            # x = x/100
            x = x[:, :, 1:257]

            x = Variable(x.type(self.dtype))
            t = Variable(t.type(self.dtype))

            o = self.model(x.unsqueeze(1))
            # o=o*100
            # x=x*100
            reference = torch.cat((dc_b, x, t), 2)
            prediction = torch.cat((dc_b, x, o.squeeze(dim=1)), 2)
            p = o.squeeze(dim=1).data.cpu().numpy()
            predictions[i*batch_size:i*batch_size+p.shape[0], :, :]= p
            # print('reference: '+str(reference.size())+' mel: '+str(mel27_b.size()))
            # Compute losses
            loss_spec = loss_spec+F.mse_loss(prediction, reference).data*x.size(0)
            loss_cqt27 = loss_cqt27+F.mse_loss(torch.bmm(prediction**2, mel27_b), torch.bmm(reference**2, mel27_b)).data*x.size(0)
            loss_cqt40 = loss_cqt40+F.mse_loss(torch.bmm(prediction**2, mel40_b), torch.bmm(reference**2, mel40_b)).data*x.size(0)

        # Reduce losses
        loss_spec = loss_spec/len(dataset)
        loss_cqt27 = loss_cqt27/len(dataset)
        loss_cqt40 = loss_cqt40/len(dataset)

        # Write log file
        # toc = time.time()
        # with open('logs/' + self.snapshot_name + '_eval.txt', 'a') as log_file:
        #     log_file.write('Loss_spec: {:.6f}\nLoss_cqt27: {:.6f}\nLoss_cqt40: {:.6f}\nElapsed: {:.3f}, Iters: {}, It/s: {:.3f}\n'.format(loss_spec, loss_cqt27, loss_cqt40, toc - tic, len(dataloader), len(dataloader)/(toc - tic)))
        obs['loss_spec'] = float(loss_spec)
        obs['loss_cqt27'] = float(loss_cqt27)
        obs['loss_cqt40'] = float(loss_cqt40)
        print(obs['loss_spec'])
        store['predictions'] = predictions
        return store, obs

def generate_audio(model, tob):
    spec = model(tob)
    samples = []
    samples = griffin_lim(spec, 1024, 512, 1000)
    return samples