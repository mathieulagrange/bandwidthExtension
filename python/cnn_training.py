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
                 optimizer_type=optim.Adam,
                 optimizer=None,
                 lr=0.001,
                 log_plus = 0,
                 weight_decay=0,
                 gradient_clipping=None,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 dtype=torch.FloatTensor,
                 spectrum_normalization=False,
                 method='dnn'):
        self.model = model
        self.method = method
        self.log_plus = log_plus
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        if optimizer is None:
            optimizer = optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = optimizer
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.spectrum_normalization = spectrum_normalization
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

        propagate = True
        if not epochs:
            epochs=1
            propagate = False
        loss = 0
        step = 0
        losses = np.zeros(epochs)
        lossValidations = np.zeros(epochs)
        fileNames = []
        tic = time.time()

        for current_epoch in range(epochs):
            loss_batch = np.zeros(len(self.dataloader))
            for current_batch, (x) in enumerate(self.dataloader):
                if self.method == 'dnn':
                    dc_b = Variable(dc.repeat(x.size(0), 1, 1).type(self.dtype))
                    # x = x/100
                    t = x[:, :, int(x.size(2)/2)+1:]
                    x = x[:, :, 1:int(x.size(2)/2)+1]
                else:
                    t = x.clone()
                    x[:, :, int(x.size(2)/2)+1:] = 0
                x = Variable(x.type(self.dtype))
                t = Variable(t.type(self.dtype))

                o = self.model(x.unsqueeze(1))

                if self.method == 'dnn':
                    reference = torch.cat((dc_b, x, t), 2)
                    prediction = torch.cat((dc_b, x, o.squeeze(dim=1)), 2)
                else:
                    reference = t
                    prediction = torch.cat((t[:, :, 0:int(x.size(2)/2)+1], o.squeeze()[:, :, int(x.size(2)/2)+1:]), 2)
                    if self.log_plus:
                        t = torch.log10(1+t)
                        x = torch.log10(1+x)
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

                if propagate:
                    loss.backward()
                loss_batch[current_batch] =  float(loss.data)*x.size(0)

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                if propagate:
                    self.optimizer.step()
                step += 1
                # time step duration
                if step == 100:
                    toc = time.time()
                    #tqdm.write('One training step takes approximately {:.6f} seconds.'.format((toc - tic) * 0.01))
            #    self.model.module.parameter_print()
            losses[current_epoch] = np.sum(loss_batch)/len(dataset)
            print('train loss:')
            print(losses)
            if dataset_validation is not None:
                storeTest, obsTest = self.test(dataset_validation, batch_size)
                lossValidations[current_epoch] = np.mean(obsTest['loss_spec'])
                print('test loss:')
                print(lossValidations)

            fileName = self.snapshot_path + '_Epoch' + str(current_epoch+1)+'.torch'
            fileNames.append(fileName)
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, fileName)
        print('Finished training at epoch {}. Final loss is {:.6f}.'.format(epochs, loss))

        obs['loss'] = losses
        obs['lossValidation'] = lossValidations
        store['modelPath'] = fileNames
        return store, obs

    # def validate(self):
    #     self.model.eval()
    #     dataset.train = False
    #
    #     # loss_weight = torch.cat((torch.linspace(10, 1, steps=513), torch.ones(512))) # 10 at 0Hz, 1 at 4kHz
    #     # loss_weight = Variable(loss_weight.type(self.dtype))
    #     # mel_filters = torch.from_numpy(lr.filters.mel(16000, 1024, n_mels=27))
    #     if not os.path.exists(self.snapshot_path):
    #         os.makedirs(self.snapshot_path)
    #
    #     total_loss = 0
    #     for x in iter(self.dataloader):
    #         t = x[:, :, 257:513]
    #         x = x[:, :, 1:257]
    #
    #         x = Variable(x.type(self.dtype))
    #         t = Variable(t.type(self.dtype))
    #
    #         o = self.model(x.unsqueeze(1))
    #         # if self.target=='spec':
    #         #     loss = F.mse_loss(o.squeeze(), t.squeeze())
    #         # elif self.target=='wspec':
    #         #     loss = torch.mean(loss_weight*(o.squeeze()-t.squeeze())**2)
    #         # elif self.target=='cpredictionqt':
    #         #     mel_filters_b = Variable(mel_filters.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
    #         #     loss = F.mse_loss(torch.bmm(o.squeeze()**2, mel_filters_b), torch.bmm(t.squeeze()**2, mel_filters_b))
    #         # else:
    #
    #         loss = F.mse_loss(o.squeeze(dim=1), t) # .squeeze()
    #         total_loss += loss.data
    #
    #     avg_loss = total_loss / len(self.dataloader)
    #     dataset.train = True
    #     self.model.train()
    #     return avg_loss

    def test(self, dataset, batch_size=1, save=False):
        self.model.eval()
        dataset.train = False
        print(dataset.frame_size)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8,
                                                 pin_memory=False)

        # DC component    print(obs['loss_spec'])
        dc = torch.zeros(1, 10, 1)
        # Initialize losses
        loss_spec = np.zeros(len(dataloader))
        loss_cqt27 = np.zeros(len(dataloader))
        loss_cqt40 = np.zeros(len(dataloader))

        store = {}
        obs = {}
        predictions = [] #np.zeros((len(dataset), 10, 256))
        # Iterate
        # print(batch_size)
        tic = time.time()
        for i, x in enumerate(dataloader):
            # print('x: '+str(x.size()))
            if i==0:
                # Mel fft filterbanks
                mel27 = torch.from_numpy(lr.filters.mel(5000, (x.size(2)*2)-1, n_mels=27))
                mel40 = torch.from_numpy(lr.filters.mel(5000, (x.size(2)*2)-1, n_mels=40))

            mel27_b = Variable(mel27.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
            mel40_b = Variable(mel40.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1).type(self.dtype))
            if self.method=='dnn':
                dc_b = Variable(dc.repeat(x.size(0), 1, 1).type(self.dtype))
                t = x[:, :, int(x.size(2)/2)+1:]
                x = x[:, :, 1:int(x.size(2)/2)+1]
            else:
                t = x.clone()
                x[:, :, int(x.size(2)/2)+1:] = 0
                if self.log_plus:
                    t = torch.log10(1+t)
                    x = torch.log10(1+x)

            x = Variable(x.type(self.dtype))
            t = Variable(t.type(self.dtype))

            o = self.model(x.unsqueeze(1))
            if self.method=='dnn':
                reference = torch.cat((dc_b, x, t), 2)
                prediction = torch.cat((dc_b, x, o.squeeze(dim=1)), 2)
                p = o.squeeze(dim=1).data.cpu().numpy()
            else:
                reference = t
                prediction = torch.cat((t[:, :, 0:int(x.size(2)/2)+1], o.squeeze()[:, :, int(x.size(2)/2)+1:]), 2)
                p = o.squeeze()[:, :, int(x.size(2)/2)+1:].data.cpu().numpy()
                if self.log_plus:
                    p = np.power(10, p)-1
            predictionFilename = self.snapshot_path+'_'+str(i+1)+'.npy'
            if save:
                # print(predictionFilename)
                np.save(predictionFilename, p)
                predictions.append(predictionFilename)
            #predictions[i*batch_size:i*batch_size+p.shape[0], :, :]= p
            # print('reference: '+str(reference.size())+' mel: '+str(mel27.size())+' melb: '+str(mel27_b.size()))
            # Compute losses
            loss_spec[i] = float(F.mse_loss(prediction, reference).data) # *x.size(0)
            loss_cqt27[i] = float(F.mse_loss(torch.bmm(prediction**2, mel27_b), torch.bmm(reference**2, mel27_b)).data) # *x.size(0)
            loss_cqt40[i] = float(F.mse_loss(torch.bmm(prediction**2, mel40_b), torch.bmm(reference**2, mel40_b)).data) # *x.size(0)

        # Reduce losses
        # loss_spec = loss_spec/len(dataset)
        # loss_cqt27 = loss_cqt27/len(dataset)
        # loss_cqt40 = loss_cqt40/len(dataset)

        # Write log file
        # toc = time.time()
        # with open('logs/' + self.snapshot_name + '_eval.txt', 'a') as log_file:
        #     log_file.write('Loss_spec: {:.6f}\nLoss_cqt27: {:.6f}\nLoss_cqt40: {:.6f}\nElapsed: {:.3f}, Iters: {}, It/s: {:.3f}\n'.format(loss_spec, loss_cqt27, loss_cqt40, toc - tic, len(dataloader), len(dataloader)/(toc - tic)))
        obs['loss_spec'] = loss_spec
        obs['loss_cqt27'] = loss_cqt27
        obs['loss_cqt40'] = loss_cqt40
        # print(obs['loss_spec'])
        store['predictions'] = predictions
        return store, obs

def generate_audio(model, tob):
    spec = model(tob)
    samples = []
    samples = griffin_lim(spec, 1024, 512, 1000)
    return samples
