import time
from cnn_model import *
from scipy.io import wavfile
import librosa as lr
import argparse
import torch
import torch.nn as nn
from third_octave import griffin_lim
from torch.autograd import Variable

def generate_and_log_samples(config):
    model_name = config.dataset_train + '_CNN' + '_t' + config.target
    if config.target == 'cqt':
        model_name = model_name + str(config.q)
    gen_model = load_latest_model_from('snapshots/'+config.dataset_train, model_name, use_cuda=True)
    gen_model.mode = 1 # Generate mode
    sr = 16000
    fLen = int(4096*sr/32000)
    hLenInput = fLen
    hLenTarget = int(fLen/4)
    
    print('----- Generation -----')
    print(' -> Dataset:     ', config.dataset_gen)
    print(' -> File:        ', config.file)
    
    x, _ = lr.load(path='/media/gontiefel/Data_1/datasets/'+config.dataset_gen+'/'+config.file+'.wav', sr=sr, mono=True)
    x = lr.util.normalize(x)
    if (x.shape[0]-2048)%2048 != 0:
        x = np.append(x, np.zeros(hLenInput-(x.shape[0]-fLen)%hLenInput))
    X = np.abs(lr.stft(x, fLen, hLenInput, window='rect', center=False))
    
    X = torch.from_numpy(X).unsqueeze(0).permute(0, 2, 1)
    X = Variable(X.type(torch.FloatTensor))
    X_CNN = gen_model(X)
    X_CNN = X_CNN.permute(1, 0).squeeze().cpu().data.numpy()
    X_CNN = X_CNN[:, :-3]
    if config.phase=='gl':
        x_rec = griffin_lim(X_CNN, fLen, hLenTarget, 1000)
    elif config.phase=='oracle':
        phiX = np.angle(lr.stft(x, fLen, hLenTarget, window='rect', center=False))
        X_rec = X_CNN*np.exp(1.0j*phiX)
        x_rec = lr.core.istft(X_rec, hLenTarget, window='rect', center=False)
    
    if not os.path.exists('generated_samples/'+config.dataset_train):
        os.makedirs('generated_samples/'+config.dataset_train)
    lr.output.write_wav('generated_samples/'+config.dataset_train+'/'+model_name+'_'+config.file+'_'+config.phase+'.wav', np.transpose(x_rec), sr, norm=True)
    
    # Metrics
    if os.path.isfile('generated_samples/'+config.dataset_train+'/'+model_name+'_'+config.file+'_'+config.phase+'.txt'):
        os.remove('generated_samples/'+config.dataset_train+'/'+model_name+'_'+config.file+'_'+config.phase+'.txt')
    xref = x
    x = x_rec
    
    # Wave loss
    loss_wave = np.mean((x-xref)**2)
    
    # Spectrogram losses
    X = np.abs(lr.stft(x, fLen, hLenTarget, window='rect', center=False))
    Xref = np.abs(lr.stft(xref, fLen, hLenTarget, window='rect', center=False))
    loss_spec = np.mean((X-Xref)**2)
    # Spectrogram loss weights: 10 at 0Hz, 1 at 4kHz
    wloss = np.append(np.linspace(10, 1, num=int(fLen/4)+1, endpoint=True), np.ones(int(fLen/4)))
    wloss = np.transpose(np.tile(wloss, (X.shape[1], 1)))
    loss_wspec = np.mean(wloss*((X-Xref)**2))
    
    # CQT losses
    mel27 = lr.filters.mel(sr, fLen, n_mels=27)
    mel40 = lr.filters.mel(sr, fLen, n_mels=40)
    mel80 = lr.filters.mel(sr, fLen, n_mels=80)
    loss_mel27 = np.mean((np.dot(mel27, X**2)-np.dot(mel27, Xref**2))**2)
    loss_mel40 = np.mean((np.dot(mel40, X**2)-np.dot(mel40, Xref**2))**2)
    loss_mel80 = np.mean((np.dot(mel80, X**2)-np.dot(mel80, Xref**2))**2)
    
    # Write log file
    with open('generated_samples/'+config.dataset_train+'/'+model_name+'_'+config.file+'_'+config.phase+'.txt', 'a') as log_file:
        log_file.write('Loss_wave: {:.6f}\nLoss_spec: {:.6f}\nLoss_wspec: {:.6f}\nLoss_mel27: {:.6f}\nLoss_mel40: {:.6f}\nLoss_mel80: {:.6f}\n'.format(loss_wave, loss_spec, loss_wspec, loss_mel27, loss_mel40, loss_mel80))
    
    print('Audio samples generated.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_train', type=str, default='dcase2017')
    parser.add_argument('--dataset_gen', type=str, default='dcase2017_eval')
    parser.add_argument('--file', type=str, default='a001_0_10')
    parser.add_argument('--phase', type=str, default='gl', help='gl, oracle')
    
    # Experience factors
    parser.add_argument('--target', type=str, default='spec', help='spec, wspec, cqt')
    parser.add_argument('--q', type=int, default=27)
    
    config = parser.parse_args()
    
    generate_and_log_samples(config)
    
