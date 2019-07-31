import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
import torchaudio.transforms as tt
import time
from tqdm import tqdm
import glob
import os
import os.path

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 file_location=None,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 textureSize=10,
                 block_size=500,
                 compute=0,
                 squeeze=0):

        self.block_size = block_size
        self.dataset_file = dataset_file
        self.textureSize = textureSize
        self.squeeze = squeeze

        if compute: # not os.path.isfile(self.dataset_file+'_1.npy'):
            assert file_location is not None, 'Error: Unspecified location of dataset files.'
            # dataset_dir, _ = os.path.split(self.dataset_file)
            # if not os.path.exists(dataset_dir):
            #     os.makedirs(dataset_dir)
            self.mono = mono
            self.normalize = normalize
            self.sampling_rate = sampling_rate
            self.dtype = dtype
            self.create_dataset(file_location)
        else:
            self.mono = None
            self.normalize = None
            self.sampling_rate = None
            self.dtype = None

        self._length = 0
        self.calculate_length()
        self.train = train

    def create_dataset(self, location):
        print('Creating dataset from audio files at', location)
        files = list_all_audio_files(location)
        processed_files = np.zeros((self.block_size, 513, self.textureSize))
        processed_files_phase = np.zeros((self.block_size, 513, self.textureSize))
        l=0
        t=0
        nb_block = 0
        print(len(files))
        if self.squeeze:
            if 'gtzan' in location:
                files = files[:20]
            else:
                files = files[:20]

        for i, file in enumerate(tqdm(files)):
            file_data, _ = lr.load(path=file,
                                   sr=self.sampling_rate,
                                   mono=self.mono)
            if self.normalize:
                file_data = lr.util.normalize(file_data)

            spec = lr.stft(file_data, 1024, 512, window='hann', center=False)
            data = np.abs(spec)
            phase = np.angle(spec)
    
            for k in range(int(np.floor(data.shape[1]/self.textureSize))):
                processed_files[l, :, :] = data[:, k*self.textureSize:(k+1)*self.textureSize]
                processed_files_phase[l, :, :] = phase[:, k*self.textureSize:(k+1)*self.textureSize]
                l = l+1
                if l==self.block_size:
                    nb_block = nb_block+1
                    l=0
                    np.save(self.dataset_file+'_'+str(nb_block)+'.npy', processed_files)
                    np.save(self.dataset_file+'_'+str(nb_block)+'_phase.npy', processed_files_phase)

        if l!=0:
            np.save(self.dataset_file+'_'+str(nb_block+1)+'.npy', processed_files[0:l, :, :])
            np.save(self.dataset_file+'_'+str(nb_block+1)+'_phase.npy', processed_files_phase[0:l, :, :])

    def get_files(self):
        nb_block = len(glob.glob(self.dataset_file+'_*[0-9].npy'))
        fileNames = [self.dataset_file+'_'+str(i)+'.npy' for i in range(1, nb_block+1)]
        print(fileNames)
        return fileNames

    def calculate_length(self):
        # dataset_dir, _ = os.path.split(self.dataset_file)
        files = glob.glob(self.dataset_file+'_*[0-9].npy')
        # files = [f for f in files if 'spec' in f]
        newest_file = max(files, key=os.path.getctime)
        input_data = np.load(newest_file, mmap_mode='r')

        self._length = (len(files)-1)*self.block_size+input_data.shape[0]


    def __getitem__(self, idx):
        ib = int(np.floor(idx/self.block_size))
        idx = idx-ib*self.block_size
        data = np.load(self.dataset_file+'_'+str(ib+1)+'.npy', mmap_mode='r')[idx, :, :]
        data = torch.from_numpy(data).type(torch.FloatTensor)
        data = data.permute(1, 0)
        return data

    def __len__(self):
        return self._length

def list_all_audio_files(location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", ".aiff", ".flac", ".au"))]:
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print('Error: Found no audio files in ' + location)
    return audio_files
