import sys
import numpy as np
import librosa as lr
from tqdm import tqdm

class ThirdOctaveTransform():
    def __init__(self, sr=32000):
        # Constants: process parameters
        self.sr = sr
        self.l_frame = int(4096*self.sr/32000)
        l_fft = int(self.l_frame/2+1)
        
        # Third-octave band analysis weights
        self.f = []
        self.H = []
        with open("tob_4096.txt") as w_file:
            for line in w_file: # = For each band
                line = line.strip()
                l_temp = line.split(',')
                # Beginning and end indices
                f_temp = [int(i) for i in l_temp[:2]]
                # Weight array (variable length)
                H_temp = [float(i) for i in l_temp[2:]]
                # Only add weights if they are in available frequencies
                if f_temp[0] < l_fft:
                    if f_temp[1] > l_fft:
                        diff_f = f_temp[1]-l_fft
                        f_temp[1] = l_fft
                        H_temp = H_temp[:-diff_f]
                    self.f.append(f_temp)
                    self.H.append(H_temp)
        
        # Third-octave band synthesis weights
        self.iH = np.genfromtxt("tob_4096_i.txt", delimiter=',')
        self.iH = self.iH[:l_fft, :len(self.f)]
        
        # Declarations/Initialisations
        self.w = np.ones(self.l_frame)
        self.fft_norm = np.sum(np.square(self.w))/self.l_frame
        
    def wave_to_third_octave(self, x, l_hop=None):
        if l_hop == None:
            l_hop = int(self.l_frame)
        
        if (x.shape[0]-self.l_frame)%l_hop != 0:
            x = np.append(x, np.zeros(l_hop-(x.shape[0]-self.l_frame)%l_hop))
        
        n_frames = int(np.floor((x.shape[0]-self.l_frame)/l_hop+1));
        
        X_tob = np.zeros((len(self.f), n_frames))
        # Process
        for ind_frame in range(n_frames):
            # Squared magnitude of RFFT
            X = np.fft.rfft(x[ind_frame*l_hop:ind_frame*l_hop+self.l_frame]*self.w)
            X = np.square(np.absolute(X))/self.fft_norm
            # Third-octave band analysis
            for ind_band in range(len(self.f)):
                X_tob[ind_band, ind_frame] = 0
                X_tob[ind_band, ind_frame] = X_tob[ind_band, ind_frame] + np.dot(X[self.f[ind_band][0]-1:self.f[ind_band][1]], self.H[ind_band])
                if X_tob[ind_band, ind_frame] == 0:
                    X_tob[ind_band, ind_frame] = 1e-15
            # dB SPL
            X_tob[:, ind_frame] = 10*np.log10(X_tob[:, ind_frame])
        return X_tob
        
    def third_octave_to_spec(self, X_tob):
        # Third-octave bands to spectrogram
        X_tob = np.power(10, X_tob/10) # Linear amplitude
        X = np.dot(self.iH, X_tob)
        # Magnitude spectrogram to STFT
        X = np.sqrt(X*self.fft_norm)
        return X
        
    def wave_to_spectrogram(self, x, l_frame_spec, l_hop_spec, phase=False):
        if (x.shape[0]-self.l_frame)%self.l_hop != 0:
            x = np.append(x, np.zeros(self.l_hop-(x.shape[0]-self.l_frame)%self.l_hop))
        X = np.abs(lr.core.stft(x, l_frame_spec, l_hop_spec, center=False))
        if phase==False:
            X = np.abs(X)
        return X
        
def griffin_lim(X, l_frame, l_hop, n_iters):
    l_wave = int(l_hop*(X.shape[1]-1) + l_frame)
    x = np.random.randn(l_wave)
    for i in tqdm(range(n_iters)):
        X_n = lr.stft(x, l_frame, l_hop, window='rect', center=False)
        X_temp = X*np.exp(1.0j*np.angle(X_n))
        x = lr.istft(X_temp, l_hop, window='rect', center=False)
    return x
    
