import  matplotlib.pyplot as plt 
from pydub import AudioSegment
import numpy as np 
import librosa

class BPM_extractor(): 

    def __init__(self, sr, novelty = "spectral"):
        self.sr = sr
        if novelty == "spectral": 
            self.novelty_f = self._spectral_novelty
        elif novelty == "energy":
            self.novelty_f = self._energy_novelity
        else: 
            print("use valid novelity choice")
        self.nfft = 8192

        
    def _energy_novelity(self, sound, winsize, hop):
        win_len = int(winsize * self.sr)
        hop_size = int(hop * self.sr)
        window = np.hanning(win_len) 

        frames = np.lib.stride_tricks.sliding_window_view(sound, window)[::hop_size]
        E = np.sum((frames * window) ** 2, axis=1) #
        # we want log E: 
        E_log = np.log(E + 1e-10) # underflow
        dE = np.diff(E_log)  # derive
        novelty = np.maximum(dE, 0) # only positive side 
        novelty /= np.maximum(novelty) 
        return novelty
        
    def _spectral_novelty(self, sound, winsize, hop):
        # Number of frames by truncation
        n_frames = 1 + (len(sound) - winsize) // hop
        frames = np.lib.stride_tricks.as_strided(
            sound, shape=(n_frames, winsize),
            strides=(sound.strides[0] * hop, sound.strides[0]), writeable=False)
        # Hann window
        window = np.hanning(winsize).astype(np.float64)
        frames = frames * window[None, :]

        S = np.fft.rfft(frames, n=self.nfft, axis=1) # only keep non negaitve 
        mag = np.log(np.abs(S) + 1e-10)
        # Spectral flux: positive frame-to-frame differences summed over frequency
        diff = mag[1:] - mag[:-1]
        diff[diff < 0.0] = 0.0
        novelty = diff.sum(axis=1)
        m = novelty.max() 
        novelty = novelty / m # normalize
        return novelty

    def plot_novelty(self, sound, win_size = 0.1, hop = 0.5): 
        time_axis = np.linspace(0, len(sound) / self.sr, num=len(sound), endpoint=False)
        novelty = self.novelty_f(sound, win_size, hop)
        # Create time axis for novelty (centered between frames)
        novelty_time = np.arange(len(novelty)) * hop + (win_size / 2)

        # Plot waveform and novelty
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, sound, color='gray')
        plt.title("Audio waveform")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(novelty_time, novelty, color='orange')
        plt.title(f"{self.novelty_f.__name__[1:].replace('_', ' ').capitalize()} function")
        plt.xlabel("Time [s]")
        plt.ylabel("Novelty")
        plt.tight_layout()
        plt.show()


    def _phase_novelty(self, sound): 
        pass

    def _fourier_tempogram(self):
        pass

    def _autoCorrelation_tempogram(self):
        pass

    def get_BPM(self, sound): 
        pass




