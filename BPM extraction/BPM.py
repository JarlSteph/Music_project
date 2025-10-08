import  matplotlib.pyplot as plt 
from pydub import AudioSegment
import numpy as np 
import librosa

class BPM_extractor(): 

    def __init__(self, sr, novelty = "spectral", tempogram = "fourier", 
                 novelty_w: float = 0.1, novelty_hop: float = 0.05,
                 tempo_w: float = 0.1, tempo_hop: float = 0.05):
        self.sr = sr
        self.nfft = 8192
        self.novelty_window = (novelty_w, novelty_hop) # the window used in novelty calc IN SECONDS (window size, hop size)
        self.tempo_window = (tempo_w, tempo_hop)
        self.bpm_max = 200
        self.bpm_min = 60

        if novelty == "spectral": 
            self.novelty_f = self._spectral_novelty
        elif novelty == "energy":
            self.novelty_f = self._energy_novelity
        else: 
            print("use valid novelity choice")

        if tempogram == "fourier":
            self.tempogram = self._fourier_tempogram
        elif tempogram == "correlation":
            self.tempogram = self._autoCorrelation_tempogram
        else: 
            print("not a valid tempogram estimator")
        
    def _energy_novelity(self, sound):
        win_len = int(self.novelty_window[0] * self.sr)
        hop_size = int(self.novelty_window[1] * self.sr)
        window = np.hanning(win_len) 

        frames = np.lib.stride_tricks.sliding_window_view(sound, win_len)[::hop_size]
        E = np.sum((frames * window) ** 2, axis=1) #
        # we want log E: 
        E_log = np.log(E + 1e-10) # underflow
        dE = np.diff(E_log)  # derive
        novelty = np.maximum(dE, 0) # only positive side 
        novelty /= np.max(novelty) 
        return novelty
        
    def _spectral_novelty(self, sound):
        # Number of frames by truncation
        win_len = int(self.novelty_window[0] * self.sr)
        hop_size = int(self.novelty_window[1] * self.sr)

        n_frames = 1 + (len(sound) - win_len) // hop_size
        frames = np.lib.stride_tricks.as_strided(
            sound, shape=(n_frames, win_len),
            strides=(sound.strides[0] * hop_size, sound.strides[0]), writeable=False)
        # Hann window
        window = np.hanning(win_len).astype(np.float64)
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

    def plot_novelty(self, sound, win_size = 0.1, hop = 0.05): 
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

    def _fourier_tempogram(self, novelty: np.ndarray):
        novelty_fs = 1.0 / self.novelty_window[1]  # novelty frames per second

        win_len = int(self.tempo_window[0] * novelty_fs)
        hop_size = int(self.tempo_window[1] * novelty_fs)
        window = np.hanning(win_len)
        frames = np.lib.stride_tricks.sliding_window_view(novelty, win_len)[::hop_size]
        window = np.hanning(win_len)
        frames = frames * window[None, :]

        F = np.fft.rfft(frames, n=self.nfft, axis=1)  # FFT
        mag = np.abs(F)

        # --- build BPM axis from novelty “frequency” (cycles per second) ---
        freqs_hz = np.fft.rfftfreq(self.nfft, d=1.0/novelty_fs)  # periodicity in Hz of novelty
        bpm_axis = freqs_hz * 60.0

        keep = (bpm_axis >= self.bpm_min) & (bpm_axis <= self.bpm_max) # trim within accepted range
        tempogram = mag[:, keep]
        bpm_axis = bpm_axis[keep]

        centers = (np.arange(frames.shape[0]) * hop_size + 0.5 * win_len) / novelty_fs
        time_axis = centers  # seconds

        # Optional normalization (per-timeframe) to emphasize relative peaks:
        # tempogram /= (tempogram.max(axis=1, keepdims=True) + 1e-10)

        return tempogram, bpm_axis, time_axis





        

    def _autoCorrelation_tempogram(self):
        pass

    def get_BPM(self, sound): 
        pass




    def _phase_novelty(self, sound): 
            pass