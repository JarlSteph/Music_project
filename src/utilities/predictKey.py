import numpy as np 
import librosa
from pydub import AudioSegment


# deals with 
#[src/libmpg123/id3.c:process_comment():587] error: No comment text / valid description?

def predict_key(audio_path):
    
    sound = AudioSegment.from_file(audio_path) # chnaged hewr from lirosa to pydub
    if sound.channels == 2:
        sound = sound.set_channels(1)
    audio =np.array(sound.get_array_of_samples()).astype(np.float32)
    fs = sound.frame_rate

    #compute chromagram
    n_fft = 4410
    hop_size = 2205
    chroma = librosa.feature.chroma_stft(y=audio, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_fft=n_fft)
    
    def norm(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    #compute pitch class profile and normalize
    pcp = chroma.mean(axis=1)
    pcp = norm(pcp)

    #Krumhansl–Schmuckler templates
    kk_major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
    kk_minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)

    #Adjusted templates
    ak_major = np.array([6.5,2.0,3.48,2.30,4.5,4.0,2.5,5.19,2.39,3.68,2.29,3.40], dtype=float)
    ak_minor = np.array([6.42,2.68,3.50,5.38,2.60,3.51,2.53,5.25,3.99,2.69,4.25,3.18], dtype=float)
    ak_major = norm(ak_major)
    ak_minor = norm(ak_minor)

    #rotate through all tonics, save the best correlation
    best_key, best_mode, best_corr = None, None, -np.inf
    for tonic in range(12):
        maj = np.roll(ak_major, tonic)
        minr = np.roll(ak_minor, tonic)

        cmaj = np.dot(pcp, maj) / (np.linalg.norm(pcp)*np.linalg.norm(maj) + 1e-12)
        cmin = np.dot(pcp, minr) / (np.linalg.norm(pcp)*np.linalg.norm(minr) + 1e-12)

        if cmaj > best_corr:
            best_key, best_mode, best_corr = tonic, 'major', cmaj
        if cmin > best_corr:
            best_key, best_mode, best_corr = tonic, 'minor', cmin

    names = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    key_mode= (names[best_key] +" "+ best_mode)
    

    #map from musical key to Camelot notation
    camelot_map = {
        "C major": "8B", "A minor": "8A",
        "G major": "9B", "E minor": "9A",
        "D major": "10B", "B minor": "10A",
        "A major": "11B", "F# minor": "11A",
        "E major": "12B", "C# minor": "12A",
        "B major": "1B", "Ab minor": "1A",
        "F# major": "2B", "Eb minor": "2A",
        "C# major": "3B", "Bb minor": "3A",
        "Ab major": "4B", "F minor": "4A",
        "Eb major": "5B", "C minor": "5A",
        "Bb major": "6B", "G minor": "6A",
        "F major": "7B", "D minor": "7A",
    }

    camelot_key = camelot_map.get(key_mode, "Unknown")
   
    return camelot_key
