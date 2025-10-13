def predict_key(audio_path):
    audio, fs = librosa.load(audio_path)

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
    kk_major = norm(kk_major)
    kk_minor = norm(kk_minor)

    #rotate through all tonics, save the best correlation
    best_key, best_mode, best_corr = None, None, -np.inf
    for tonic in range(12):
        maj = np.roll(kk_major, tonic)
        minr = np.roll(kk_minor, tonic)

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
