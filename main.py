from src.utilities.BPM import BPM_extractor
from src.utilities.predictKey import predict_key
from src.utilities.transition_cost import *
from src.utilities.cost_matrix import CostMatrix

from pathlib import Path
import pandas as pd 
from pydub import AudioSegment
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing as tsp_annealing
import  matplotlib.pyplot as plt 
from tqdm import tqdm




DATA_PATH = Path("data/KEY-BPM_train-val")
#DATA_PATH = Path("data/quick_test")


class sound_C(): 
    def __init__(self, path):
        sound = AudioSegment.from_file(path)
        self.fs = sound.frame_rate
        if sound.channels == 2:
            sound = sound.set_channels(1)
        # normalize: 
        audio = np.array(sound.get_array_of_samples())
        audio = audio / np.max(np.abs(audio))
        self.sound = audio
        
    def shorten(self, start = 0.0, end = 1.1): # in secs
        self.sound = self.sound[int(start* self.fs):int(end*self.fs)]
    
    def give_short(self, start = 0.0, end = 1.1):
        return self.sound[int(start* self.fs):int(end*self.fs)]

    def __len__(self):
        return len(self.sound)


def loop(path: str, novelty = "spectral", tempogram = "fourier"):
    p_name = []
    bpms = []
    keys = []

    files = list(Path(path).glob("*.[mw][ap][3v]"))
    for file in tqdm(files, desc="Processing songs"):
        sound = sound_C(file)
        sr, audio = sound.fs, sound.sound
        Extractor = BPM_extractor(
            sr,
            novelty=novelty,
            tempogram=tempogram,
            tempo_w=30, tempo_hop=3, novelty_w = 5.0, novelty_hop= 0.05 #0.3 is max, wonder why 
        )
        bpm = Extractor.get_BPM(audio, plot=False)
        bpms.append(bpm)

        pred=predict_key(file)
        keys.append(pred)

        p_name.append(file.stem) # just song name

    return p_name, bpms, keys



def plot_ordered_songs(df, cost_matrix, best_perm, best_cost, plot=True):
    """Visualize the ordered transitions and their costs."""
    ordered_songs = []
    for i in range(len(best_perm) - 1):
        i1, i2 = best_perm[i], best_perm[i + 1]
        song1, song2 = df.iloc[i1], df.iloc[i2]
        c = cost_matrix.get_cost(i1, i2)
        ordered_songs.append({"from": song1["Name"], "to": song2["Name"], "cost": c})

    ordered_df = pd.DataFrame(ordered_songs)
    if plot: 
        # Plot transition costs
        plt.figure(figsize=(12, 5))
        plt.plot(ordered_df["cost"], marker="o")
        plt.xticks(range(len(ordered_df)), ordered_df["to"], rotation=90)
        plt.title("Transition Cost Between Ordered Songs (Heuristic TSP)")
        plt.xlabel("Song Transition")
        plt.ylabel("Cost")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.text(0.02, max(ordered_df["cost"]) * 0.9, f"Total Cost: {best_cost:.2f}")
        plt.tight_layout()
        plt.show()

    return 


def plot_bpm_key_variation(df):
    """
    Plot BPM and Key of each song, and their variation to the next song.
    Expects df with columns ['Name', 'BPM', 'Key'] in playlist order.
    """

    # numeric key encoding (A=0, A#/Bb=1, ..., G#=11) + mode (major/minor)
    def encode_key(key):
        if pd.isna(key):
            return np.nan
        note_map = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
                    'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
        try:
            base = ''.join([c for c in key if c.isalpha() or c == '#'])
            mode = 0 if key.endswith('m') or key.endswith('moll') else 1
            return note_map.get(base, np.nan) + (0 if mode else 0.5)
        except:
            return np.nan

    df = df.copy()
    df["Key_num"] = df["Key"].apply(encode_key)

    # differences
    df["ΔBPM"] = df["BPM"].diff().abs()
    df["ΔKey"] = df["Key_num"].diff().abs()

    fig, ax1 = plt.subplots(figsize=(12,6))

    # BPM line
    ax1.plot(df["Name"], df["BPM"], color="tab:blue", marker="o", label="BPM")
    ax1.set_ylabel("BPM", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # second axis for key
    ax2 = ax1.twinx()
    ax2.plot(df["Name"], df["Key_num"], color="tab:orange", marker="s", label="Key")
    ax2.set_ylabel("Key (encoded)", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # annotate changes
    for i in range(1, len(df)):
        ax1.text(i, df["BPM"].iloc[i]+1, f"Δ{df['ΔBPM'].iloc[i]:.1f}", 
                 color="tab:blue", fontsize=8, ha="center")
        ax2.text(i, df["Key_num"].iloc[i]+0.1, f"Δ{df['ΔKey'].iloc[i]:.1f}", 
                 color="tab:orange", fontsize=8, ha="center")

    plt.title("BPM and Key Progression Across Playlist")
    plt.xticks(rotation=90)
    plt.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()




def main(path = None):
    path = path or DATA_PATH

    # we work with one central DF: 
    df = pd.DataFrame()
    print("Extracting features")
    p_name, bpms, keys=loop(path)
    df["Name"] = p_name ; df["BPM"] = bpms ; df["Key"] = keys

    print("Finding the optimal order")
    cost_matrix = CostMatrix(df)
    cost_matrix.compute_matrix()
    best_cost = float("inf")

    if len(df) < 3:
        print("Not enough songs to solve TSP.")
        return

    best_cost = float("inf")
    for _ in range(50):
        perm, cost = tsp_annealing(cost_matrix.matrix)
        if cost < best_cost:
            best_perm, best_cost = perm, cost
    ordered_df = df.iloc[best_perm].reset_index(drop=True)  # full song info, ordered
    #plot_ordered_songs(df, cost_matrix, best_perm, best_cost, plot=False) # plot = False just to not show it 
    print(ordered_df)
    plot_bpm_key_variation(ordered_df)



if __name__ == "__main__":
    main()
    