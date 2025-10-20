import os
import sys
os.chdir("D:/Documents/DT2470 Music Informatics/Music_project")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd())))

# Input: path to folder of songs
# Output: Order of songs
# TODO: error handling for file input

import FreeSimpleGUI as sg
from src.model.playlist_optimizer import create_optimized_path
import pandas as pd

def is_valid_file(filepath):
    path = filepath.strip()
    return os.path.isfile(path)

# Define the window's contents
layout = [  [sg.Text("Which file of songs do you want to order?")],     # Part 2 - The Layout
            [sg.Input(key="-INPUT-")],
            [sg.Multiline(size=(60,20), key='-OUTPUT-')],
            [sg.Button('Ok'), sg.Button('Quit')]]

window = sg.Window('Playlist sorter', layout, resizable=True)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break

    path = values['-INPUT-']
    if is_valid_file(path):
    
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns] # Remove space, make lowercase

        original_order = df['song'].tolist()
        
        permutation, total_cost = create_optimized_path(path)
        optimized_order = [df.iloc[i]['song'] for i in permutation]

        # Make a big string with all songs in the order separated by return
        original_str = "\n".join([f"Song {i + 1}: {song}" for i, song in enumerate(original_order)])
        optimized_str = "\n".join([f"Song {i + 1}: {song}" for i, song in enumerate(optimized_order)])
        
        # Put them thangs together
        output_text = f"Original order:\n{original_str}\n\nOptimized order:\n{optimized_str}"

        window['-OUTPUT-'].update(output_text)
    
    else:
        window['-OUTPUT-'].update(f"The filepath {values['-INPUT-']} is invalid")

# Finish up by removing from the screen
window.close()