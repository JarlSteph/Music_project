import os
import sys
os.chdir("D:/Documents/DT2470 Music Informatics/Music_project")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd())))

import PySimpleGUI as sg


# 1- the layout

layout = [[sg.Text('My one-shot window.')],      
          [sg.InputText(key='-IN-')],      
          [sg.Submit(), sg.Cancel()]]      

# 2 - the window

window = sg.Window('Window Title', layout)    

# 3 - the read
event, values = window.read()    

# 4 - the close
window.close()

# finally show the input value in a popup window
sg.popup('You entered', values['-IN-'])
