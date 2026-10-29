import matplotlib.pyplot as plt

from spec_utils import *
from set_workup_params import t_rep, sync_ch_name
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
initial_dir = None
ttbin_path = filedialog.askopenfilename(initialdir=initial_dir)  # Open file dialog and get selected file path
root.withdraw()  # Hide the main window

timestamps, channels, channel_names, n_photons, exp_length=get_exp_data(ttbin_path)
channel_signals = create_channels(timestamps, channels, channel_names)


sync_index = channel_names.index(sync_ch_name)

sync_times = channel_signals[sync_index]

det_channel_signals = channel_signals.copy()
det_channel_signals.pop(sync_index)

get_lifetime_offsets(sync_times, det_channel_signals, t_rep)

