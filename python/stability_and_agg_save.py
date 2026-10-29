import numpy as np

from BXQY_functions import *
from spec_utils import *
from pathlib import Path
import re
import os
import matplotlib
matplotlib.use("Qt5Agg")
import tkinter as tk
from tkinter import filedialog
from set_workup_params import sync_ch_name

save=True

root = tk.Tk()
initial_dir = None
ttbin_path = filedialog.askopenfilename(initialdir=initial_dir)  # Open file dialog and get selected file path
root.withdraw()  # Hide the main window

intensity_trace_path = ttbin_path.replace('.ttbin', '_intensity_trace.csv')
stability_path = ttbin_path.replace('.ttbin', '_stable_region.csv')
transition_path = ttbin_path.replace('.ttbin', '_transition_times.csv')
agg_path = ttbin_path.replace('.ttbin', '_agg_ints.csv')


check_stability_and_aggs_global(ttbin_path, sync_ch_name, intensity_trace_path, stability_path, transition_path, agg_path, save=save)
