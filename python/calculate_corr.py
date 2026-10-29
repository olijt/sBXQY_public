from BXQY_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from spec_utils.photon_stream_utils import get_photons
import time
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")
import os
import re
from spec_utils import *
from set_workup_params import *


lt_align=True
truncate=True
filter_aggs=True

segment=True
save=False
use_csvs=True
use_slices=False

if segment:
    segment_len_ps = segment_length_ps
else:
    segment_len_ps=None



root = tk.Tk()
initial_dir = None
ttbin_path = filedialog.askopenfilename(initialdir=initial_dir)  # Open file dialog and get selected file path
root.withdraw()  # Hide the main window

perform_sBXQY_workup([ttbin_path], sync_ch_name, t_rep, bin_info,
                     save=save, segment=segment, segment_len_ps=segment_len_ps,
                     lt_align=lt_align, lt_offsets=lt_offsets, dark_count_info=dark_count_info,
                     use_csvs=use_csvs, use_slices=use_slices,
                     calculate_lt=True,
                     calculate_qy=True)







