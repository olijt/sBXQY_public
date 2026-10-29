import matplotlib.pyplot as plt
import numpy
import numpy as np
from spec_utils import *
from BXQY_functions import *
import re
from pathlib import Path
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
from set_workup_params import *
import matplotlib as mpl

bin_centers, bin_edges, norm_factor = JH_bin_generator(t_rep, num_synch_bins, later_bin_coarseness,
                                                       later_bin_scaling_factor, max_bin)

'''Script for analyzing the cumulative BXQY from the experiment'''

bx_file_path=None

print(bx_file_path)

with open(bx_file_path, "r") as f:
    segment_length_s_line = f.readline()
    segment_length_s=float(segment_length_s_line.split('=')[-1].strip())
    avg_cr_line = f.readline()
    avg_cr=float(avg_cr_line.split('=')[-1].strip())
    avg_joint_cr_line = f.readline()
    avg_joint_cr = float(avg_joint_cr_line.split('=')[-1].strip())
    data = np.loadtxt(f, delimiter=",", skiprows=1)
tau = data[:, 0]
xcorr = data[:, 1]

bxqy, error, fcs_height = calculate_BXQY(tau, xcorr, avg_joint_cr, norm_factor, plot=True, verbose=False)

