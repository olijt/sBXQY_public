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


'''
Script for analyzing series of stable data saved from multiple pulsed-solution experiments
Analysis is focused on stability in intensity and lifetime across experiments
Sum the stable data and save for analysis in STEP_06
'''

root = tk.Tk()
initial_dir = None
corrs_path = filedialog.askdirectory(initialdir=initial_dir)  # Open file dialog and get selected file path
root.withdraw()  # Hide the main window

save=True

read_lt=True
read_BXQY=True


t_rep=2000131

corrs_path=Path(corrs_path)


bin_centers, bin_edges, norm_factor = JH_bin_generator(t_rep, num_synch_bins, later_bin_coarseness,
                                                       later_bin_scaling_factor, max_bin)


filename_pattern = re.compile(r"_(LT|BXQY$")

grouped_files = defaultdict(list)
if corrs_path.exists() and corrs_path.is_dir():
    for file_path in corrs_path.glob("*"):
        if file_path.is_file():
            match = filename_pattern.search(file_path.stem)
            if match:
                prefix = match.group(1)  # 'LT' or 'BXQY'
                grouped_files[prefix].append(file_path)

lt_files = grouped_files['LT']
bxqy_files = grouped_files['BXQY']


print(f"Total LT files found: {len(lt_files)}")
print(f"Total BXQY files found: {len(bxqy_files)}")

'''Initialize the experimental quantities'''
#Values to save from lifetime files
sample_lt_counts = []
sample_mean_lt=np.array([])

#Values to save from correlation files
sample_time=np.array([])
sample_xcorr = []
sample_jcr = np.array([])
sample_num_qds=np.array([])
sample_cr=np.array([])

sample_bxqy=np.array([])
sample_error=np.array([])

if read_lt:
    for i, file_path in enumerate(lt_files):
        data = np.loadtxt(file_path, delimiter=',')
        bins = data[:, 0]
        counts = data[:, 1]

        sample_lt_counts.append(counts)
if read_BXQY:
    for i, file_path in enumerate(bxqy_files):
        with open(file_path, "r") as f:
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

        sample_xcorr.append(xcorr)
        sample_jcr=np.append(sample_jcr,avg_joint_cr)

        sample_bxqy=np.append(sample_bxqy, bxqy*100)
        sample_error=np.append(sample_error, error*100)

'''The main thing we want to check is that the lifetime is consistent between runs'''
for counts in sample_lt_counts:
    norm_counts=normalize_lifetime_for_log_plot(bins, counts, np.median(counts[50:100]))
    plt.semilogy(bins, norm_counts)
plt.show()

#Values to save from lifetime files
sample_lt_counts = np.array(sample_lt_counts)

sum_lt=np.sum(sample_lt_counts, axis=0)


#Values to save from correlation files
sample_xcorr=np.array(sample_xcorr)
sample_jcr=np.array(sample_jcr)

sum_jcr=np.sum(sample_jcr)
sum_xcorr=np.sum(sample_xcorr, axis=0)

'''Save the experimental quantity'''
if save:
    print('Saving...')
    folder_path = Path(save_folder)

    date_info = corrs_path.parent.name
    LT_save_name = f"{date_info}_LT_stable_lt.csv"
    BXQY_save_name = f"{date_info}_BXQY_stable_lt.csv"

    BXQY_csv_path = folder_path / BXQY_save_name
    LT_csv_path = folder_path / LT_save_name

    if read_lt:
        np.savetxt(LT_csv_path, np.column_stack((bins, sum_lt)), delimiter=',', header='t, counts')
    if read_BXQY:
        with open(BXQY_csv_path, "w") as save_csv:
            save_csv.write(f"Segment Length (s) = {segment_length_ps / 1e12}\n")
            save_csv.write(f"# Avg CR (cps) = {0}\n")
            save_csv.write(f"# Avg Joint CR (denom) = {sum_jcr}\n")
            np.savetxt(save_csv, np.column_stack((bin_centers, sum_xcorr)), delimiter=",", header="tau,numerator",
                       comments="")