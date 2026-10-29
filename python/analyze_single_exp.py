import matplotlib.pyplot as plt
import numpy
import numpy as np
from spec_utils import *
from BXQY_functions import *
from set_workup_params import *
import re
from pathlib import Path
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog


'''
Script for analyzing the data saved from a pulsed-solution experiment
Analysis is focused on stability in intensity and lifetime. 
Generally not advisable to use data where lifetime dramatically alters over course of an experiment
Sum the stable data and save for comparison to other experiments
'''

'''Set parameters'''
save=True

search_for_valid=True
num_folders=1 #how many separately saved experiments were done on that sample (i.e. was water added?)

read_lt=True
read_BXQY=True

save_folder = None

paths=[]

'''Get the folders you want to read'''
for i in range(num_folders):

    root = tk.Tk()
    initial_dir = None
    corrs_path = filedialog.askdirectory(initialdir=initial_dir, title='Choose folders of saved data in sequential order')  # Open file dialog and get selected file path
    root.withdraw()  # Hide the main window

    paths.append(Path(corrs_path))

'''Initialize the experimental quantities'''
#Values to save from lifetime files
exp_lt_counts = []
exp_mean_lt=np.array([])

#Values to save from correlation files
exp_time=np.array([])
exp_xcorr = []
exp_jcr = np.array([])
exp_num_qds=np.array([])
exp_cr=np.array([])


bin_centers, bin_edges, norm_factor = JH_bin_generator(t_rep, num_synch_bins, later_bin_coarseness,
                                                       later_bin_scaling_factor, max_bin)

'''Search through files in each folder and classify their type'''
filename_pattern = re.compile(
    r"_(LT|BXQY)(?:_file_(\d+))?_slice_(\d+)_segment(\d+)$"
)

for path_index, corrs_path in enumerate(paths):
    grouped_files = defaultdict(lambda: {"LT": [],
                                         "BXQY": []})

    if corrs_path.exists() and corrs_path.is_dir():

        # -------- PASS 1: collect structured metadata --------
        files = []

        for file_path in corrs_path.glob("*"):
            if not file_path.is_file():
                continue

            stem = file_path.stem
            match = filename_pattern.search(stem)

            if match:
                prefix = match.group(1)
                # Capture the file number (default to 0 if it doesn't exist)
                file_num = int(match.group(2)) if match.group(2) else 0
                slice_num = int(match.group(3))
                segment_num = int(match.group(4))

                files.append((file_num, slice_num, segment_num, prefix, file_path))
        print(files)

        # -------- PASS 2: sort globally (file → slice → segment) --------
        # This guarantees files are ordered: file0_seg0, file0_seg1, file1_seg0...
        files.sort(key=lambda x: (x[0], x[1], x[2], x[4]))

        # -------- PASS 3: group by FILE number --------
        for file_num, slice_num, segment_num, prefix, file_path in files:
            grouped_files[file_num][prefix].append(file_path)

    for file_num in sorted(grouped_files.keys()):
        print(f"=== File {file_num} ===")
        prefixes = grouped_files[file_num]



        # Helper to sort by segment, then slice if segment numbers tie
        def seg_slice_key(p):
            m = filename_pattern.search(p.stem)
            # sort by segment first, then slice
            return (int(m.group(4)), int(m.group(3)))

        sorted_lt_files   = sorted(prefixes['LT'], key=seg_slice_key)
        sorted_bxqy_files = sorted(prefixes['BXQY'], key=seg_slice_key)

        print(f"LT files: {len(sorted_lt_files)}")
        print(f"BXQY files: {len(sorted_bxqy_files)}")


        '''Read each file and extract the data'''


        # Now this will print file 0 segment 0, file 0 segment 1, etc.
        if read_lt:
            for file_path in sorted_lt_files:
                print(file_path)
                data = np.loadtxt(file_path, delimiter=',')
                bins = data[:, 0]
                counts = data[:, 1]

                popt, fit, frac_ints, mean_lts = fit_lifetime_n_exp(bins, counts, n=4, plot=False, verbose=False)

                mean_lt=mean_lts[-1]

                exp_lt_counts.append(counts)
                exp_mean_lt=np.append(exp_mean_lt,mean_lt)
        if read_BXQY:
            for i, file_path in enumerate(sorted_bxqy_files):
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

                if path_index==0 and i==0 and file_num==0:
                    start_time=0
                else:
                    start_time=exp_time[-1]+segment_length_s

                exp_time=np.append(exp_time, start_time)
                bxqy, error, fcs_height = calculate_BXQY(tau, xcorr, avg_joint_cr, norm_factor, plot=False, verbose=False)

                exp_xcorr.append(xcorr)
                exp_jcr=np.append(exp_jcr, avg_joint_cr)
                exp_cr=np.append(exp_cr, avg_cr)

print('DONE READING')

#in case the correlation is still running but you want to take a look so far
if len(sorted_bxqy_files)<len(sorted_lt_files):
    exp_lt_counts=exp_lt_counts[:-1]
    exp_mean_lt=exp_mean_lt[:-1]


'''We'll now look at experimental values as function of experimental time'''
coarse_num_qds=[]
coarsen_factor=3

coarse_time=coarsen_array_mean(exp_time, coarsen_factor)
coarse_cr = coarsen_array_mean(exp_cr, coarsen_factor)

coarse_xcorr=coarsen_array_sum(exp_xcorr, coarsen_factor)
coarse_jcr=coarsen_array_sum(exp_jcr, coarsen_factor)


for xcorr, jcr in zip(coarse_xcorr, coarse_jcr):
    _,_, fcs_height=calculate_BXQY(tau, xcorr, jcr, norm_factor, plot=False, verbose=False)
    num_qds=(fcs_height-1)**-1
    coarse_num_qds.append(num_qds)

plot_changes_in_solution(exp_time, exp_cr, exp_mean_lt, coarse_time, coarse_cr, coarse_num_qds)

'''Looking for changes in lifetime is a good indiciation of intrinsic sample degradation. 
   Counts can fluctuate due to changes in concentration.
   Choose an end index where you feel comfortable the sample hasn't changed much '''

if not search_for_valid:
    plot_changes_in_solution(exp_time, exp_cr, exp_mean_lt, coarse_time, coarse_cr, coarse_num_qds, end_index=end_index)
    plot_lifetime_changes(bins, exp_lt_counts)

end_index=-1
while search_for_valid:
    if end_index==-1:
        lts_to_plot=exp_lt_counts
    else:
        lts_to_plot=exp_lt_counts[:end_index]
    print(len(lts_to_plot))

    plot_lifetime_changes(bins, lts_to_plot)

    user_input=int(input('Try new index: (enter 0 to finish) ').lower())
    if user_input==0:
        break
    else:
        end_index=user_input

plot_changes_in_solution(exp_time, exp_cr, exp_mean_lt, coarse_time, coarse_cr, coarse_num_qds, end_index=end_index)

'''Truncate all the experimental arrays to the region where the sample is stable, average to get a final experimental quantity'''

#Values to save from lifetime files
exp_lt_counts = np.array(exp_lt_counts[:end_index])
exp_mean_lt=exp_mean_lt[:end_index]

sum_lt=np.sum(exp_lt_counts, axis=0)


#Values to save from correlation files
exp_time=exp_time[:end_index]
exp_xcorr=np.array(exp_xcorr[:end_index])
exp_jcr=np.array(exp_jcr[:end_index])
exp_num_qds=exp_num_qds[:end_index]
exp_cr=exp_cr[:end_index]

sum_jcr=np.sum(exp_jcr)
sum_xcorr=np.sum(exp_xcorr, axis=0)


'''Save the experimental quantity'''

if save:
    print('Saving...')
    folder_path=Path(save_folder)

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








