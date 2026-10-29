import numpy as np
'''Contains all the parameters for the workup to avoid consistency errors.'''

#sync information
t_rep=2000131
t_rep=1000065
sync_ch_name=1

#offsets to align lifetimes (ps)
shift0 = 0
shift1 = -100
shift2 = -140
shift3 = -200
lt_shifts=[shift0, shift1, shift2, shift3]
lt_delay=115000 #from t=0 to start of lifetime (ps)

lt_offsets=[lt_shifts, lt_delay] #package together for use

#length of segment when calculating corr to remove drift effects
segment_length_ps=1200e12

#bin info for the cross-correlation
num_synch_bins = 5
later_bin_coarseness = 8
later_bin_scaling_factor = 2
max_bin = 1e12
bin_info=[num_synch_bins, later_bin_coarseness, later_bin_scaling_factor, max_bin]

#dark count info (5,6,7,8)
dc_list=[196, 233, 59, 115]
dark_count_info=np.array(dc_list)
