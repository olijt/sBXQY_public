import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from spec_utils import *
from set_workup_params import *
from BXQY_functions import *
from scipy.fft import fft, ifft, fftfreq

save=True
path1=None

timestamps, channels, channel_names, _, _ =get_exp_data(path1)
channel_signals=create_channels(timestamps, channels, channel_names)

sync_index = channel_names.index(sync_ch_name)
sync_times = channel_signals[sync_index]

# create signals excluding sync timestamps
det_channel_signals = channel_signals.copy()
det_channel_signals.pop(sync_index)

det_channel_names = channel_names.copy()
det_channel_names.pop(sync_index)

shifts=lt_offsets[0]
for i, ch in enumerate(det_channel_signals):
    det_channel_signals[i] = ch - shifts[i]

chT_sync, chT_phot = filter_channels_by_sync_index(det_channel_names, det_channel_signals,
                                                   sync_times)

irf = []
for sync, ch in zip(chT_sync, chT_phot):
    irf_x, lt_signal_y = lifetime(sync, ch, t_rep, plot=False, n_bins=3000)
    irf.append(lt_signal_y)
irf = np.sum(np.array(irf), axis=0)

plt.semilogy(irf_x, irf)
plt.show()

if save:
    stacked_data=np.column_stack([irf_x, irf])
    save_path=None

    save_stacked_data_to_csv(save_path, stacked_data)