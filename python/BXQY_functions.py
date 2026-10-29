import time
import numpy as np
from spec_utils import *
#numba functions need to be explictly imported
from spec_utils.photon_stream_utils import photons_in_bins, calculate_intensity_trace
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import re

'''For assessing sample quickly'''
def check_stability_and_aggs_global(ttbin_path, sync_name, intensity_trace_path, stability_path, transition_path, agg_path, save=False):

    timestamps, channels, _, _, exp_length = get_exp_data(ttbin_path)

    # Calculate coarse intensity trace
    binwidth = exp_length / 1000
    time, cps, _ = calculate_intensity_trace(timestamps, binwidth=binwidth)
    start_time, end_time = assess_intensity_stability(time,cps)

    timestamps, channels = truncate_photon_stream(timestamps, channels, start_time, end_time)

    user_threshold=int(input('Choose aggregate threshold in cps based on intensity trace (enter 0 to skip): ').lower())

    print(f'Threshold set to: {user_threshold} cps')
    binwidth=1e12
    timestamps, channels, agg_intervals =find_and_remove_aggs(timestamps, channels, binwidth=binwidth, mode='ABS', input_threshold=user_threshold, plot=True)

    time, cps, avg_cps=calculate_intensity_trace(timestamps[channels!=sync_name], 1e12)

    plt.plot(time, cps)
    plt.ylim(0, avg_cps*1.5)
    plt.title('Using this intensity stream')
    plt.show()
    if save:

        np.savetxt(intensity_trace_path, np.column_stack((time, cps)), delimiter=",", header="time,cps", comments="")


    slice_edges=get_slices(timestamps)

    if save:
        np.savetxt(stability_path, np.column_stack((start_time, end_time)), delimiter=",", header="start,end", comments="")
        np.savetxt(transition_path, np.column_stack(slice_edges), delimiter=",")
        agg_intervals = np.array(agg_intervals)  # example

        with open(agg_path, "w") as save_csv:
            save_csv.write(f" Binwidth \n{binwidth}\n")
            np.savetxt(save_csv, agg_intervals, delimiter="," ,comments="")

        print('Saved')
    else:
        print('NOT Saved')
def check_stability_and_aggs_individual(ttbin_paths, intensity_trace_path, stability_path, save=False):
    exp_time_list = []
    exp_cps_list = []

    for path in ttbin_paths:
        timestamps, channels, _, _, exp_length = get_exp_data(path)

        binwidth = exp_length * len(ttbin_paths) / 1000
        time, cps, _ = calculate_intensity_trace(timestamps, binwidth=binwidth)

        exp_time_list.append(time)
        exp_cps_list.append(cps)

    exp_time = np.concatenate(exp_time_list)
    exp_cps = np.concatenate(exp_cps_list)

    start_time, end_time = assess_intensity_stability(exp_time, exp_cps)

    if end_time != -1:
        mask = exp_time <= end_time

        exp_time = exp_time[mask]
        exp_cps = exp_cps[mask]
    if start_time != 0:
        mask = exp_time >= start_time

        exp_time = exp_time[mask]
        exp_cps = exp_cps[mask]

    plt.plot(exp_time, exp_cps)
    plt.title('Using this intensity stream')
    plt.show()

    if save:
        np.savetxt(intensity_trace_path, np.column_stack((exp_time, exp_cps)), delimiter=",", header="time,cps",
                   comments="")
        np.savetxt(stability_path, np.column_stack((start_time, end_time)), delimiter=",", header="start,end",
                   comments="")
        print('Saved')
    else:
        print('NOT Saved')

'''For doing the experimental workup, files, etc.'''
def perform_sBXQY_workup(ttbin_paths, sync_name, t_rep, bin_info,
                         save=False,
                         segment=False, segment_len_ps=None,
                         lt_align=False, lt_offsets=None, dark_count_info=None,
                         use_csvs=True, use_slices=False,
                         calculate_lt=True,
                         calculate_qy=True,
                         ):
    """

    Parameters
    ----------
    ttbin_paths: list of ttbin_paths to analyze
    sync_name: integer value of channel
    t_rep: time between sync pulses in ps
    bin_info: formatted for compatibility with JH_bin_generator
    save: whether to save the data
    segment: whether to slice the photon stream to avoid drift artifacts
    segment_len_ps: length of each slice in ps
    lt_align: whether to align the signals on each detetcor by lifetime (recommended for any lifetime analysis)
    shifts: shifts for the lt alignment (ps)
    use_csvs
    use_slices
    calculate_lt
    calculate_PNRL
    calculate_qy

    Returns
    -------

    """
    for path_idx, ttbin_path in enumerate(ttbin_paths):
        print(f'Analyzing path no. {path_idx}...')
        selected_path = Path(ttbin_path)

        parent_dir = selected_path.parent


        if save:
            corrs_path = parent_dir / "saved_data"
            corrs_path.mkdir(exist_ok=True)

            prefix = selected_path.parent.name
        if use_csvs:
            truncate=True
            filter_aggs=False
            by_slice=use_slices
        else:
            truncate=False
            filter_aggs=False
            by_slice=False


        (all_timestamps,
         all_channels,
         channel_names,
         all_n_photons,
         all_exp_length) = get_stable_sBXQY_data(ttbin_path,
                                                 truncate=truncate,
                                                 filter_aggs=filter_aggs,
                                                 by_slice=by_slice)

        num_slices=len(all_timestamps)
        print(f'No. Slices: {num_slices}')

        for slice_idx in range(num_slices):
            timestamps=all_timestamps[slice_idx]
            channels=all_channels[slice_idx]
            n_photons=all_n_photons[slice_idx]
            exp_length=all_exp_length[slice_idx]

            slice_start=timestamps[0]

            if segment:
                num_segments = int(np.ceil(exp_length / segment_len_ps))
                print('NUM SEGMENTS', num_segments )

                for seg_idx in tqdm(range(num_segments)):

                    start_time = int(slice_start + segment_len_ps * seg_idx)
                    start_index = get_index_for_timestamps(timestamps, start_time)

                    if seg_idx < num_segments - 1:
                        stop_time = int(slice_start + segment_len_ps * (seg_idx + 1))
                        stop_index = get_index_for_timestamps(timestamps, stop_time)
                    else:
                        stop_index = -1

                    #take specific segment of stream without inteferring with original variables
                    loc_timestamps = timestamps[start_index:stop_index]
                    loc_channels = channels[start_index:stop_index]

                    #calculate true segment length and countrate
                    non_sync_stamps=loc_timestamps[loc_channels!=sync_name]
                    n_det_counts=len(non_sync_stamps)
                    segment_len_s = (loc_timestamps[-1]-loc_timestamps[0]) / 1e12 #sometimes segment will be shorter than max segment length
                    countrate=n_det_counts/segment_len_s

                    #create local variables for the segment
                    loc_channel_names = channel_names.copy()
                    sync_index = channel_names.index(sync_name)
                    loc_channel_signals = create_channels(loc_timestamps, loc_channels, loc_channel_names)
                    sync_times = loc_channel_signals[sync_index]

                    #create signals excluding sync timestamps
                    det_channel_signals = loc_channel_signals.copy()
                    det_channel_signals.pop(sync_index)

                    det_channel_names = loc_channel_names.copy()
                    det_channel_names.pop(sync_index)

                    if lt_align:
                        lt_shifts=lt_offsets[0]
                        for i, ch in enumerate(det_channel_signals):
                            det_channel_signals[i]=ch-lt_shifts[i]

                    chT_sync, chT_phot = filter_channels_by_sync_index(det_channel_names, det_channel_signals,
                                                                       sync_times)

                    #for making dark count correction for correlations
                    observed_avg_countrate=np.array([len(ch)/(np.max(ch)-np.min(ch)) for ch in chT_phot]) #counts/ps
                    dark_avg_countrate=dark_count_info/1e12 #counts/ps
                    true_avg_countrate=observed_avg_countrate-dark_avg_countrate #counts/ps

                    if calculate_lt:
                        num_bins=3000
                        all_lt_counts=[]
                        for sync, ch in zip(chT_sync, chT_phot):
                            lt_bins, lt_counts=lifetime(sync, ch, t_rep, plot=False, n_bins=num_bins)
                            all_lt_counts.append(lt_counts)
                        sum_lt_counts=np.sum(np.array(all_lt_counts), axis=0)

                        if save:
                            if len(ttbin_paths)>1:
                                print(f"Saving LT for file {path_idx}, slice {slice_idx}, segment {seg_idx}...")
                                LT_save_name = f"{prefix}_LT_file_{path_idx}_slice_{slice_idx}_segment{seg_idx}.csv"
                            else:
                                print(f"Saving LT for slice {slice_idx}, segment {seg_idx}...")
                                LT_save_name = f"{prefix}_LT_slice_{slice_idx}_segment{seg_idx}.csv"
                            LT_csv_path = corrs_path / LT_save_name
                            np.savetxt(LT_csv_path, np.column_stack((lt_bins, sum_lt_counts)), delimiter=',',
                                       header='t, counts')

                    if calculate_qy:
                        bin_centers, bin_edges, norm_factor = JH_bin_generator(t_rep, *bin_info)

                        xcorr, final_joint_countrate= calculate_final_xcorr(chT_sync, chT_phot,
                                                                            bin_centers, bin_edges, norm_factor,
                                                                            true_avg_countrate, dark_avg_countrate, segment_len_ps,
                                                                            plot=False)
                        if save:
                            if len(ttbin_paths)>1:
                                print(f"Saving BXQY for file {path_idx}, slice {slice_idx}, segment {seg_idx}...")
                                BXQY_save_name = f"{prefix}_BXQY_file_{path_idx}_slice_{slice_idx}_segment{seg_idx}.csv"
                            else:
                                print(f"Saving BXQY for slice {slice_idx}, segment {seg_idx}...")
                                BXQY_save_name = f"{prefix}_BXQY_slice_{slice_idx}_segment{seg_idx}.csv"
                            BXQY_csv_path = corrs_path / BXQY_save_name
                            with open(BXQY_csv_path, "w") as save_csv:
                                save_csv.write(f"Segment Length (s) = {segment_len_s}\n")
                                save_csv.write(f"# Avg CR (cps) = {countrate}\n")
                                save_csv.write(f"# Avg Joint CR (denom) = {final_joint_countrate}\n")
                                np.savetxt(save_csv, np.column_stack((bin_centers, xcorr)), delimiter=",",
                                           header="tau,numerator",
                                           comments="")


            else:
                print('Non-segmented experiment needs configuring')
                raise RuntimeError
def get_stable_sBXQY_data(ttbin_path, truncate=True, filter_aggs=True, by_slice=True):
    # Load and structure data
    print("Loading data...")

    timestamps, channels, channel_names, n_photons, exp_length = get_exp_data(ttbin_path)

    if truncate:
        stability_path = re.sub(r"(?:\.\d+)?\.ttbin$", "_stable_region.csv", str(ttbin_path))
        data = np.loadtxt(stability_path, delimiter=',', skiprows=1)
        start_time, end_time = data[0], data[1]

        timestamps, channels = truncate_photon_stream(timestamps, channels, start_time, end_time)
    if filter_aggs:
        agg_path = re.sub(r"(?:\.\d+)?\.ttbin$", "_agg_ints.csv", ttbin_path)

        with open(agg_path, "r") as f:
            # skip header line
            f.readline()
            # read binwidth
            binwidth_line = f.readline()
            binwidth = float(binwidth_line.strip())

            # read the rest as CSV
            agg_intervals = np.loadtxt(f, delimiter=",")

        agg_intervals = np.asarray(agg_intervals)
        if agg_intervals.ndim == 1:
            agg_intervals = agg_intervals[None, :]  # make it (1, 2)

        timestamps, channels = remove_aggs(timestamps, channels, binwidth, agg_intervals)

    if by_slice:
        transition_path = re.sub(r"(?:\.\d+)?\.ttbin$", "_transition_times.csv", ttbin_path)

        data = np.loadtxt(transition_path, delimiter=',')
        transition_times = data

        if isinstance(transition_times, np.ndarray):
            try:
                n_slices = len(transition_times) + 1
            except TypeError:
                transition_times = [transition_times]
                if transition_times[0]==-1:
                    n_slices = 1
                else:
                    n_slices = 2

        all_timestamps = []
        all_channels = []
        all_n_photons = []
        all_exp_length = []

        print('Num slices ', n_slices)

        for i in range(n_slices):
            print(f'Slice {i}')

            if i == 0:
                start_time = 0
            else:
                start_time = transition_times[i - 1]

            if i == n_slices-1:
                end_time = -1
            else:
                end_time = transition_times[i]
            print(f'start {start_time}  end {end_time}')

            loc_timestamps, loc_channels = truncate_photon_stream(timestamps, channels, start_time, end_time)

            loc_exp_start = loc_timestamps[0]
            loc_exp_length = loc_timestamps[-1] - loc_exp_start
            loc_n_photons = len(loc_timestamps)

            all_timestamps.append(loc_timestamps)
            all_channels.append(loc_channels)
            all_n_photons.append(loc_n_photons)
            all_exp_length.append(loc_exp_length)

        return all_timestamps, all_channels, channel_names, all_n_photons, all_exp_length
    else:
        exp_start = timestamps[0]
        timestamps -= exp_start
        exp_length=timestamps[-1]
        n_photons=len(timestamps)

        return [timestamps], [channels], channel_names, [n_photons], [exp_length]

'''For computing the correlation'''


def JH_bin_generator(t_rep, num_sync_bins=10, later_bin_coarseness=3, later_bin_scaling_factor=2, max_bin=1e12, pulsewise=True):

    # When using sync as the first photon event. Gives bin_edges : [0, t_rep, 2*t_rep, etc ... ]
    if pulsewise:
        sync_bin_edges = t_rep * np.arange(num_sync_bins + 1)

    # When using first detection event as the first photon event. Gives bin edges [0, 0.5*t_rep, 1.5*t_rep, ...]
    else:
        sync_bin_edges = np.zeros(num_sync_bins + 1)
        if num_sync_bins > 0:
            sync_bin_edges[1:] = (np.arange(num_sync_bins) + 0.5) * t_rep



    # 2. Generate later bin edges (the half-integer offset propagates cleanly here)
    later_bin_edges = np.zeros(100_000)
    integer_factor = 1
    for i in range(len(later_bin_edges)):
        if i == 0:
            later_bin_edges[i] = sync_bin_edges[-1]
        else:
            later_bin_edges[i] = later_bin_edges[i - 1] + integer_factor * t_rep
            if (i + 1) % int(later_bin_coarseness) == 0:
                integer_factor *= int(later_bin_scaling_factor)
            if later_bin_edges[i] > max_bin:
                break
    later_bin_edges_filtered = later_bin_edges[later_bin_edges > 0]  # remove excess bins

    # There is a redundant bin edge in the first entry of
    # later_bin_edges, so don't include that in concatenation
    bin_edges = np.hstack((sync_bin_edges, later_bin_edges_filtered[1:]))
    bin_centers=bin_edges_to_bin_centers(bin_edges, mode='linear')

    norm_factor = np.diff(bin_edges)

    return bin_centers, bin_edges, norm_factor
def filter_channels_by_sync_index(channel_names, channel_signals, sync_times):
    '''
    Generates list of sync times for each detector.
    Important to prevent long tau photons on one detector being considered as first after a sync.
    Also eliminates afterpulsing.
    Parameters
    ----------
    channel_names (should not include sync)
    channel_signals: (Nx4) tuple of detected events
    sync_times: all events on the sync channel

    Returns
    -------
    chT_sync: horizonatally concatenated column vectors of all sync times assoc'd with a detector's count
    chT_phot: horizonatally concatenated column vectors of all of a detector's photon times
    '''
    detector_times_list = []
    detector_syncs_list = []

    for i in range(len(channel_names)):
        print(f"Filtering detector {channel_names[i]} data...")

        # Extract timestamps for this detector
        det_times = channel_signals[i]
        # Assign each photon to its preceding sync pulse
        sync_idx = np.searchsorted(sync_times, det_times, side="right") - 1

        # Discard photons that occur before the first sync
        valid = sync_idx >= 0
        det_times = det_times[valid]
        sync_idx = sync_idx[valid]

        # Remove afterpulses (keep first photon per sync)
        _, first_indices = np.unique(sync_idx, return_index=True)

        filtered_times = det_times[first_indices]
        filtered_syncs = sync_times[sync_idx[first_indices]]

        # Store results
        detector_times_list.append(filtered_times)
        detector_syncs_list.append(filtered_syncs)



    return detector_syncs_list, detector_times_list

def calculate_local_xcorr(this_sync, this_phot, bin_centers, bin_edges, norm_factor):
    """
    Standard calculation of photon correlation with custom bin parameters.
    Parameters
    ----------
    this_sync
    this_phot
    bin_centers
    bin_edges
    norm_factor

    Returns
    -------

    """
    print('Calculating correlation...')
    local_xcorr = photons_in_bins(this_sync, this_phot, bin_edges)

    # normalization code from photon_corr
    ch0_min = min(this_sync)
    ch1_min = min(this_phot)

    ch0_max = max(this_sync)
    ch1_max = max(this_phot)

    tag_range = max(ch1_max, ch0_max) - min(ch1_min, ch0_min)  # range of tags in the entire dataset
    infinite_acq_correction = tag_range / (tag_range - bin_centers)  # has greater effect at long tau

    local_xcorr *= infinite_acq_correction  # if your experiment was infinitely long with proportional count rate, how many correlation events in each bin

    local_ch0_count = len(this_sync)
    local_ch1_count = len(this_phot)

    local_joint_counts = local_ch0_count * local_ch1_count / tag_range #two photon event rate = (N/T)^2. Summed over T is N^2/T

    local_norm_xcorr = (local_xcorr / norm_factor) / local_joint_counts

    return local_xcorr, local_norm_xcorr, local_joint_counts

def calculate_final_xcorr(chT_sync, chT_phot,
                          bin_centers, bin_edges, norm_factor,
                          true_avg_countrate, dark_avg_countrate, segment_len_ps,
                          plot=False):
    """

    Parameters
    ----------
    sync_ch_name: integer channel value
    channel_signals: (Nx4) tuple
    channel_names
    t_rep: in ps
    bin_info: must be compatible with JH_bin_generator
    perform_lifetime_alignment
    plot:

    Returns
    -------
    bin_centers, xcorr, final_joint_countrate

    """
    # Calculate cross-correlation
    print("Calculating cross-correlations")

    final_xcorr = np.zeros(len(bin_edges)-1)
    final_joint_counts=0


    num_xcorr_runs = 0
    num_detectors = len(chT_sync)
    for i in range(num_detectors):
        this_sync = chT_sync[i]

        for j in range(num_detectors):
            if i!=j :
                    print((i, j))
                    print(f"Correlating run {num_xcorr_runs+1} of {(num_detectors)*(num_detectors-1)}...")
                    this_phot = chT_phot[j]

                    true_cr_1 = true_avg_countrate[i]
                    true_cr_2 = true_avg_countrate[j]

                    dark_cr_1 = dark_avg_countrate[i]
                    dark_cr_2 = dark_avg_countrate[j]


                    #calculate from the full observed signal, including dark counts
                    local_xcorr, local_norm_xcorr, local_joint_counts= calculate_local_xcorr(this_sync,
                                                                                                this_phot,
                                                                                                bin_centers,
                                                                                                bin_edges,
                                                                                                norm_factor)

                    local_xcorr/=norm_factor #divide by binwidth

                    #adjust for dark counts
                    correction_term = (segment_len_ps*(true_cr_1*dark_cr_2) +
                                       segment_len_ps*(true_cr_2*dark_cr_1) +
                                       segment_len_ps*(dark_cr_1*dark_cr_2))

                    corrected_local_xcorr = local_xcorr-correction_term
                    corrected_local_joint_counts = local_joint_counts - correction_term


                    final_xcorr += corrected_local_xcorr
                    final_joint_counts+=corrected_local_joint_counts


                    num_xcorr_runs += 1


    intensity_corrected_xcorr = final_xcorr / final_joint_counts

    if plot:
        print("Plotting...")
        plt.plot(bin_centers, intensity_corrected_xcorr)
        plt.axhline(1, color='black', ls='--')
        plt.xscale("log")
        plt.xlabel("$\\tau$ (ps)")
        plt.ylabel('$g^{x}(\\tau)$')
        plt.show()

    return final_xcorr, final_joint_counts

'''After computing the correlations'''
def calculate_BXQY(tau, xcorr, joint_cr, norm_factor, error_type='shot', plot=True, verbose=True):
    """

    Parameters
    ----------
    tau
    xcorr
    joint_cr
    norm_factor
    error_type
    plot
    verbose

    Returns
    -------

    """
    corr_counts = xcorr * norm_factor
    normalized_corr = xcorr / joint_cr


    # 3. Handle zeros to avoid division-by-zero warnings
    # (Where counts are 0, error is effectively 0 or undefined)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Relative error of a Poisson process is 1 / sqrt(Counts)
        relative_std = 1.0 / np.sqrt(corr_counts)
        relative_95 = relative_std*1.96
        relative_std[corr_counts == 0] = 0
        relative_95[corr_counts == 0] = 0

    # 4. Apply the relative error to the normalized data
    absolute_std = (normalized_corr * relative_std)
    absolute_95 = (normalized_corr * relative_95)

    lb_std = normalized_corr - absolute_std
    ub_std = normalized_corr + absolute_std

    lb_95 = normalized_corr - absolute_95
    ub_95 = normalized_corr + absolute_95


    # Optional: Prevent lower bound from dipping below zero if physically impossible
    lb_std = np.clip(lb_std, 0, None)
    lb_95 = np.clip(lb_95, 0, None)


    # Exclude the center point when fitting
    fit_tau = tau[1:]
    fit_corr = normalized_corr[1:]

    fit = fit_fcs(fit_tau, fit_corr, plot=False, verbose=verbose)
    fcs_height=fit[0]

    fit_error = np.sqrt(np.mean((normalized_corr[1:10] - fit[1:10]) ** 2))
    shot_noise_std = (ub_std[0] - lb_std[0]) / 2
    shot_noise_95 = (ub_95[0] - lb_95[0]) / 2

    # decide which error to use
    if error_type=='fit':
        error = fit_error
    else:
        error = shot_noise_std



    bxqy = (normalized_corr[0] - 1) / (fcs_height - 1)
    bxqy_error = error/(fcs_height - 1)
    print(f'BXQY: {bxqy * 100:.0f} ± {bxqy_error * 100:.0f} %')

    if plot:
        plt.semilogx(tau/1e12, normalized_corr, color='#159affff', lw=3)
        plt.fill_between(tau/1e12, lb_95, ub_95, color='k', alpha=0.1)
        plt.fill_between(tau/1e12, lb_std, ub_std, color='k', alpha=0.1)
        plt.semilogx(fit_tau/1e12, fit, color='white', ls='--', lw=3)

        # plt.errorbar(
        #     tau[0]/1e12, normalized_corr[0],
        #     yerr=fit_error,
        #     fmt='o',  # marker style (scatter)
        #     capsize=4,  # little caps on error bars
        #     linestyle='none',
        #     label='Fit Noise'
        # )

        plt.errorbar(
            tau[0] / 1e12,
            normalized_corr[0],
            yerr=error,
            fmt='o',  # Marker: circle
            markersize=8,  # Slightly smaller for precision
            markerfacecolor='white',  # Hollow look makes it "pop" without clutter
            markeredgecolor='red',
            markeredgewidth=2,
            capsize=5,  # Smaller caps for a cleaner look
            linestyle='none',
            label='Shot Noise',
            elinewidth=2.5,  # Thinner lines don't hide the trend line
            capthick=2.5,
            ecolor='red'  # Darker red for better contrast
        )

        plt.xlabel('$\\tau$ (s)', size=16)
        plt.ylabel('$g^\\times(\\tau) $', size=16)
        plt.xticks(size=16)
        plt.yticks(size=16)
        # plt.legend()
        plt.xscale('log')
        plt.tight_layout()
        # if savefig:
        #     plt.savefig(fname=fname, dpi=1200)
        plt.show()

    return bxqy, bxqy_error, fcs_height

def plot_changes_in_solution(exp_time, exp_cr, exp_mean_lt, coarse_time, coarse_cr, coarse_num_qds, end_index=None):


    x_axis=exp_time/3600
    coarse_x_axis=coarse_time/3600

    counts_per_dot=coarse_cr/coarse_num_qds

    all_cr_rel = (exp_cr - exp_cr[0]) / exp_cr[0] * 100
    all_qds_rel = (coarse_num_qds - coarse_num_qds[0]) / coarse_num_qds[0] * 100
    all_cps_per_qd_rel = (counts_per_dot - counts_per_dot[0]) / counts_per_dot[0] * 100
    mean_lts_rel = (exp_mean_lt - exp_mean_lt[0]) / exp_mean_lt[0] * 100

    largest_change = np.max((
        np.max(np.abs(all_cr_rel)),
        np.max(np.abs(all_qds_rel)),
        np.max(np.abs(all_cps_per_qd_rel)),
        np.max(np.abs(mean_lts_rel))
    ))
    ylim = largest_change * 1.1

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 7))

    ax[0].plot(x_axis, exp_cr, marker='o')
    ax[0].set_ylabel('Intensity (cps)')

    ax[1].plot(coarse_x_axis, coarse_num_qds, marker='o')
    ax[1].set_ylabel('# QDs from FCS')

    ax[2].plot(coarse_x_axis, counts_per_dot, marker='o')
    ax[2].set_ylabel('Intensity/QD (cps)')

    ax[3].plot(x_axis, exp_mean_lt, marker='o')
    ax[3].set_ylabel('Mean Lifetime (ns)')
    ax[3].set_xlabel('Time (hr)')
    if end_index is not None:
        for graph in ax:
            graph.axvline(x_axis[end_index], color='k', ls='--')



    fig, ax = plt.subplots()

    ax.plot(x_axis,all_cr_rel, marker='o', label='Intensity')
    ax.plot(coarse_x_axis,all_qds_rel, marker='o', label='# QDs')
    ax.plot(coarse_x_axis,all_cps_per_qd_rel, marker='o', label='QD Brightness')
    ax.plot(x_axis,mean_lts_rel, marker='o', label='Mean LT')

    if end_index is not None:
        ax.axvline(x_axis[end_index], color='k', ls='--')

    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('$\\Delta$ (%)')
    plt.legend()
    plt.show()

def plot_lifetime_changes(lt_bins, all_lt_counts):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    cmap = plt.cm.turbo
    colors = cmap(np.linspace(0.1, 1.0, len(all_lt_counts)))  # avoid very light near-white

    init_lifetime = None

    for i in range(len(all_lt_counts)):
        # 1. Normalize current trace
        baseline=np.mean(all_lt_counts[i][:10])
        norm_lifetime=normalize_lifetime_for_log_plot(lt_bins, all_lt_counts[i], baseline)

        # 2. Plot on ax[0] and capture the line object
        line, = ax[0].semilogy(lt_bins, norm_lifetime, label=f'Trace {i}', color=colors[i])

        # Get the color assigned by matplotlib for this specific trace
        trace_color = line.get_color()

        if i == 0:
            init_lifetime = norm_lifetime
            # Optional: Plot a horizontal zero-line on ax[1] using the first trace's color
            # to represent the baseline reference
            ax[1].axhline(0, color=trace_color, ls='--', alpha=0.7, label='Baseline (Trace 0)')
        else:
            # 3. Plot difference on ax[1] using the exact same color
            ax[1].plot(lt_bins, norm_lifetime - init_lifetime, color=trace_color, label=f'Trace {i} - 0')
            ax[1].set_ylim(-0.1,0.1)

    # Setup labels and titles
    ax[0].set_title("Normalized Lifetimes")
    ax[0].set_xlabel("Time (ns)")
    ax[0].set_ylabel("Normalized Counts")

    ax[1].set_title("Difference from Baseline (Trace 0)")
    ax[1].set_xlabel("Time (ns)")
    ax[1].set_ylabel("Δ Normalized Counts")

    plt.tight_layout()
    plt.show()