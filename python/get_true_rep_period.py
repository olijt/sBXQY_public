from spec_utils import *

def run_laser_rep_analysis():
    file=OT_flexible_file_reader()
    timestamps, channels, channel_names, n_photons, exp_time=get_exp_data(file)
    channel_signals=create_channels(timestamps, channels, channel_names)

    extract_precise_period(channel_signals[0], 0.5e6, 500)
    extract_precise_period(channel_signals[0], 1e6, 500)
    extract_precise_period(channel_signals[0], 2e6, 500)

if __name__ == "__main__":
    run_laser_rep_analysis()
