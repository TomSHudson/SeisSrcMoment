#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Functions to perform coda-based amplitude processsing.

# Input variables:

# Output variables:

# Created by Tom Hudson, 31st March 2021

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import sys, os
import glob
from obspy import UTCDateTime as UTCDateTime
import gc
from scipy.optimize import curve_fit
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)
from SeisSrcMoment import moment, moment_linear_reg # Imports neccessary submodules from SeisSrcMoment



# ------------------- Define functions -------------------
def moving_average(x, win_samp):
    return np.convolve(x, np.ones(win_samp), 'valid') / win_samp


def linear_reg_func(x, m, c):
    """Linear regression function."""
    return (m*x) + c 


def calc_Qc(nonlinloc_event_hyp_filenames, mseed_filenames=None, mseed_dir=None, st=None, filt_freqs=None, t1_lapse=13.5, coda_win_s=15., T_moving_av_win=3., freq_approx=5., alpha_gs=2., verbosity_level=0):
    """Function to calculate coda attenuation quality factor, Qc.
    Note: Currently only applied for S-wave coda.
    
    alpha_gs = 1.5 # Geometrical spreading term. 1.5 for 3D diffusion theory, 2 for single-scattering body wave coda (see Wang2019, p580)
    """
    # Perform initial checks:

    # Setup datastores:
    event_obs_dicts = {}

    # Calculate Qc for all events and all stations:
    # Loop over events:
    for i in range(len(nonlinloc_event_hyp_filenames)):
        if verbosity_level > 0:
            print("Processing for event:", nonlinloc_event_hyp_filenames[i])

        event_obs_dict = {}

        # 1. Import mseed data, nonlinloc data:
        # 1.a. Import nonlinloc data:
        nonlinloc_event_hyp_filename = nonlinloc_event_hyp_filenames[i]
        try:
            nonlinloc_hyp_file_data = read_nonlinloc.read_hyp_file(nonlinloc_event_hyp_filename)
        except ValueError:
            if verbosity_level > 0:
                print("Warning: Cannot get event info from nlloc for event:", nonlinloc_event_hyp_filename, "therefore skipping event...")
            continue
        # 1.b. Import mseed data:
        if not st:
            if mseed_dir:
                st = moment_linear_reg.load_st_from_archive(mseed_dir, nonlinloc_hyp_file_data.origin_time, win_before_s=10, win_after_s=80)
            else:
                if mseed_filenames:
                    st = moment.load_mseed(mseed_filenames[i], filt_freqs)
        # 1.c. Check mseed data:
        # Check if any components are labelled ??1 or ??2, and relabel N and E just for magnitude processing (for rotation below):
        st = moment.relabel_one_and_two_channels_with_N_E(st)
        # Try and trim the stream to a smaller size (if using database mseed):
        origin_time = nonlinloc_hyp_file_data.origin_time
        st.trim(starttime=origin_time - 10.0, endtime=origin_time + 80.0)
        # And force allignment of traces (if out by less than 1 sample):
        st = moment.force_stream_sample_alignment(st)
        st = moment.check_st_n_samp(st)

        # 2. Find amplitude decay of coda:
        # (for all stations in stream that have a phase pick)
        # 2.a. Trim stream about coda lapse times:
        st_coda = st.copy()
        del st
        gc.collect()
        st_coda.trim(starttime=nonlinloc_hyp_file_data.origin_time+t1_lapse, endtime=nonlinloc_hyp_file_data.origin_time+t1_lapse+coda_win_s)
        # 2.b. Loop over stations, calculating coda power decrease, and hence Qc through coda:
        stations_to_process = []
        for tr in st_coda:
            if tr.stats.station not in stations_to_process:
                stations_to_process.append(tr.stats.station)
        for station in stations_to_process:
            # Get power data:
            try:
                tr_N = st_coda.select(station=station, channel="??N")[0]
                tr_E = st_coda.select(station=station, channel="??E")[0]
            except:
                continue
            power_N = moving_average(tr_N.data**2, int(T_moving_av_win*tr_N.stats.sampling_rate))
            power_E = moving_average(tr_E.data**2, int(T_moving_av_win*tr_E.stats.sampling_rate))
            power_NE = power_N + power_E 

            # Calculate power ratio (using a linear regression through time):
            t = (t1_lapse - (T_moving_av_win/2)) + (np.arange(len(power_N)) / tr_N.stats.sampling_rate) #t1_lapse, t1_lapse+coda_win_s, 
            popt, pcov = curve_fit(linear_reg_func, t, np.log10(power_NE))
            t_solns = np.array([t1_lapse + (T_moving_av_win/2), t1_lapse + coda_win_s - (T_moving_av_win/2)])
            log_powers_out_lin_reg = linear_reg_func(t_solns, *popt)

            # And calculate Qc:
            P_0 = 10**log_powers_out_lin_reg[0]
            P_1 = 10**log_powers_out_lin_reg[1]
            t_0 = t_solns[0] # Lapse time (secs after origin time)
            t_1 = t_solns[1]
            # Rough calculation of Qc (from Eq. 1, Wang2019, adapted to be ratio).
            # Equation: P_1 / P_0 = ( t_1^{-\alpha} e^{-2 * pi * f * t_1 / Q_{c,f}} ) / t_0^{-\alpha} e^{-2 * pi * f * t_0 / Q_{c,f}}
            # Therefore:
            Q_c_S = - 2. * np.pi * freq_approx * (t_1 - t_0) / np.log( (P_1 / P_0) * (( t_0 / t_1 ) ** -alpha_gs) )

            # And estimate error:
            # calculate uncertainty in linear reg.:
            m_upp, c_upp = np.sign(popt[0])*( np.abs(popt[0]) + np.sqrt(pcov[0,0]) ), np.sign(popt[1])*( np.abs(popt[1]) + np.sqrt(pcov[1,1]) )
            log_powers_out_lin_reg_upper_uncert = linear_reg_func(t_solns, m_upp, c_upp)
            m_low, c_low = np.sign(popt[0])*( np.abs(popt[0]) + np.sqrt(pcov[0,0]) ), np.sign(popt[1])*( np.abs(popt[1]) + np.sqrt(pcov[1,1]) )
            log_powers_out_lin_reg_lower_uncert = linear_reg_func(t_solns, m_upp, c_upp)
            # And calculate Qc err:
            P_0 = 10**log_powers_out_lin_reg_upper_uncert[0]
            P_1 = 10**log_powers_out_lin_reg_lower_uncert[1]
            Q_c_S_err = np.abs(- 2. * np.pi * freq_approx * (t_1 - t_0) / np.log( (P_1 / P_0) * (( t_0 / t_1 ) ** -alpha_gs) ) - Q_c_S)

            # And append result to output:
            event_obs_dict[station] = {}
            event_obs_dict[station]["Qc"] = Q_c_S
            event_obs_dict[station]["Q_c_S_err"] = Q_c_S_err

        
        # And append to overall obs. dict for all events (if solved for events individually):
        event_obs_dicts[nonlinloc_event_hyp_filename] = event_obs_dict

    return event_obs_dicts   









        
