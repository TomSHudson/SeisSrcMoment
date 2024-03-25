#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor inversion result (e.g. from MTFIT) and mseed data and calculate moment magnitude, using 
# linearised regression method, from specified stations.
# The code currently relies on a moment tensor inversion of the event (for the radiation pattern) based on the output of MTFIT (see: https://djpugh.github.io/MTfit/)
# or otherwise uses average radiation pattern.
# Script returns moment magnitude as calculated at each station, as well as combined value with uncertainty estimate.

# Input variables:

# Output variables:

# Created by Tom Hudson, 26th April 2021
# Based on theory developed by Sacha Lapins.

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import sys, os
import glob
from obspy.geodetics.base import gps2dist_azimuth # Function to calc distance and azimuth between two coordinates (see help(gps2DistAzimuth) for more info)
from obspy import UTCDateTime as UTCDateTime
import subprocess
import gc
import math
import pickle
from scipy.optimize import curve_fit, lsq_linear 
from sklearn import linear_model # For solving linear path regression inversion 
# from mtspec import mtspec # For multi-taper spectral analysis
from multitaper import MTSpec # (new, suported Prieto2022 package)
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)
import SeisSrcMoment.moment as moment # Imports moment package from SeisSrcMoment directory

# ------------------- Define generally useful functions -------------------
def nCr(n,r):
    """n choose r (n C r) function."""
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))


def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def load_st_from_archive(mseed_dir, event_origin_time, win_before_s=10, win_after_s=80, filt_freqs=[]):
    """Function to load stream from mseed archive.
    Archive must be of format archive/year/julday/*.m"""
    mseed_filenames = os.path.join(mseed_dir, str(event_origin_time.year).zfill(4), str(event_origin_time.julday).zfill(3), "*.m")
    st = obspy.core.Stream()
    for fname in glob.glob(mseed_filenames):
        try:
            st_tmp = moment.load_mseed(fname, filt_freqs=filt_freqs)
            st_tmp.trim(starttime=event_origin_time-win_before_s, endtime=event_origin_time+win_after_s)
        except TypeError:
            continue
        for i in range(len(st_tmp)):
            try:
                st.append(st_tmp[i])
                del st_tmp
                gc.collect()
            except UnboundLocalError:
                continue   
    return st


def get_velocity_given_depth(vel_model_df, event_depth_km_bsl, phase_to_process):
    """Function to find seismic velocity for a particular phase, at particular depth, for a 
    given velocity model. Returns the velocity in m/s."""
    if len(vel_model_df) > 0:
        val, idx = find_nearest_idx(np.array(vel_model_df['Z'])/-1000., event_depth_km_bsl)   
        if phase_to_process == 'P':
            Vp_out = vel_model_df['Vp'][idx] * 1000.
        elif phase_to_process == 'S':
            Vp_out = vel_model_df['Vs'][idx] * 1000.
        else:
            print("Error. phase_to_process = ", phase_to_process, "not supported. Exiting.")
            sys.exit()
    else:
        print("Error. If Vp=from_depth, then vel_model_df must be specified. Exiting.")
        sys.exit()
    return Vp_out


def set_spectra_from_freq_inv_intervals(freq_inv_intervals, freq_disp, Axx_disp):
    """Function to find nearest freqeuncy and displacement amplitudes to freq_inv_intervals values.
    Returns sampled spectra distributions."""
    freq_disp_out = np.zeros(len(freq_inv_intervals))
    Axx_disp_out = np.zeros(len(freq_inv_intervals))
    for i in range(len(freq_inv_intervals)):
        f_tmp, idx = find_nearest_idx(freq_disp, freq_inv_intervals[i])
        freq_disp_out[i] = freq_disp[idx]
        Axx_disp_out[i] = Axx_disp[idx]
    return freq_disp_out, Axx_disp_out


def prep_inv_objects(all_event_inv_params, inv_option):
    """Function to prep. inversion objects (X, y) for many events."""
    # Get events and stations keys for all events:
    event_keys = list(all_event_inv_params.keys())
    n_events = len(event_keys)
    station_keys_all_events = []
    for i in range(len(event_keys)):
        event_inv_params = all_event_inv_params[event_keys[i]]
        station_keys = list(event_inv_params.keys())
        for j in range(len(station_keys)):
            if not station_keys[j] in station_keys_all_events:
                station_keys_all_events.append(station_keys[j])
    n_stations_all_events = len(station_keys_all_events)
    # Set number of overall parameters to solve for:    
    if inv_option == "method-1":
        n_params = (n_stations_all_events * n_events) # Invert for Q only
    elif inv_option == "method-2":
        n_params = (n_stations_all_events * (n_events + 1)) # Invert for kappa (average over all events / station) and Q
    elif inv_option == "method-3":
        n_params = (n_stations_all_events * ((2 * n_events) + 1)) # Invert for kappa, R and Q
    elif inv_option == "method-4":
        n_params = (n_stations_all_events * ((2 * n_events) + 1)) + 1 # +1 for beta term. Invert for kappa, R, Q and beta
    # Setup overall X and y:
    y_overall = np.empty(0)
    X_overall = np.empty((0, n_params))
    # Loop over each event:
    for event_key in event_keys:
        # Get event inv params for current event:
        event_inv_params = all_event_inv_params[event_key]
        station_keys = list(event_inv_params.keys())
        n_stations = len(station_keys)
        # Set other parameters fpr current event:
        freq_inv_intervals = event_inv_params[station_keys[0]]['freq'] # (Assumes all frequencies for all events are the same)
        n_freq_intervals = len(freq_inv_intervals)
        n_obs_eqns = int( nCr(n_stations,2) * n_freq_intervals ) # Number of source-receiver pairs
        y = np.zeros(n_obs_eqns)
        X = np.zeros((n_obs_eqns, n_params))
        # Fill X and y:
        row_count = 0
        for ii in range(n_stations):
            for jj in range(n_stations):
                if ii < jj:
                    # Do some initial checks:
                    try:
                        if len(event_inv_params[station_keys[ii]]['freq']) > n_freq_intervals:
                            event_inv_params[station_keys[ii]]['freq'] = event_inv_params[station_keys[ii]]['freq'][0:n_freq_intervals]
                            event_inv_params[station_keys[ii]]['Axx_disp'] = event_inv_params[station_keys[ii]]['Axx_disp'][0:n_freq_intervals]
                        if len(event_inv_params[station_keys[jj]]['freq']) > n_freq_intervals:
                            event_inv_params[station_keys[jj]]['freq'] = event_inv_params[station_keys[jj]]['freq'][0:n_freq_intervals]
                            event_inv_params[station_keys[jj]]['Axx_disp'] = event_inv_params[station_keys[jj]]['Axx_disp'][0:n_freq_intervals]      
                    except KeyError:     
                        print("Skipping station pair", ii, "-", jj, "as doesn't exist for current event")
                        continue
                    # Set row indices:
                    curr_start_row_idx = int(row_count * n_freq_intervals)
                    curr_end_row_idx = int((row_count * n_freq_intervals) + n_freq_intervals)
                    # Fill y for current set of frequencies:
                    if inv_option == "method-1" or inv_option == "method-2":
                        y[curr_start_row_idx:curr_end_row_idx] = (np.log(event_inv_params[station_keys[ii]]['Axx_disp']) - np.log(event_inv_params[station_keys[jj]]['Axx_disp'])) - ((np.log(event_inv_params[station_keys[jj]]['r_m']) - np.log(event_inv_params[station_keys[ii]]['r_m'])) + (np.log(event_inv_params[station_keys[ii]]['R_term']) - np.log(event_inv_params[station_keys[jj]]['R_term']))) # (Note: Applies for method-1 or method-2 as kappa terms are set to zero if not inverting for kappa)
                    elif inv_option == "method-3":
                        y[curr_start_row_idx:curr_end_row_idx] = (np.log(event_inv_params[station_keys[ii]]['Axx_disp']) - np.log(event_inv_params[station_keys[jj]]['Axx_disp'])) - (np.log(event_inv_params[station_keys[jj]]['r_m']) - np.log(event_inv_params[station_keys[ii]]['r_m']))
                    elif inv_option == "method-4":
                        y[curr_start_row_idx:curr_end_row_idx] = np.log(event_inv_params[station_keys[ii]]['Axx_disp']) - np.log(event_inv_params[station_keys[jj]]['Axx_disp'])                        
                    # Fill X for current set of frequencies:
                    f_count = 0
                    for k in range(curr_start_row_idx, curr_end_row_idx):
                        # Get current frequency:
                        f_curr = event_inv_params[station_keys[ii]]['freq'][f_count]
                        # Get current receiver overall idx:
                        first_receiver_overall_idx = station_keys_all_events.index(station_keys[ii])
                        second_receiver_overall_idx = station_keys_all_events.index(station_keys[jj])
                        # And get kappa idxs for receivers (used in methods 2,3,4 only):
                        if inv_option == "method-2" or inv_option == "method-3":
                            first_receiver_kappa_idx = (n_params - (n_stations_all_events - first_receiver_overall_idx)) - 1 # -1 due to python indexing
                            second_receiver_kappa_idx = (n_params - (n_stations_all_events - second_receiver_overall_idx)) - 1 # -1 due to python indexing
                        elif inv_option == "method-4":
                            first_receiver_kappa_idx = (n_params - (n_stations_all_events - first_receiver_overall_idx)) - 2 # -2 due to beta term
                            second_receiver_kappa_idx = (n_params - (n_stations_all_events - second_receiver_overall_idx)) - 2 # -2 due to beta term
                        # And set parameters for X for both receivers:
                        if inv_option == "method-1":
                            # Q term for first receiver:
                            X[k, first_receiver_overall_idx] = -np.pi * f_curr * event_inv_params[station_keys[ii]]['tt_s'] # Q term
                            # Q term for second receiver:
                            X[k, second_receiver_overall_idx] = np.pi * f_curr * event_inv_params[station_keys[jj]]['tt_s'] # Q term
                        elif inv_option == "method-2":
                            # Q term for first receiver:
                            X[k, first_receiver_overall_idx] = -np.pi * f_curr * event_inv_params[station_keys[ii]]['tt_s'] # Q term
                            # Q term for second receiver:
                            X[k, second_receiver_overall_idx] = np.pi * f_curr * event_inv_params[station_keys[jj]]['tt_s'] # Q term
                            # kappa term for first receiver:
                            X[k, first_receiver_kappa_idx] = -np.pi * f_curr # kappa term
                            # kapper term for second receiver:
                            X[k, second_receiver_kappa_idx] = np.pi * f_curr # kappa term
                        elif inv_option == "method-3" or inv_option == "method-4":
                            # Write Q and R terms:
                            first_receiver_start_idx = int(2 * first_receiver_overall_idx)
                            first_receiver_end_idx = int((2 * first_receiver_overall_idx) + 2)
                            second_receiver_start_idx = int(2 * second_receiver_overall_idx)
                            second_receiver_end_idx = int((2 * second_receiver_overall_idx) + 2)
                            # X values for first receiver:
                            X[k, first_receiver_start_idx:first_receiver_end_idx] = [1, -np.pi * f_curr * event_inv_params[station_keys[ii]]['tt_s']] # R, Q terms
                            # X values for second receiver:
                            X[k, second_receiver_start_idx:second_receiver_end_idx] = [-1, np.pi * f_curr * event_inv_params[station_keys[jj]]['tt_s']] # R, Q terms
                            # Write kappa terms:
                            # kappa term for first receiver:
                            X[k, first_receiver_kappa_idx] = -np.pi * f_curr # kappa term
                            # kapper term for second receiver:
                            X[k, second_receiver_kappa_idx] = np.pi * f_curr # kappa term
                            # And allocate X beta (geometrical spreading) term, if specified to invert for this:
                            if inv_option == "method-4":
                                X[k, -1] = (np.log(event_inv_params[station_keys[jj]]['r_m']) - np.log(event_inv_params[station_keys[ii]]['r_m'])) # beta term
                        f_count+=1
                    row_count += 1
        # And append X and y to X_overall and y_overall:
        y_overall = np.concatenate((y_overall, y), axis=0)
        X_overall = np.concatenate((X_overall, X), axis=0)

    return X_overall, y_overall


def run_linear_moment_inv(X, y, inv_option, all_event_inv_params, n_cpu=-1, verbosity_level=0):
    """Function to run linear inversion for single event.
    Returns event_obs_dict"""    
    # Perform inversion based on inversion option:
    if inv_option == "method-1":
        # Lasso with regularisation path:
        model = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], max_iter=10000000, n_jobs=n_cpu, selection='cyclic', positive=True)#, eps=1e-6, selection='random', tol=1e-6) # (Varies the regularisation value, alpha)
        reg = model.fit(X, y)
    elif inv_option == "method-2":
        # Lasso with regularisation path:
        model = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], max_iter=10000000, n_jobs=n_cpu, selection='cyclic', positive=True)#, eps=1e-6, selection='random', tol=1e-6) # (Varies the regularisation value, alpha)
        reg = model.fit(X, y)
    elif inv_option == "method-3":
        # Lasso with regularisation path:
        # (Note: Do not apply positive constraint as R terms can be negative)
        model = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], max_iter=10000000, n_jobs=n_cpu, selection='cyclic')#, positive=True)#, eps=1e-6, selection='random', tol=1e-6) # (Varies the regularisation value, alpha)
        reg = model.fit(X, y)
    elif inv_option == "method-4":
        # Lasso with regularisation path:
        # (Note: Do not apply positive constraint as R terms and beta term can be negative)
        model = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], max_iter=10000000, n_jobs=n_cpu, selection='cyclic')#, positive=True)#, eps=1e-6, selection='random', tol=1e-6) # (Varies the regularisation value, alpha)
        reg = model.fit(X, y)
    # Get inv raw outputs:
    event_inv_outputs = model.coef_
    # And calculate uncertainty associated with raw outputs (by calculating the covariance matrix):
    # (Note: NOT CURRENTLY USED!)
    # y_hat = model.predict(X)
    # residuals = y - y_hat
    # residual_sum_of_squares = residuals.T @ residuals # Cross-product
    # N = np.count_nonzero(X) # Count number of parameters in sparse X
    # sigma_squared_hat = residual_sum_of_squares / N
    # var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat # Covariance matrix. (var = (X.T x X) sigma^2)
    # event_inv_outputs_stderr = np.sqrt(var_beta_hat).diagonal() / np.sqrt(N)
    # del y_hat, residuals, residual_sum_of_squares, sigma_squared_hat, var_beta_hat
    # gc.collect()

    # Get inversion outputs:
    # Setup inversion output info:
    linear_moment_inv_outputs = {}
    event_keys = list(all_event_inv_params.keys())
    n_events = len(event_keys)
    station_keys_all_events = []
    for i in range(n_events):
        event_inv_params = all_event_inv_params[event_keys[i]]
        station_keys = list(event_inv_params.keys())
        for j in range(len(station_keys)):
            if not station_keys[j] in station_keys_all_events:
                station_keys_all_events.append(station_keys[j])
    n_stations_all_events = len(station_keys_all_events)
    event_count = 0
    # Loop over events, appending event outputs:
    for i in range(n_events):
        linear_moment_inv_outputs[event_keys[i]] = {}
        linear_moment_inv_outputs[event_keys[i]]["stations"] = list(all_event_inv_params[event_keys[i]].keys())
        n_stations = len(linear_moment_inv_outputs[event_keys[i]]["stations"])
        if inv_option == "method-1" or inv_option == "method-2":
            # Qs:
            with np.errstate(divide='ignore'):
                linear_moment_inv_outputs[event_keys[i]]["Qs"] = 1. / event_inv_outputs[(event_count * n_stations):(event_count * n_stations) + n_stations]
        elif inv_option == "method-3" or inv_option == "method-4":
            # Qs:
            start_idx = (event_count * n_stations * 2)
            end_idx = (event_count * n_stations * 2) + (2 * n_stations)
            with np.errstate(divide='ignore'):
                linear_moment_inv_outputs[event_keys[i]]["Qs"] = 1. / event_inv_outputs[start_idx:end_idx][0::2]
                
            # R-terms:
            linear_moment_inv_outputs[event_keys[i]]["Rs"] = np.exp(event_inv_outputs[start_idx:end_idx][1::2])
        event_count+=1
    # And get kappas, if inverted for:
    if inv_option == "method-2" or inv_option == "method-3":
        # kappas:
        linear_moment_inv_outputs["all_stations"] = station_keys_all_events
        linear_moment_inv_outputs["all_kappas"] = event_inv_outputs[-n_stations_all_events:]
    elif inv_option == "method-4":
        # kappas:
        linear_moment_inv_outputs["all_stations"] = station_keys_all_events
        linear_moment_inv_outputs["all_kappas"] = event_inv_outputs[-(n_stations_all_events+1):-1]
        # beta term:
        linear_moment_inv_outputs["beta_term"] = event_inv_outputs[-1]

    # And print any notifications:
    if verbosity_level > 0:
        n_iters = model.n_iter_
        print("Successfully inverted for event spectral params (in ", n_iters, "iter)")
    
    return linear_moment_inv_outputs

def calc_fc_from_fixed_Q_Brune(event_inv_params, Qs_curr_event, density, Vp, A_rad_point, surf_inc_angle_rad=0., verbosity_level=0):
    """Function to calculate fc by fitting Brune model to spectra with fixed Q."""
    # Calculate f_c based on curve-fit of spectra with Brune model for fixed Q:
    # Setup some data outputs for current event:
    event_obs_dict = {}

    # Fit Brune model with reduced degrees of freedom (i.e. fixed t-star):
    # Loop over stations:
    station_keys = list(event_inv_params.keys())
    for ii in range(len(station_keys)):
        # Find f_c:
        manual_fixed_Q = Qs_curr_event[ii]
        if manual_fixed_Q > 0. and manual_fixed_Q < np.inf:
            fixed_t_star = event_inv_params[station_keys[ii]]['tt_s'] / manual_fixed_Q
            bounds = ([0, event_inv_params[station_keys[ii]]['freq'][1]], [np.inf, event_inv_params[station_keys[ii]]['freq'][-1]])
            # Tweek amplitudes to improve least-squares fitting, and then reset:
            disp_spect_amp_tmp = event_inv_params[station_keys[ii]]['Axx_disp'][1:] * (10**9)
            param, param_cov = curve_fit(lambda f, Sigma_0, f_c: moment.Brune_model(f, Sigma_0, f_c, fixed_t_star), event_inv_params[station_keys[ii]]['freq'][1:], disp_spect_amp_tmp, p0=[np.max(disp_spect_amp_tmp), 10.], bounds=bounds)
            param[0] = param[0] / (10**9)
            # And if f_c inversion failed to resolve f_c:
            if param[1] == 10.:
                param[1] = 0.
            # And append manually fixed t_star:
            param = np.append(param, fixed_t_star)
        # Else set all params to zero, if f_c inversion fails:
        else:
            param = np.zeros(3)
            param_cov = np.zeros((2,2))
            fixed_t_star = 0.0
        # And get parameters explicitely:
        Sigma_0 = param[0]
        t_star = fixed_t_star
        f_c = param[1]
        Q = manual_fixed_Q
        Sigma_0_stdev = np.sqrt(param_cov[0,0])
        f_c_stdev = np.sqrt(param_cov[1,1])
        Q_stdev = 0. # Note that currently, do not calculate this value
        t_star_stdev = 0. # Note that currently, do not calculate this value

        # Calculate moment for current station:
        r_m = event_inv_params[station_keys[ii]]['r_m']
        if type(A_rad_point) is not float:
            if type(A_rad_point) is not np.float64:
                A_rad_point_curr = A_rad_point[ii]
            else:
                A_rad_point_curr = A_rad_point
        else:
            A_rad_point_curr = A_rad_point
        C = moment.calc_constant_C(density, Vp, r_m, A_rad_point_curr, surf_inc_angle_rad=surf_inc_angle_rad)
        f_peak_idx = np.argmax(event_inv_params[station_keys[ii]]['Axx_disp'][1:])
        f_peak = event_inv_params[station_keys[ii]]['freq'][1:][f_peak_idx]
        atten_fact = moment.calc_const_freq_atten_factor(f_peak,Q,Vp,r_m)
        seis_M_0 = np.abs(C*Sigma_0*atten_fact)
        # And output parameters for each value:
        event_obs_dict[station_keys[ii]] = {}
        event_obs_dict[station_keys[ii]]['M_0'] = seis_M_0
        event_obs_dict[station_keys[ii]]['Sigma_0'] = Sigma_0
        event_obs_dict[station_keys[ii]]['f_c'] = f_c
        event_obs_dict[station_keys[ii]]['t_star'] = t_star
        event_obs_dict[station_keys[ii]]['Q'] = Q
        event_obs_dict[station_keys[ii]]['Sigma_0_stdev'] = Sigma_0_stdev
        event_obs_dict[station_keys[ii]]['f_c_stdev'] = f_c_stdev
        event_obs_dict[station_keys[ii]]['t_star_stdev'] = t_star_stdev
        event_obs_dict[station_keys[ii]]['Q_stdev'] = Q_stdev

    if verbosity_level > 1:
        Qs_tmp = []
        f_cs_tmp = []
        for stat_tmp in list(event_obs_dict.keys()):
            Qs_tmp.append(event_obs_dict[stat_tmp]['Q'])
            f_cs_tmp.append(event_obs_dict[stat_tmp]['f_c'])
        print("Qs:", Qs_tmp)
        print("f_cs:", f_cs_tmp)

    return event_obs_dict


def calc_moment_via_linear_reg(nonlinloc_event_hyp_filenames, density, Vp, mseed_filenames=None, mseed_dir=None, phase_to_process='P', channel_to_process=None, inventory_fname=None, 
                                instruments_gain_filename=None, window_before_after=[0.004, 0.196], filt_freqs=[], stations_not_to_process=[], MT_data_filenames=[], MT_six_tensors=[], 
                                surf_inc_angle_rad=0., st=None, invert_for_geom_spreading=False, num_events_per_inv=1, freq_inv_intervals=None, vel_model_df=None, inv_option="method-1", 
                                verbosity_level=0, remove_noise_spectrum=False, return_spectra_data=False, outdir=None, plot_switch=False):
    """Function to calculate seismic moment using a linearised Brune spectral moment method. This method first involves using linearised spectral ratios based on a Brune 
    source model (some basis from Barthwal et al 2019), before calculating the corner frequency by fitting the data using a Brune model with fixed Q, with Q derived in the 
    first step.
    Based on concept by Sacha Lapins (ref. to follow).

    Notes:
    - Input waveform mseed data should be in velocity format for this version.
    - Must specify one of either <mseed_filenames> or <mseed_dir>.

    Arguments:
    Required:
    nonlinloc_event_hyp_filenames - List of the filenames of the nonlinloc location file containing the phase arrival information for the event. Must be in the same ordeer as 
                                the events in <mseed_filenames>. (str)
    density - Density of the medium (float)
    Vp - The velocity of the medium at the source (could be P or S phase velocity), or = from_depth if wish to use a specified velocity model (specified using the <vel_model_df>
            variable). (float or str)
    Note: Must also specify one of either inventory_fname or instruments_gain_filename (see optional args for more info)
    Optional:
    mseed_filenames - List of of the mseed data to use. Note: This data should be velocity data for this current version. If this is specified then it will overrule <mseed_dir>
                        input parameter. (list of strs)
    mseed_dir - Path to overall mseed archive to get mseed data for each event from. Archive must be in the format: overall_archive_dir/year/julday/yearjulday_STATION_COMP.m.
                If <mseed_filenames> is specified then this parameter is redundant. (str)
    phase_to_process - Phase to process (P or S). If P, will use P picks on L component. If S, will use S picks on T component only. If no moment tensor 
                        information is specified, will use average radiation pattern values for the phase specified here (either P or S). Average values are 0.44 
                        for P and 0.6 for S (Stork et al 2014) (str)
    channel_to_process - Stream channel to process. If specified, then channel selection will force this channel rather than the channel specified by <phase_to_process>.
                        Note that this is an advanced feature and should only be used for very specific tasks, as it could neccessarily conflict with assumptions made 
                        in the moment calculation. Default is None, where it won't be used. (str)
    inventory_fname - Filename of a .dataless file containing the network instrument info (dataless information), for correcting for instrument response (str)
    instruments_gain_filename - Filenmae of a text file containing the instrument gains, with the columns: [Station instrument_gain(Z N E) digitaliser_gain(Z N E)].
                                Not required if specified inventory_fname. Note: Will not correct for the gains if instruments_gain_filename is specified. (str)
    window_before_after - The time in seconds, for the window start before and after the phase arrival, in the format [t_before, t_after] (list of floats of len 2)
    filt_freqs - A list of two values, specifying the high pass and low pass filter values to apply to the imported data [high_pass, low_pass]. Default is to not apply
                    a filter (filt_freqs = []) (list of two floats)
    stations_not_to_process - List of station names not to process (list of strs)
    MT_data_filenames - List of ilenames of the moment tensor inversion output, in MTFit format. If None, will use specified MT_six_tensor instead (default = None) (list of 
                        strs)
    MT_six_tensors - List of 6 moment tensors to use if MT_data_filenames not specified. Of length of nonlinloc_event_hyp_filenames. (list of len 6 arrays)
    surf_inc_angle_rad - The angle of incidence (in radians) of the ray with the surface used for the free surface correction. A good approximation (default) 
    is to set to zero. If the instruments are not at the free surface, set this to pi/2. (float)
    st - Obspy stream to process. If this is specified, the parameter mseed_filenames will be ignored and this stream for a single event used instead. Note that still 
            need to provide nonlinloc_event_hyp_filenames, but with only one entry in the list. Default is None. (obspy Stream object)
    invert_for_geom_spreading - If True, this function also inverts for the geometrical spreading parameter, beta. Default is False. Useful if wish to check geometrical 
                                spreading for an area. For body waves, beta = 1. For standard physics understanding behaviour, reccomend not inverting for geometrical 
                                spreading (default). (bool)
    num_events_per_inv - Number of events to include in the linearised LASSO regression inversion. Only set to greater than 1 for potentially faster inversion or if 
                            inverting for geometrical spreading. Default is 1. (int)
    freq_inv_intervals - The frequency intervals to perform the inversion over. Default is None, which means that this function uses every available frequency (advised). (np 
                            array of floats)
    remove_noise_spectrum - If remove_noise_spectrum is set to True, will remove noise spectrum of window directly before trace from signal. Only available if 
                            use_full_spectral_method is set to True. (Default = False) (bool)
    vel_model_df - Dataframe containing velocity model. Only used if <Vp> = from_depth
    inv_option - The method/number of parameters to invert for. Parameters are: Q - quality-factor; R - radiation+free-surface term; kappa - near-surface attenuation term; 
                 beta - geometrical spreading term.
                 The following methods are supported:
                    method-1 - Invert for Q. Fix R, kappa, beta (performs on single event)
                    method-2 - Invert for Q, kappa. Fix R, beta
                    method-3 - Invert for Q, kappa, R. Fix beta
                    method-4 - Invert for Q, kappa, R and beta (all params).
                 Note: Solution becomes less well constrained with increasing number of free parameters.
                 Default is method-1. (specific str)
    verbosity_level - Verbosity level of output. 1 for moment only, 2 for major parameters, 3 for plotting of traces. (defualt = 0) (int)
    return_spectra_data - If True, returns displacement spectrum data to output dict. Default is False. (bool)
    outdir - Path and output directory for saving data. Default is None, and if None, does not save data out. (default = None) (str)
    plot_switch - Switches on plotting of analysis if True. Default is False (bool)
    Returns:
    seis_M_0s - The average seismic moment each event, in Nm (float)
    event_obs_dicts - The fitted Brune spectral parameters for all events (dict of floats)
    """
    # Do initial development checks:
    if num_events_per_inv != 1:
        print("Error: num_events_per_inv > 1 not yet implemented. Exiting.")
        sys.exit()
    if invert_for_geom_spreading:
        print("Warning: <invert_for_geom_spreading> not currently allowed to be specified by user. Instead controlled by <inv_option> parameter. Continuing")

    # Perform initial tests on input parameters:
    if not mseed_filenames and not mseed_dir:
        print("Error: One of <mseed_filenames> or <mseed_dir> must be specified. Exiting.")
        sys.exit()
    if mseed_filenames:
        if len(nonlinloc_event_hyp_filenames) != len(mseed_filenames):
            print("Error: length of nonlinloc_event_hyp_filenames != length of mseed_filenames. Exiting.")
            sys.exit()
    if (phase_to_process != 'P') and (phase_to_process != 'S'):
        print("Error: phase_to_process = ", phase_to_process, "not recognised. Please specify either P or S. Exiting.")
        sys.exit()
    if inv_option!="method-1" and inv_option!="method-2" and inv_option!="method-3" and inv_option!="method-4":
        print("Error: inv_option", inv_option,"not supported. Exiting.")
        sys.exit()

    # Specify arguments for inversion method option:
    invert_for_geom_spreading = False
    invert_for_R_term = False
    invert_for_kappa_term = False
    if inv_option=="method-2":
        invert_for_kappa_term = True
    elif inv_option=="method-3":
        invert_for_kappa_term = True
        invert_for_R_term = True
    elif inv_option=="method-4":
        invert_for_kappa_term = True
        invert_for_R_term = True
        invert_for_geom_spreading = True

    # Deal with if st specified rather than mseed_filenames, nonlinloc_event_hyp_filenames:
    if st:
        mseed_filenames = ['']

    # Create datastores:
    seis_M_0s = []
    event_obs_dicts = {}
    all_event_inv_params = {}
    if Vp == 'from_depth':

        Vps = []

    # Loop over events:
    initial_event_switch = True
    for i in range(len(nonlinloc_event_hyp_filenames)):
        if verbosity_level > 0:
            print("Processing for event:", nonlinloc_event_hyp_filenames[i])
        # Import mseed data, nonlinloc data:
        # Import nonlinloc data:
        nonlinloc_event_hyp_filename = nonlinloc_event_hyp_filenames[i]
        try:
            nonlinloc_hyp_file_data = read_nonlinloc.read_hyp_file(nonlinloc_event_hyp_filename)
        except ValueError:
            if verbosity_level > 0:
                print("Warning: Cannot get event info from nlloc for event:", nonlinloc_event_hyp_filename, "therefore skipping event...")
            continue
        # Import mseed data:
        if not st:
            if mseed_dir:
                st = load_st_from_archive(mseed_dir, nonlinloc_hyp_file_data.origin_time, win_before_s=10, win_after_s=80)
            else:
                if mseed_filenames:
                    st = moment.load_mseed(mseed_filenames[i], filt_freqs)

        # Check if any components are labelled ??1 or ??2, and relabel N and E just for magnitude processing (for rotation below):
        st = moment.relabel_one_and_two_channels_with_N_E(st)
        # Try and trim the stream to a smaller size (if using database mseed):
        origin_time = nonlinloc_hyp_file_data.origin_time
        st.trim(starttime=origin_time - 10.0, endtime=origin_time + 80.0)
        # And force allignment of traces (if out by less than 1 sample):
        st = moment.force_stream_sample_alignment(st)
        st = moment.check_st_n_samp(st)

        # Get seismic velocity at source depth, if <Vp> = 'from_depth':
        if Vp == 'from_depth':
            event_depth_km_bsl = nonlinloc_hyp_file_data.max_prob_hypocenter['depth']
            Vp_curr = get_velocity_given_depth(vel_model_df, event_depth_km_bsl, phase_to_process)
            Vps.append(Vp_curr)
        else:
            Vp_curr = Vp 

        # ---------------- Process data for current event: ----------------

        # 1. Correct for instrument response:
        st_inst_resp_corrected = moment.remove_instrument_response(st, inventory_fname, instruments_gain_filename) # Note: Only corrects for full instrument response if inventory_fname is specified

        # 2. Rotate trace into LQT:
        st_inst_resp_corrected_rotated = moment.rotate_ZNE_to_LQT_axes(st_inst_resp_corrected, nonlinloc_event_hyp_filename, stations_not_to_process)
        # #==========
        # print(st_inst_resp_corrected_rotated)
        # sys.exit()
        # #==========


        # 3. Get various data for each source-receiver pair (for current event):
        station_count = 0
        stations_to_calculate_moment_for = []
        for tr in st_inst_resp_corrected_rotated:
            if tr.stats.station not in stations_to_calculate_moment_for:
                stations_to_calculate_moment_for.append(tr.stats.station)
        for station in stations_to_calculate_moment_for:
            if verbosity_level>0:
                print("Processing data for station:", station)
            # Check that station available and continue if not:
            if len(st_inst_resp_corrected_rotated.select(station=station,component="L")) == 0:
                continue

            # 3.a. Get displacement for current station:
            # if remove_noise_spectrum:
            #     tr, tr_disp, tr_noise, tr_noise_disp = get_trimmed_disp_tr_and_noise(st_inst_resp_corrected_rotated, station, nonlinloc_hyp_file_data, window_before_after=window_before_after, phase_to_process=phase_to_process, remove_noise_spectrum=remove_noise_spectrum, verbosity_level=verbosity_level)
            # else:
            #     tr, tr_disp = get_trimmed_disp_tr_and_noise(st_inst_resp_corrected_rotated, station, nonlinloc_hyp_file_data, window_before_after=window_before_after, phase_to_process=phase_to_process, remove_noise_spectrum=remove_noise_spectrum, verbosity_level=verbosity_level)
            # # Check if there is no data in trace after trimming:
            # if len(tr_disp.data) == 0:
            #     continue
            # Get trace:
            if phase_to_process == 'P':
                tr = st_inst_resp_corrected_rotated.select(station=station, component="L")[0] # NOTE: MUST ROTATE TRACE TO GET TOTAL P!!!
            elif phase_to_process == 'S':
                tr = st_inst_resp_corrected_rotated.select(station=station, component="T")[0] # NOTE: MUST ROTATE TRACE TO GET TOTAL S!!!
            if channel_to_process:
                tr = st_inst_resp_corrected.select(station=station, channel=channel_to_process)[0]
            if remove_noise_spectrum:
                tr_noise = tr.copy()
            # Trim trace approximately (+/- 10 s around event) (for displacement calculation):
            try:
                phase_arrival_time = nonlinloc_hyp_file_data.phase_data[station][phase_to_process]['arrival_time']
            except KeyError:
                if verbosity_level > 0:
                    print("Cannot find P arrival phase information for station:",station,"therefore skipping this station.")
                continue
            tr.trim(starttime=phase_arrival_time-(window_before_after[0]+10.0), endtime=phase_arrival_time+(window_before_after[1]+10.0)).detrend('demean')
            if len(tr) == 0:
                if verbosity_level > 0:
                    print("Cannot find sufficient data for station:",station,"therefore skipping this station.")
                continue
            if remove_noise_spectrum:
                event_origin_time = nonlinloc_hyp_file_data.origin_time
                tr_noise.trim(starttime=event_origin_time-((window_before_after[0] + float(len(tr.data) - 1)/tr.stats.sampling_rate) + 10.0), endtime=event_origin_time-window_before_after[0]).detrend('demean')
                if len(tr_noise) > len(tr):
                    tr_noise.data = tr_noise.data[0:len(tr)]
                if not len(tr_noise.data) == len(tr.data):
                    print("Warning: Noise trace must be same length as data trace. Therefore skipping data for station: ",station,".")
                    continue
            if verbosity_level>=3:
                tr.plot()
            # Convert trace to displacement with padding either side, before final trim:
            tr_disp = moment.integrate_trace_through_time(tr, detrend=True)
            if remove_noise_spectrum:
                tr_noise_disp = moment.integrate_trace_through_time(tr_noise, detrend=True)
            # Trim trace to final event duration:
            try:
                phase_arrival_time = nonlinloc_hyp_file_data.phase_data[station][phase_to_process]['arrival_time']
            except KeyError:
                if verbosity_level > 0:
                    print("Cannot find P arrival phase information for station:",station,"therefore skipping this station.")
                continue
            tr_disp.trim(starttime=phase_arrival_time-window_before_after[0], endtime=phase_arrival_time+window_before_after[1]).detrend('demean')
            if len(tr_disp) == 0:
                if verbosity_level > 0:
                    print("Cannot find sufficient data for station:",station,"therefore skipping this station.")
                continue
            if remove_noise_spectrum:
                event_origin_time = nonlinloc_hyp_file_data.origin_time
                tr_noise_disp.trim(starttime=event_origin_time-(window_before_after[0] + float(len(tr_disp.data) - 1)/tr_disp.stats.sampling_rate ), endtime=event_origin_time-window_before_after[0]).detrend('demean')
                if len(tr_noise_disp) > len(tr_disp):
                    tr_noise_disp.data = tr_noise_disp.data[0:len(tr_disp)]
                if not len(tr_noise_disp.data) == len(tr_disp.data):
                    print("Warning: Noise trace must be same length as data trace. Therefore skipping data for station: ",station,".")
                    continue
            if verbosity_level>=3:
                tr_disp.plot()

            # 3.b. Get displacement spectrum for current station:
            # Get spectra of displacement (using multi-taper spectral analysis method):
            # (See: Thomson1982, Park1987, Prieto2009, Pozgay2009a for further details)
            # Pxx, freq_displacement = mtspec(tr_disp.data, delta=1./tr_disp.stats.sampling_rate, time_bandwidth=2, number_of_tapers=5)
            # (Now use python implementation of Prieto2022)
            mtspec_obj = MTSpec(tr_disp.data, nw=2, kspec=5, dt=1./tr_disp.stats.sampling_rate) # (nw = time-bandwidth product, kspec=number of tapers)
            freq_displacement, Pxx = mtspec_obj.rspec()
            freq_displacement, Pxx = freq_displacement.flatten(), Pxx.flatten()
            freq_displacement, Pxx = freq_displacement[::2], Pxx[::2] # (Downsample to match mtspec package output)
            del mtspec_obj
            gc.collect()
            Axx_disp = np.sqrt(Pxx)
            # Remove noise, if tr_noise specified:
            if remove_noise_spectrum:
                if len(tr_noise_disp.data) == len(tr_disp.data):
                    # And get noise spectrum:
                    # Pxx_noise, freq_displacement_noise = mtspec(tr_noise_disp.data, delta=1./tr_noise_disp.stats.sampling_rate, time_bandwidth=2, number_of_tapers=5)
                    mtspec_obj = MTSpec(tr_noise_disp.data, nw=2, kspec=5, dt=1./tr_noise_disp.stats.sampling_rate) # (nw = time-bandwidth product, kspec=number of tapers)
                    freq_displacement_noise, Pxx_noise = mtspec_obj.rspec()
                    freq_displacement_noise, Pxx_noise = freq_displacement_noise.flatten(), Pxx_noise.flatten()
                    freq_displacement_noise, Pxx_noise = freq_displacement_noise[::2], Pxx_noise[::2] # (Downsample to match mtspec package output)
                    del mtspec_obj
                    gc.collect()
                    disp_spect_amp_noise = np.sqrt(Pxx_noise)
                    # And remove noise spectrum:
                    disp_spect_amp_no_noise = disp_spect_amp - disp_spect_amp_noise
                    disp_spect_amp_no_noise[disp_spect_amp_no_noise<0.] = 0.
                    # And replace original displacement spectrum with noise corrected one:
                    disp_spect_amp_before_corr = disp_spect_amp.copy()
                    disp_spect_amp = disp_spect_amp_no_noise
                else:
                    print("Error: Noise trace must be same length as data trace. Exiting.")
                    sys.exit()
            if verbosity_level>=2:
                print("Calculated diplacement spectra")

            # 3.c. Get source-receiver distance, r:
            r_km = moment.get_event_hypocentre_station_distance(nonlinloc_event_hyp_filename, station)
            r_m = r_km*1000.
            if verbosity_level>=2:
                print("r (m):", r_m)

            # 3.d. Get source-receiver travel time:
            phase_arrival_time = nonlinloc_hyp_file_data.phase_data[station][phase_to_process]['arrival_time']
            tt_s = phase_arrival_time - nonlinloc_hyp_file_data.origin_time

            # 3.e. Get R term (radiation pattern and free surface correction effects), if using method-1 or method-2:
            if inv_option == "method-1" or inv_option == "method-2":
                # Import MT data:
                if len(MT_data_filenames)>0:
                    MT_data_filename = MT_data_filenames[i]
                elif len(MT_six_tensors)>0:
                    MT_six_tensor = MT_six_tensors[i]
                    MT_data_filename = None
                else:
                    MT_data_filename = None
                    MT_six_tensor = []
                if MT_data_filename:
                    uid, MTp, MTs, stations = moment.load_MT_dict_from_file(MT_data_filename)
                    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
                    MT_max_prob = MTs[:,index_MT_max_prob]
                    full_MT_max_prob = moment.get_full_MT_array(MT_max_prob)
                elif len(MT_six_tensor)>0:
                    full_MT_max_prob = moment.get_full_MT_array(MT_six_tensor)
                else:
                    if verbosity_level > 0:
                        print("Warning: Need to specify MT_six_tensor or MT_data_filename for accurate radiation pattern correction. \n Using average radiation pattern value instead.")
                # And get radiation pattern amplitide from moment tensor, or average radiation pattern:
                if MT_data_filename or len(MT_six_tensor)>0:
                    theta, phi = moment.get_theta_and_phi_for_stations_from_NLLoc_event_hyp_data(nonlinloc_hyp_file_data, station)
                    A_rad_point = moment.get_normallised_rad_pattern_point_amplitude(theta, phi, full_MT_max_prob)
                    A_rad_point = np.abs(A_rad_point) # (and take absolute value, as only an amplitude)
                else:
                    # Else use average radiation pattern:
                    if phase_to_process == 'P':
                        A_rad_point = 0.44
                    elif phase_to_process == 'S':
                        A_rad_point = 0.6
                # And get R-term (contribution of free surface effect and radiation pattern):
                R_term_curr_station = 2. * np.cos(surf_inc_angle_rad) * A_rad_point
            else:
                R_term_curr_station = np.inf
                if verbosity_level > 2:
                    print("Inverting for R term.")

            # 3.f. Append data for current station to data stores:
            if station_count==0:
                # Create data stores for current event:
                event_inv_params = {}
            # Resample spectra, if specified:
            if not (freq_inv_intervals is None):
                freq_displacement, Axx_disp = set_spectra_from_freq_inv_intervals(freq_inv_intervals, freq_displacement, Axx_disp)
            # And append data for current station:
            if initial_event_switch:
                len_freqs_spectra = len(freq_displacement)
                initial_event_switch = False
            else:
                if len(freq_displacement) < len_freqs_spectra:
                    print("Skipping station as spectra of length less than usual.")
                    continue
            if freq_displacement[1] > freq_displacement[-1]:
                print("Skipping station as spectra behaviour odd.")
                continue
            event_inv_params[station] = {}
            event_inv_params[station]['freq'] = freq_displacement
            event_inv_params[station]['Axx_disp'] = Axx_disp
            event_inv_params[station]['r_m'] = r_m
            event_inv_params[station]['tt_s'] = tt_s
            if inv_option == "method-1" or inv_option == "method-2":
                event_inv_params[station]['R_term'] = R_term_curr_station # Append R term, if required (method-1,-2)
            
            # And update station count:
            station_count+=1
        
        # Clean up:
        del st_inst_resp_corrected_rotated, st_inst_resp_corrected, st
        gc.collect()
        st = None # Reset st (as currently only supports single st)

        # 4. Append any data needed for multi-event analysis:
        # Check if 2 or more stations (as need >=2 stations for spectral ratios analysis),
        # and append if satisfied:
        try:
            if len(event_inv_params.keys()) > 1:
                all_event_inv_params[nonlinloc_event_hyp_filename] = event_inv_params.copy()
            else:
                print("Insufficient observations for spectral ratios, therefore skipping event", nonlinloc_event_hyp_filename)
                del event_inv_params
                continue
        except UnboundLocalError:
            # (if event_inv_params not created due to insufficient obs)
            print("Insufficient observations for spectral ratios, therefore skipping event", nonlinloc_event_hyp_filename)
            continue

        # 5. And prep. inversion structures and run inversion for single event (if method-1):
        if inv_option == "method-1":
            # Prep. for inversion:
            event_inv_params_tmp = {}
            event_inv_params_tmp[nonlinloc_event_hyp_filename] = event_inv_params.copy()
            try:
                X, y = prep_inv_objects(event_inv_params_tmp, inv_option)
            except ValueError:
                print("Error: prep_inv_objects() failed due to issue (unresolved)... Skipping event.")
                continue
            # Run linear inversion to find Q:
            # event_obs_dict = run_linear_inv_single_event(X, y, event_inv_params, density, Vp, A_rad_point, surf_inc_angle_rad=surf_inc_angle_rad, verbosity_level=verbosity_level)
            try:
                linear_moment_inv_outputs = run_linear_moment_inv(X, y, inv_option, event_inv_params_tmp, n_cpu=4, verbosity_level=verbosity_level)
            except ValueError:
                print("Error: run_linear_moment_inv() failed due to X or y containing NaN or inf values (unresolved)... Skipping event.")
                continue
            # And find fc from fixed Q:
            Qs_curr_event = linear_moment_inv_outputs[nonlinloc_event_hyp_filename]["Qs"]
            try:
                event_obs_dict = calc_fc_from_fixed_Q_Brune(event_inv_params, Qs_curr_event, density, Vp_curr, A_rad_point, surf_inc_angle_rad=0., verbosity_level=0)
            except RuntimeError:
                print("Warning: RuntimeError, where fitting Brune model did not converge. Skipping event.")
                continue
            # And append to overall obs. dict for all events (if solved for events individually):
            event_obs_dicts[nonlinloc_event_hyp_filename] = event_obs_dict
        
        # ---------------- End: Process data for current event ----------------
    
    # And perform inversion for all events together (if inv_option = method-2, -3, -4):
    if inv_option == "method-2" or inv_option == "method-3" or inv_option == "method-4":
        # Prep. inversion objects:
        X, y = prep_inv_objects(all_event_inv_params, inv_option)
        # Perform inversion to find Q and other params:
        linear_moment_inv_outputs = run_linear_moment_inv(X, y, inv_option, all_event_inv_params, n_cpu=4, verbosity_level=verbosity_level)

        # 2. Calculate f_c based on curve-fit of spectra with Brune model for fixed Q, found above, for each event:
        count_tmp = 0
        for event_key in list(linear_moment_inv_outputs.keys()):
            if not event_key=="all_stations":
                if not event_key=="all_kappas":
                    if not event_key=="beta_term":
                        event_inv_params = all_event_inv_params[event_key]
                        Qs_curr_event = linear_moment_inv_outputs[event_key]["Qs"]
                        if inv_option == "method-3" or inv_option == "method-4":
                            A_rad_point = linear_moment_inv_outputs[event_key]["Rs"] / ( 2. * np.cos(surf_inc_angle_rad) )
                        event_obs_dict = calc_fc_from_fixed_Q_Brune(event_inv_params, Qs_curr_event, density, Vps[count_tmp], A_rad_point, surf_inc_angle_rad=0., verbosity_level=0)
                        # And append data for event, for outputing:
                        event_obs_dicts[event_key] = event_obs_dict
                        count_tmp += 1

    # And save and return data:
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    if inv_option == "method-2" or inv_option == "method-3" or inv_option == "method-4":
        if return_spectra_data:
            if outdir:
                pickle.dump(event_obs_dicts, open(os.path.join(outdir, "event_obs_dicts.pkl"), 'wb'))    
                pickle.dump(linear_moment_inv_outputs, open(os.path.join(outdir, "linear_moment_inv_outputs.pkl"), 'wb'))   
                pickle.dump(event_inv_params, open(os.path.join(outdir, "event_inv_params.pkl"), 'wb'))   
            return event_obs_dicts, linear_moment_inv_outputs, event_inv_params
        else:
            if outdir:
                pickle.dump(event_obs_dicts, open(os.path.join(outdir, "event_obs_dicts.pkl"), 'wb'))    
                pickle.dump(linear_moment_inv_outputs, open(os.path.join(outdir, "linear_moment_inv_outputs.pkl"), 'wb'))   
            return event_obs_dicts, linear_moment_inv_outputs
    else:
        if return_spectra_data:
            if outdir:
                pickle.dump(event_obs_dicts, open(os.path.join(outdir, "event_obs_dicts.pkl"), 'wb'))    
                pickle.dump(event_inv_params, open(os.path.join(outdir, "event_inv_params.pkl"), 'wb'))   
            return event_obs_dicts, None, event_inv_params
        else:
            if outdir:
                pickle.dump(event_obs_dicts, open(os.path.join(outdir, "event_obs_dicts.pkl"), 'wb'))    
            return event_obs_dicts, None

    

    

        
