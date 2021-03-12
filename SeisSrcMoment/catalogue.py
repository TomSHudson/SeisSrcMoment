#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to calculate moment magnitudes for an entire catalogue. This sub-module also contains functions for Gutenberg-Richter 
# distribution analysis and calculating b-values.

# Input variables:

# Output variables:

# Created by Tom Hudson, 12th October 2020

# Notes:
# Requires NonLinLocPy, which can be found on TomSHudson github and PyPi repository.

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import sys, os
import glob
import NonLinLocPy
from NonLinLocPy import read_nonlinloc
import pandas as pd
import gc
import pickle
from scipy.optimize import curve_fit
import math
import pandas as pd
import matplotlib.dates as mdates
from SeisSrcMoment import moment


# ------------------- Define generally useful functions -------------------
def find_nearest_idx(array, value):
    """Function to find nearest index of value in array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_event_moment_magnitudes(nonlinloc_hyp_files_dir, nonlinloc_hyp_files_list, mseed_dir, out_fname, window_before_after, filt_freqs, density, Vp, phase_to_process='P', MT_six_tensor=[], stations_not_to_process=[], inventory_fname=None, instruments_gain_filename=None, remove_noise_spectrum=False, vel_model_df=[], verbosity_level=0, plot_switch=False):
    """Function to find the event moment magnitudes for a specified list of NonLinLoc relocated 
    earthquakes. Outputs a python dictionary to file containing all the event magnitude catalogue 
    information for all the earthquakes. This is saved to <out_fname>.
    Notes:
    - Currently moment magnitude is calculated by fitting a Brune spectrum (see 
    SeisSrcMoment.moment.calc_moment for further details).
    - This function requires event information, currently in NonLinLoc format only, 
    as well as continuous mseed data in an archive format. This is currently fixed for the format:
    archive/years/julday/year_julday_station_*.m
    - To correctly calculate the event magnitudes, this function also requires either an instrument 
    gains file (contains passband instrument static gains in specific format) or a full inventory 
    file, in IRIS dataless format. The latter performs a full frequency-response instrument 
    corrected analysis.

    Arguments:
    Required:
    nonlinloc_hyp_files_dir - Path to the directory containing the NonLinLoc event .grid0.loc.hyp 
                                files. (str)
    nonlinloc_hyp_files_list - List of the NonLinLoc hyp filenames to process. These must be within
                                nonlinloc_hyp_files_dir. (list of strs)
    mseed_dir - Path to the overall continuous mseed archive directory <archive>, where the archive 
                is in the format: <archive>/years/julday/year_julday_station_*.m . (str)
    out_fname - The filename of the python dictionary output by this function. The data is saved 
                pickled and can be read using pickle or the inbuilt function in this module 
                read_magnitude_catalogue(). (str)
    window_before_after - The time before and after the phase pick to use for calculating the 
                            magnitude within. It is a list of the format [time before, time after].
                            Units are in seconds. Default values are not specified, but a good 
                            starting point might be [0.1, 0.6] for a 10 Hz local earthquake. 
                            (list of two floats)
    filt_freqs - The frequencies with which to filter the data prior to calculating the moment 
                magnitude. Ideally the data should be as unfiltered as possible, as the noise
                spectrum can be automatically approximately removed by specifying 
                remove_noise_spectrum = True. filt_freqs takes the format [high-pass-freq, 
                low-pass-freq]. Units are Hz. (list of two floats)
    density - The density of the medium, in kg/m3. (float)
    Vp - The phase-velocity of the medium in m/s OR str "from_depth" if using from depth. If 
            "from_depth", will use vel_model_df to determine velocity at the event depth). Note 
            that if "from_depth" is used, then vel_model_df optional argument must be specified.
            (float or specific str)

    Optional:
    phase_to_process - Must be P or S. Phase to process. If P, will use L component, if S will 
                        use T component. Default is "P" (str)
    MT_six_tensor - The moment 6-tensor defining the source radiation pattern. If specified, the 
                    moment magnitudes output are radiation-pattern-corrected.If not specified 
                    (i.e. []), default is to not take a specific moment tensor, in which case 
                    an isotropic moment tensor [1, 1, 1, 0, 0, 0] will be used. Moment tensor 
                    is of the form: [m_xx, m_yy, m_zz, m_xy, m_xz, m_yz]. (list of 6 floats)
    stations_not_to_process - A list of stations not to process. Default is []. (list of strs)
    inventory_fname - Inventory filename. Currently only a .dataless format is supported. Either 
                    this or <instruments_gain_filename> must be specified, as instrument 
                    corrections have to be applied somehow. Default is None. (str)
    instruments_gain_filename - File containing instrument gains (in an ascii space-delineated 
                                file of the format: Station instrument_gain(Z N E) 
                                digitaliser_gain(Z N E)). Either this or 
                                <inventory_fname> must be specified, as instrument corrections 
                                have to be applied somehow. Default is None. (str)
    remove_noise_spectrum - If True, removes noise spectrum from measured spectrum. Default is
                            False. (bool)
    vel_model_df - A pandas dataframe object containing the velocity model (with columns: Z, 
                    Vp, Vs). This is used if <Vp> is specified to be selected "from_depth".
                    Default is for it not to be specified, and hence not used. (pandas df)
    verbosity_level - Verbosity level (1 for moment only) (2 for major parameters) (3 for 
                    plotting of traces). Defualt = 0. (int)
    plot_switch - If True, plots out displacement spectra and Brune model fits. (bool)

    Returns:
    out_dict - A python dictionary of momnet magnitudes calculated for every event in the 
                catalogue. The format of this is:
                out_dict[<event NonLinLoc id>][<station>][etc]. (Use out_dict.keys() to
                see full format of the dict). (python dict)
    """
    # Loop over events:
    out_dict = {}
    for i in range(len(nonlinloc_hyp_files_list)):
        print("-----------------------------------------------------")
        # 1. Get nonlinloc info:
        NLLoc_event_hyp_filename = nonlinloc_hyp_files_dir+"/"+nonlinloc_hyp_files_list[i]
        print("Processing for event:", NLLoc_event_hyp_filename)
        nonlinloc_event_hyp_data = read_nonlinloc.read_hyp_file(NLLoc_event_hyp_filename)
        event_origin_time = nonlinloc_event_hyp_data.origin_time
        stations_to_calculate_moment_for = list(nonlinloc_event_hyp_data.phase_data.keys())
        
        # 2. Get stream to pass to calc_moment:
        mseed_filename = mseed_dir+"/"+str(event_origin_time.year).zfill(4)+"/"+str(event_origin_time.julday).zfill(3)+"/*.m"
        st = obspy.core.Stream()
        for fname in glob.glob(mseed_filename):
            try:
                st_tmp = moment.load_mseed(fname, filt_freqs=filt_freqs)
            except TypeError:
                continue
            for i in range(len(st_tmp)):
                try:
                    st.append(st_tmp[i])
                    del st_tmp
                    gc.collect()
                except UnboundLocalError:
                    continue      

        # 3. Get Vp from velocity model, if specified:
        if Vp == 'from_depth':
            if len(vel_model_df) > 0:
                event_depth_km_bsl = nonlinloc_event_hyp_data.max_prob_hypocenter['depth']
                idx, val = find_nearest_idx(np.array(vel_model_df['Z'])/-1000., event_depth_km_bsl)   
                if phase_to_process == 'P':
                    Vp = vel_model_df['Vp'][idx] * 1000.
                elif phase_to_process == 'S':
                    Vp = vel_model_df['Vs'][idx] * 1000.
                else:
                    print("Error. phase_to_process = ", phase_to_process, "not supported. Exiting.")
                    sys.exit()
            else:
                print("Error. If Vp=from_depth, then vel_model_df must be specified. Exiting.")
                sys.exit()

        # 4. Get seismic moment:
        seis_moment, std_err_seis_M_0, n_obs, event_obs_dict = moment.calc_moment(mseed_filename, NLLoc_event_hyp_filename, stations_to_calculate_moment_for, density, Vp, phase_to_process=phase_to_process, inventory_fname=inventory_fname, instruments_gain_filename=instruments_gain_filename, window_before_after=window_before_after, filt_freqs=filt_freqs, stations_not_to_process=stations_not_to_process, MT_six_tensor=MT_six_tensor, st=st, remove_noise_spectrum=remove_noise_spectrum, verbosity_level=verbosity_level, plot_switch=plot_switch)
        # Calculate average Mw from all M_0 measurements:
        M_ws_tmp = []
        for station_tmp in list(event_obs_dict.keys()):
            M_ws_tmp.append( 0.66 * np.log10(event_obs_dict[station_tmp]['M_0']) - 6.0 )
        M_w = np.average(np.array(M_ws_tmp))
        M_w_plus_err = np.std(np.array(M_ws_tmp))
        print("Moment magnitude and error:", M_w, "(+/-", M_w_plus_err, "for n_obs=", n_obs, ")")
        out_dict[NLLoc_event_hyp_filename] = event_obs_dict 

        # Clean up:
        del st, NLLoc_event_hyp_filename, event_origin_time, stations_to_calculate_moment_for, mseed_filename
        gc.collect()

        # 5. And save data out:
        pickle.dump(out_dict, open(out_fname, 'wb'))    
        print("-----------------------------------------------------")

    return out_dict


def get_cummulative_magnitude_dist(Mw_hist):
    """Function to get cumulative magnitude distribution."""
    cum_Mw_hist = np.zeros(len(Mw_hist))
    for i in range(len(Mw_hist)):
        cum_Mw_hist[i] = np.sum(Mw_hist[i:])
    return cum_Mw_hist


def find_Mw_b_value(Mw_arr, num_bins=200, Mc_method="BVS", sig_level=0.1, Nmin=10, Mw_err=[], plot_switch=True, plot_mag_hist_data=True, fig_out_fname='', min_max_mag_plot_lims=[]):
    """Function to find the b-value of a magnitude distribution. (Adapted from BUMPs matlab code).
    Arguments:
    Required:
    Mw_arr - Array of Mw values for the events to analyse (array of floats)
    Optional:
    num_bins - The initial number of bins to use (int)
    Mc_method - The method to use to find the magnitude of completeness. Currently only KS (Kolmogorov–Smirnov
                 test) and BVS (b-value stability method (see Roberts2015 for further details)) are supported.  (str)
    sig_level - The significance level, e.g. 0.1 would be a 10 % significance level. (float)
    Nmin - Minimum number of events in window for Mc method. (int)
    Mw_err - An array containing the estimated uncertainty in Mw_arr (array of floats)
    plot_switch - If true, plot data (Default is True) (bool)
    plot_mag_hist_data - If true, plot histogram values as well as cumulative magnitude values (Default is True) (bool)
    fig_out_fname - The filename to save the output figure to. Default is empty string, which will result in not
                    saving to file. (str)
    min_max_mag_plot_lims - The lower and upper x-axis bounds (magnitude) for the output figure, with a list of length
                            of two floats corresponding to [min. mag. limit, max. mag. limit]. Default is [], which
                            results in limits being ignored. (list of two floats)

    Returns:
    a, b - Gutenberg-Richter values.
    Mw_c - The magnitude of completeness found using the specified Mc method and significance confidence level.
    If specified, a plot of teh Gutenberg-Richter distribution for the input magnitudes array."""
    # Specify any initial parameters:
    error_count = 0
    if Mc_method=="KS":
        signum = [1.07, 1.14, 1.22, 1.36, 1.63]
        sigval = [0.2, 0.15, 0.1, 0.05, 0.01]
        try:
            sig = signum[ sigval.index(sig_level) ]
        except:
            print("Error: sig_level must be one of the following values: 0.2, 0.15, 0.1, 0.05, 0.01. Exiting.")
            sys.exit()
    if Mc_method=="BVS":
        b_value_rolling_win = np.zeros(5, dtype=float) # Array to store last 5 b-values for BVS method
        b_value_rolling_win_err = np.zeros(5, dtype=float) # Array to store last 5 b-value errors for BVS method
        a_rolling_win = np.zeros(5, dtype=float) # Array to store last 5 a values for BVS method
        Mc_rolling_win = np.zeros(5, dtype=float) # Array to store last 5 a values for BVS method

    # Get a binned histogram of the Mw data:
    bin_range=[np.min(Mw_arr), np.max(Mw_arr)+0.5] # The lower and upper magnitude ranges to bin (list of two floats)
    bin_inc = (bin_range[1] - bin_range[0])/(2. * num_bins)
    Mw_hist, Mw_bin_edges = np.histogram(Mw_arr, bins=num_bins, range=(bin_range[0]-bin_inc, bin_range[1]+bin_inc))
    Mw_bin_centres = Mw_bin_edges[1:] - (Mw_bin_edges[1] - Mw_bin_edges[0])
    # And find the cummulative magnitudes cumulative density function:
    pcdf = get_cummulative_magnitude_dist(Mw_hist)
    # And find min mag completeness based upon max. frequency of events:
    if Mc_method=="BVS":
        min_Mc_value = Mw_bin_centres[np.argmax(Mw_hist)]

    # Loop over shortening Mw windows, to find optimal mag. of completeness, Mc:
    for i in range(len(Mw_bin_centres)):
        # Get data within window:
        xc = Mw_bin_centres[i:]
        Mw_min = np.min(xc)
        Mc = Mw_arr[Mw_arr >= Mw_min]
        
        # Check that not below minimum event number criteria:
        if len(Mc) < Nmin:
            error_count = 1
            Mw_c = np.nan
            b = np.nan
            a = np.nan
            break
        
        # Find Gutenberg-Richter values (a,b) (from Aki 1965 (See Greenfield et al 2018, and 2020 for more details)):
        min_diff = np.min(xc[1:]-xc[0:-1])
        min_mag = np.min(Mc)
        mean_mag = np.average(Mc)
        b = np.log10(np.e) / ( mean_mag - (min_mag - (min_diff/2.)) ) # Note that np.log10(np.e) = 1. / np.ln(10)
        a = np.log10(len(Mc)) + b * min_mag
        #print("a, b, Mw_min", a, ",", b, ",", Mw_min)

        # And perform Kolmogorov–Smirnov test, if specified by user:
        if Mc_method=="KS":
            # Test reconstructed distribution to see if KS test conditions are met:
            # Get GR distribution, calculated from a and b found above:
            n_recon = 10.**(-b * xc)
            n_recon = n_recon/np.max(n_recon)
            # Get real data distribution (from within window):
            bin_width = xc[1] - xc[0]
            Mw_hist_new_window, Mw_bin_edges_new_window = np.histogram(Mc, bins=len(xc), range=(np.min(xc)-bin_width/2., np.max(xc)+bin_width/2.))
            Mw_bin_centres_new_window = Mw_bin_edges_new_window[1:] - (Mw_bin_edges_new_window[1] - Mw_bin_edges_new_window[0])
            pcdf_new_window = get_cummulative_magnitude_dist(Mw_hist_new_window)
            n_recon_real_data = pcdf_new_window/np.max(pcdf_new_window)
            # And test the difference between the real data and the reconstructed distribution profile:
            d = np.max(np.abs( n_recon_real_data - n_recon ))
            sig_norm = sig / np.sqrt( len(Mc) )
            print(d, sig_norm)
            if d < sig_norm:
                # And write the various variables:
                Mw_c = np.min(Mc)
                b_err = 0.
                break

        # Or perform b-value stability method (see Roberts2015), if specified by user:
        elif Mc_method=="BVS":
            # Test b-value stability to see if b-value stability criterion is met:
            # (Note: See Roberts2015 and Cao2002 for more details)
            # Get new b-value before rolling window:
            b_value_before_rolling_win = b_value_rolling_win[0].copy()
            b_value_before_rolling_win_err = b_value_rolling_win_err[0].copy()
            # Calculate new rolling window b-value and error:
            # b-values:
            b_value_rolling_win[0:-1] = b_value_rolling_win[1:]
            b_value_rolling_win[-1] = b
            # error (from Aki 1965):
            b_value_rolling_win_err[0:-1] = b_value_rolling_win_err[1:]
            b_err = ((b**2) / np.log10(np.e)) * np.std(Mc) / np.sqrt(len(Mc)) #(1./ np.log10(np.e)) * (b**2) * np.sqrt( np.sum( (Mc - np.mean(Mc))**2 ) / len(Mc) * ( len(Mc) - 1. ) )
            b_value_rolling_win_err[-1] = b_err
            # And append a and Mc history:
            a_before_rolling_win = a_rolling_win[0].copy()
            Mc_before_rolling_win = Mc_rolling_win[0].copy()
            a_rolling_win[0:-1] = a_rolling_win[1:]
            a_rolling_win[-1] = a
            Mc_rolling_win[0:-1] = Mc_rolling_win[1:]
            Mc_rolling_win[-1] = np.min(Mc)
            # Only compare b-value to forward rolling window if calculated full rolling window:
            if i > len(b_value_rolling_win):
                if np.min(Mc) > min_Mc_value:
                    #print("BVS:", (b_value_before_rolling_win - b_value_before_rolling_win_err), "<", np.average(b_value_rolling_win), "<", (b_value_before_rolling_win + b_value_before_rolling_win_err), "?" )
                    if (np.average(b_value_rolling_win)) <= (b_value_before_rolling_win + b_value_before_rolling_win_err):
                        if (np.average(b_value_rolling_win)) >= (b_value_before_rolling_win - b_value_before_rolling_win_err):
                            # And write the various variables:
                            b = b_value_before_rolling_win
                            b_err = b_value_before_rolling_win_err
                            a = a_before_rolling_win
                            Mw_c = Mc_before_rolling_win
                            break
        
        # Else throw method not recognised error:
        else:
            print("Error. Mc_method:", Mc_method, "is not recognised. Exiting.")
            sys.exit()

    # And plot, if specified:
    if plot_switch:
        plt.figure(figsize=(8,6))
        plt.scatter(Mw_bin_centres, pcdf, c='k', s=5, alpha=0.75)
        if Mc_method=="KS":
            print(len(Mw_bin_centres_new_window), len(pcdf_new_window))
            plt.scatter(Mw_bin_centres_new_window, pcdf_new_window, c='b', s=5)
        if plot_mag_hist_data:
            plt.scatter(Mw_bin_centres, Mw_hist, c='#187A5B', s=5)
        x = np.array([Mw_c, bin_range[1]])
        y = 10.**( a - b*x )
        plt.plot(x, y, label="a: "+str.format('{0:.2f}', a)+", b: "+str.format('{0:.2f}', b)+", Mc: "+str.format('{0:.2f}', Mw_c)+", Method: "+Mc_method, c='#1073A9')
        plt.vlines(Mw_c, 1.0, np.max(pcdf), ls="--", color="#801C92")
        plt.yscale('log')
        plt.xlabel('Magnitude')
        plt.ylabel('Cumulative number of events')
        plt.legend()
        if len(min_max_mag_plot_lims) > 0:
            plt.xlim(min_max_mag_plot_lims)
        # plt.xlim(-1.0, 3.5)
        if len(fig_out_fname) > 0:
            print('Saving figure to',fig_out_fname)
            plt.savefig(fig_out_fname, dpi=300, transparent=True)
        plt.show()

    return a, b, Mw_c, b_err


def weighted_avg_and_std(values, weights=None):
    """ Function to return the weighted average and standard deviation
    of a set of values.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)


def bin_Q_with_depth(Qs, depths, bin_size=5.0, weights=[]):
    """Function to bin Q with depth."""
    depth_bins = np.arange(round(np.min(depths),0), np.round(np.max(depths),0), bin_size, dtype=float)
    Q_values_store = []
    weights_store = []
    for i in range(len(depth_bins)):
        Q_values_store.append([])
        weights_store.append([])
    for i in range(len(Qs)):
        if len(depth_bins)>0:
            idx, val = find_nearest_idx(depth_bins, depths[i])
        else:
            continue
        Q_values_store[idx].append(Qs[i])
        if len(weights) == 0:
            weights_store[idx].append(None)
        else:
            weights_store[idx].append(weights[i])
    Q_av_with_depth = np.zeros(len(depth_bins))
    Q_stdev_with_depth = np.zeros(len(depth_bins))
    n_bin_depths = np.zeros(len(depth_bins))
    for i in range(len(depth_bins)):
        if len(Q_values_store[i])>0:
            Q_vals_curr = np.array(Q_values_store[i])
            Q_av_with_depth[i], Q_stdev_with_depth[i] = weighted_avg_and_std(Q_vals_curr, weights=np.array(weights_store[i]))
            n_bin_depths[i] = len(Q_vals_curr)
        else:
            Q_av_with_depth[i] = np.nan
            Q_stdev_with_depth[i] = np.nan
    return depth_bins, Q_av_with_depth, Q_stdev_with_depth, n_bin_depths


def read_magnitude_catalogue(pkl_data_fname):
    """Function to read magnitude catalogue output by get_event_moment_magnitudes() into python.
    
    Arguments:
    Required:
    pkl_data_fname - The filename of the python dict containing the magnitudes information.
                    This is an output file generated by get_event_moment_magnitudes() in 
                    this module. (str).

    Returns:
    out_dict - A python dictionary containnig the moment magnitude information output by
                get_event_moment_magnitudes(). (python dict)
    """
    # Import data:
    out_dict = pickle.load(open(pkl_data_fname, 'rb'))
    return out_dict


def plot_summary_Gutenberg_Richter(pkl_data_fname, Q_filt=1000., upper_Mw_filt=4.0, Mc_method="BVS", fig_out_fname=''):
    """Function to plot summary Gutenberg-Richter distribution of data for a given SeisSrcMoment 
    catalogue dict file output.
    
    Arguments:
    Required:
    pkl_data_fname - The filename of the python dict containing the magnitudes information.
                    This is an output file generated by get_event_moment_magnitudes() in 
                    this module. (str).
    Optional:
    Q_filt - The path-average Q to filter the catalogue by (default = 1000) (float)
    upper_Mw_filt - The upper magnitude filter to apply to the data. Any values of Mw greater 
                    than this value will be removed from the catalogue prior to analysis.
                    (default = 4.0) (float)
    Mc_method - The method to use to find the magnitude of completeness. Currently only KS 
                (Kolmogorov–Smirnov test) and BVS (b-value stability method (see Roberts2015 
                for further details)) are supported.  (str)
    fig_out_fname - The filename to save the output figure to. Default is empty string, which will 
                    result in not saving to file. (str)
    min_max_mag_plot_lims - The lower and upper x-axis bounds (magnitude) for the output figure, with 
                            a list of length of two floats corresponding to [min. mag. limit, max. mag.
                            limit]. Default is [], which results in limits being ignored. (list of two 
                            floats)

    Returns:
    Plot of the Gutenberg-Richter distribution for the specified catalogue.

    """
    # Import data:
    out_dict = read_magnitude_catalogue(pkl_data_fname)

    # Get stations included in data:
    stations_list = []
    for key in list(out_dict.keys()):
        for station_key in list(out_dict[key].keys()):
            if station_key not in stations_list:
                stations_list.append(station_key)

    # Plot Gutenberg-Richter disribution (magnitude vs. number of events):
    # Get clean magnitude catalogue:
    av_Mws = []
    all_Mws = []
    av_Mws_err = []
    for nlloc_fname in list(out_dict.keys()): 
        Mw_tmp_arr = []
        for station in list(out_dict[nlloc_fname].keys()):
            if not station == 'lit_Mw':
                if out_dict[nlloc_fname][station]['Q'] < Q_filt:
                    Mw_tmp = 0.66*np.log10(out_dict[nlloc_fname][station]['M_0']) - 6.0
                    if Mw_tmp < upper_Mw_filt:
                        Mw_tmp_arr.append(Mw_tmp)
        if len(Mw_tmp_arr)>0:
            av_Mws.append(np.mean(np.array(Mw_tmp_arr)))
            av_Mws_err.append(np.std(np.array(Mw_tmp_arr)))
            all_Mws.append(Mw_tmp_arr)
    av_Mws = np.array(av_Mws)
    av_Mws_err = np.array(av_Mws_err)
    # And get the magnitude relationship:
    a, b, Mw_c, b_err = find_Mw_b_value(av_Mws, num_bins=200, Mc_method=Mc_method, sig_level=0.1, Nmin=10, plot_switch=True) #, Mw_err=av_Mws_err)



def sort_nonlinloc_fnames_into_chrono_order(nonlinloc_fnames):
    """Function to sort list of nonlinloc fnames into chronological order.
    Arguments:
    nonlinloc_fnames - list of nonlinloc fnames (list of strs)
    Returns:
    nonlinloc_fnames_time_sorted - list of chronoloically sorted nonlinloc fnames
    """
    # Sort nonlinloc event fnames into ascending order:
    nonlinloc_fnames_time_sorted = np.empty(len(nonlinloc_fnames), dtype=str).tolist()
    event_datetimes = []
    for nonlinloc_fname in nonlinloc_fnames:
        nonlinloc_hyp_data_curr = NonLinLocPy.read_nonlinloc.read_hyp_file(nonlinloc_fname)
        event_datetimes.append(nonlinloc_hyp_data_curr.origin_time.datetime)
    event_datetimes_sorted = sorted(event_datetimes)
    for nonlinloc_fname in nonlinloc_fnames:
        nonlinloc_hyp_data_curr = NonLinLocPy.read_nonlinloc.read_hyp_file(nonlinloc_fname)
        idx = event_datetimes_sorted.index(nonlinloc_hyp_data_curr.origin_time.datetime)
        nonlinloc_fnames_time_sorted[idx] = nonlinloc_fname
    return nonlinloc_fnames_time_sorted


def calc_b_values_through_time_probabilistic(nonlinloc_fnames_time_sorted, mags_dict, min_eq_samp_size=50, max_eq_samp_size=1000, number_of_eq_samp_windows=100, Q_filt=1000, upper_Mw_filt=4.0, return_Mcs=False):
    """Function to calculate b-values through time using the probabilistic method of Roberts et al (2016).
    Deviates slightly from this method in that it first allocates windows through time one way, then when 
    it reaches the end of the time series, it reverses, so as to provide approximately unbiased sampling 
    through the entire time series. It repeats this for the number of windows, number_of_eq_samp_windows.
    Note: Must input event fnames (nonlinloc_fnames_time_sorted) in chronological order!"""
    # 1. Filter events by upper cuttof magnitude and Q_filt:
    nonlinloc_fnames_time_sorted_filtered = []
    Mws_filtered = []
    event_origin_times_filtered = []
    # Loop over events through time:
    for nonlinloc_fname in nonlinloc_fnames_time_sorted:
        # Find magnitude, to see if filter:
        # Find av M_0:
        stations_tmp = list(mags_dict[nonlinloc_fname].keys())
        M_0s_tmp = []
        for station_tmp in stations_tmp:
            if mags_dict[nonlinloc_fname][station_tmp]['Q'] < Q_filt:
                    M_0s_tmp.append(mags_dict[nonlinloc_fname][station_tmp]['M_0'])
        M_0s_tmp = np.array(M_0s_tmp)
        Mw_curr_event = np.average((2. / 3.) * np.log10(M_0s_tmp) - 6.0)
        # Filter by upper limit:
        if Mw_curr_event < upper_Mw_filt:
            # Append fname and magnitude if the event meets the filter criteria:
            nonlinloc_fnames_time_sorted_filtered.append(nonlinloc_fname)
            Mws_filtered.append(Mw_curr_event)
            # And find event origin time:
            nonlinloc_hyp_data_curr = NonLinLocPy.read_nonlinloc.read_hyp_file(nonlinloc_fname)
            event_origin_times_filtered.append(nonlinloc_hyp_data_curr.origin_time)
    Mws_filtered = np.array(Mws_filtered)
    total_filtered_event_num = len(Mws_filtered)

    # 2. Define event number windows:
    if max_eq_samp_size > total_filtered_event_num:
        max_eq_samp_size = total_filtered_event_num
    eq_samp_window_sizes = np.random.randint(int(min_eq_samp_size), high=int(max_eq_samp_size), size=int(number_of_eq_samp_windows)) 
    # And check that number of windows is divisible by two, otherwise throw a warning:
    if not number_of_eq_samp_windows % 2 == 0:
        print("Warning: number_of_eq_samp_windows not divisible by two, so output number of window results is going to be of different shape.")
    
    # 3. Calculate b-values and b-values errors for all eq sample windows:
    # (Note that now the window assignment bounces back and forth between two ends of time series,
    # to minimise sampling bias for main body of data)
    b_values_all_windows = [] 
    b_values_errs_all_windows = [] 
    b_values_associated_times_all_windows = []
    Mc_all_windows = []
    # Loop over eq sample windows:
    for i in range(len(eq_samp_window_sizes)):
        if i % 1000 == 0:
            print('Processing for eq samp window:', i, '/', len(eq_samp_window_sizes))
        
        # 3.a. Specify current window start idx:
        # (note bounce back along time series window assignment, as stated above):
        if i == 0:
            random_win_start_idx = i # Starts at beginning of time series
            prog_foward = True # To progress forward in time initially
        else:
            # Check if still stepping forward in time:
            if prog_foward == True:
                win_end_idx = random_win_start_idx + eq_samp_window_sizes[i-1] + eq_samp_window_sizes[i]
                if win_end_idx >= total_filtered_event_num:
                    prog_foward = False # Bounce back, reversing window in time
                    random_win_start_idx = total_filtered_event_num # Reset win start idx accordingly
            else:
                win_end_idx = random_win_start_idx - eq_samp_window_sizes[i] #(eq_samp_window_sizes[i-1] + eq_samp_window_sizes[i])
                if win_end_idx < 0:
                    prog_foward = True # Bounce back, reversing window in time to be forwards again
                    random_win_start_idx = 0 - eq_samp_window_sizes[i-1] # Reset win start idx accordingly (to make zero after stepping forward in time)
            # If stepping forward in time:
            if prog_foward: 
                random_win_start_idx = random_win_start_idx + eq_samp_window_sizes[i-1] # Finds next window start idx
            else:
                random_win_start_idx = random_win_start_idx - eq_samp_window_sizes[i] # Finds next window start idx, if reveresed in time
            # And if window length is same length as entire window:
            if total_filtered_event_num == eq_samp_window_sizes[i]:
                random_win_start_idx = 0
                
        # 3.b. Get magnitudes associated with the eq sample window:
        Mws_current_window = Mws_filtered[random_win_start_idx:random_win_start_idx + eq_samp_window_sizes[i]]
        
        # 3.c. And calculate b-value for current window:
        a, b, Mw_c, b_err = find_Mw_b_value(Mws_current_window, num_bins=int(len(Mws_current_window)/10.), Mc_method="BVS", sig_level=0.1, Nmin=10, plot_switch=False)
        if b >= np.nan:
            if b_err >= np.nan:
                print('Skipping current window as cannot find b-value using Mc_method BVS.')
                continue
                
        # 3.d. And calculate b-value window centre based on average window time:
        all_event_times_curr_window = event_origin_times_filtered[random_win_start_idx:random_win_start_idx + eq_samp_window_sizes[i]]
        all_time_diffs_rel_to_first = np.zeros(len(all_event_times_curr_window))
        for j in range(len(all_event_times_curr_window)):
            all_time_diffs_rel_to_first[j] = all_event_times_curr_window[j] - all_event_times_curr_window[0]
        av_event_time = all_event_times_curr_window[0] + np.average(all_time_diffs_rel_to_first)
        
        # 3.e. And append all data:
        b_values_all_windows.append(b)
        b_values_errs_all_windows.append(b_err)
        b_values_associated_times_all_windows.append(av_event_time.datetime)
        Mc_all_windows.append(Mw_c)

    # And rename data out:
    event_times = b_values_associated_times_all_windows
    b_values_through_time = b_values_all_windows
    b_values_errs_through_time = b_values_errs_all_windows    
    Mc_through_time = Mc_all_windows
    
    # 4. Sort data into ascending order:
    event_times_sorted = sorted(event_times)
    b_values_through_time_sorted = np.zeros(len(b_values_through_time))
    b_values_errs_through_time_sorted = np.zeros(len(b_values_errs_through_time))
    Mc_through_time_sorted = np.zeros(len(Mc_through_time))
    for i in range(len(event_times_sorted)):
        curr_idx = event_times.index(event_times_sorted[i])
        b_values_through_time_sorted[i] = b_values_through_time[curr_idx]
        b_values_errs_through_time_sorted[i] = b_values_errs_through_time[curr_idx]
        Mc_through_time_sorted[i] = Mc_through_time[curr_idx]
    event_times = event_times_sorted
    b_values_through_time = b_values_through_time_sorted
    b_values_errs_through_time = b_values_errs_through_time_sorted
    Mc_through_time = Mc_through_time_sorted
    
    if return_Mcs:
        return event_times, b_values_through_time, b_values_errs_through_time, Mc_through_time
    else:
        return event_times, b_values_through_time, b_values_errs_through_time


def gauss_func(x, mu, sigma):
    """Gaussian function"""
    return( (1. / (sigma * np.sqrt(2. * np.pi) ) ) * np.exp( -0.5 * ( ( (x - mu) / sigma ) ** 2 ) ) )


def calc_event_window_pdf(event_times, b_values_through_time, b_values_errs_through_time, num_samps_per_window=50, b_value_res=0.01, pdf_area_norm=False, pdf_amp_norm=False):
    """Function for calculating the window pdf.
    Note, we do not currently introduce the error ratio, R, as 
    in Roberts et al (2016).
    Note: Data must be in ascending order."""
    # 1. Remove nan values and create arrays to save data to:
    # Remove nans:
    non_nan_idxs = np.argwhere(~np.isnan(np.array(b_values_through_time))).flatten()
    event_times_nan_filt = []
    b_values_through_time_nan_filt = []
    b_values_errs_through_time_nan_filt = []
    for i in non_nan_idxs:
        event_times_nan_filt.append(event_times[i])
        b_values_through_time_nan_filt.append(b_values_through_time[i])
        b_values_errs_through_time_nan_filt.append(b_values_errs_through_time[i])
    event_times = event_times_nan_filt
    b_values_through_time = b_values_through_time_nan_filt
    b_values_errs_through_time = b_values_errs_through_time_nan_filt
    # Creat arrays to save data to:
    b_value_labels = np.arange(0.5, 3.0, b_value_res)
    num_samp_windows = int(len(event_times)/num_samps_per_window)
    time_labels = event_times[0:num_samp_windows] # Note that these times will be changed later
    b_value_time_array = np.zeros((len(b_value_labels), len(time_labels)))
    
    # 2. Convert all the b-values and errors to pdfs:
    b_values_through_time_pdfs = []
    for i in range(len(b_values_through_time)):
        # Calculate Gaussian PDF for given b-value labels and b value and b value error
        pdf_curr = gauss_func(b_value_labels, b_values_through_time[i], b_values_errs_through_time[i])
        if pdf_area_norm:
            area = np.sum( pdf_curr * b_value_res )
            pdf_curr = pdf_curr / area # normalise
        elif pdf_amp_norm:
            pdf_curr = pdf_curr / np.max(np.abs(pdf_curr))
        b_values_through_time_pdfs.append(pdf_curr)
    b_values_through_time_pdfs = np.array(b_values_through_time_pdfs)
    
    # 3. Combine pdfs in a fixed (i.e. not a moving!) window:
    for i in range(len(time_labels)):
        # Combine all PDFs within window:
        # (note that the time is centred around this window, as defined in (1.))
        window_start_idx = int(i * num_samps_per_window)
        window_end_idx = int((i * num_samps_per_window) + num_samps_per_window)        
        # Perform a normallised sum, as specified in Greenfield2018/20, section 2.4.1:
        combined_window_pdf = np.sum(b_values_through_time_pdfs[window_start_idx:window_end_idx, :], axis=0) / np.sum(b_values_through_time_pdfs[window_start_idx:window_end_idx, :])  #len(non_nan_window_idxs)
        b_value_time_array[:, i] = combined_window_pdf
        # And reset average time for window:
        all_event_times_curr_window = event_times[window_start_idx:window_end_idx]
        all_time_diffs_rel_to_first = np.zeros(len(all_event_times_curr_window))
        for j in range(len(all_event_times_curr_window)):
            all_time_diffs_rel_to_first[j] = obspy.UTCDateTime(all_event_times_curr_window[j]) - obspy.UTCDateTime(all_event_times_curr_window[0])
        av_event_time = obspy.UTCDateTime(all_event_times_curr_window[0]) + np.average(all_time_diffs_rel_to_first)
        time_labels[i] = av_event_time.datetime
    
    return time_labels, b_value_labels, b_value_time_array, b_values_through_time_pdfs


def plot_temporal_b_values(time_labels, b_value_labels, b_value_time_array, fig_fname='', max_amp=None):
    """Function to plot b-values through time."""
    # Plot b-values through time, smoothed:
    if not max_amp:
        max_amp = 1.1*np.max(b_value_time_array)
    fig, ax = plt.subplots(figsize=(8,4))
    X, Y = np.meshgrid(np.arange(len(time_labels)), b_value_labels)
    Z = b_value_time_array.copy()
    im = ax.contourf(time_labels, b_value_labels, Z, levels=np.arange(0.,max_amp,max_amp/10.))
    cax = fig.add_axes([0.775, 0.25, 0.025, 0.5])
    cb = fig.colorbar(im, cax=cax, ax=ax, orientation='vertical', label='Probability density')
    # And plot maximum likelihood b-value locations:
    max_likelihood_b_values = np.zeros(len(time_labels))
    for i in range(len(time_labels)):
        max_likelihood_b_values[i] = b_value_labels[np.argmax(b_value_time_array[:,i])]
    ax.plot(time_labels, max_likelihood_b_values, c='r', alpha=0.75)
    # And set dates formatting:
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.set_xlim([obspy.UTCDateTime(year=2009,julday=110).datetime, obspy.UTCDateTime(year=2012,julday=301).datetime])
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # And do final plot settings:
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('b-value')
    ax.set_ylim(0.5, 2.0)
    fig.suptitle('b-values through time')
    if fig_fname:
        plt.savefig(fig_fname, dpi=300)
        print('Saved figure to:', fig_fname)
    return fig


def find_noise_long_period_spectral_level_through_time(mseed_archive_dir, starttime, endtime, inventory_fname=None, daily_sample_hr=12, secs_to_sample=0.7, comp_to_use='Z', stations_not_to_use=[], out_fname=''):
    """Function to find noise long period spectral level through time.
    Note that the sampling starts at 1 minute past the chosen hour.
    Note that events still could occur within the noise window. This function does not deal with this,
    so bare it in minde when analysing the data output.
    Arguments:
    Required:
    mseed_archive_dir - The path to the mseed archive dir containing continuous mseed data for the time period.
                        Note that currently archive must be in archive/year/julday/*<component (Z,N,E,1,2,etc)>.m
                        format. (str)
    starttime - The start time of the period to process data for. (obspy UTCDateTime object)
    endtime - The end time of the period to process data for. (obspy UTCDateTime object)
    Optional:
    inventory_fname - The inventory filename (.dataless) that contains the instrument response data for the 
                        network. Default is None. (str)
    daily_sample_hr - The hour of the day to sample the data for, in UTC time. Note that the sampling actually
                        starts at 1 minute past this hour. Default is 12. (int)
    secs_to_sample - The duration of the window to calculate the noise spectrum over, in seconds. Note that 
                    secs_to_sample should be the same length as that used for the magnitudes analysis for which
                    this data is being compared to (so that the frequency bins match). Default is 0.7 s.
                    (float)
    comp_to_use - Seismic trace component to use. Selects this component using the filename in the mseed archive.
                    Therefore, only use a component label that exists in the archive. Default is Z. (str)
    stations_not_to_use - A list of stations not to process. Default is []. (list of strs)
    out_fname - The filename of the file to save the data to (as a python dict). Note that if it is not specified
                (i.e. empty string) then it will not save data out. Default is not to save data out. (str)

    Returns:
    time_labels - List of datetime objects labelling each noise level measurement. (list of python datetime objects)
    noise_LP_spectral_levels - The long-period spectral level of the noise. Units of noise_LP_spectral_levels are
                             m / Hz. (np array of floats)
    (will also return python dict to out_fname if out_fname specified).
    
    """
    # Loop over days from starttime to endtime:
    num_days = int((endtime - starttime) / (3600 * 24))
    noise_LP_spectral_levels = []
    time_labels = []
    for day in np.arange(num_days):
        # Get noise sample start time:
        noise_sample_starttime_curr = starttime + (day * 3600 * 24)
        noise_sample_starttime_curr = obspy.UTCDateTime(year=noise_sample_starttime_curr.year, month=noise_sample_starttime_curr.month, day=noise_sample_starttime_curr.day, hour=daily_sample_hr, minute=1, second=0)

        # Get stream:
        mseed_filename = os.path.join(mseed_archive_dir, str(noise_sample_starttime_curr.year).zfill(4), str(noise_sample_starttime_curr.julday).zfill(3), "*"+comp_to_use+".m")
        st = obspy.core.Stream()
        for fname in glob.glob(mseed_filename):
            try:
                st_tmp = obspy.read(fname).detrend('demean').trim(starttime=noise_sample_starttime_curr-60., endtime=noise_sample_starttime_curr+9.*60.) # Note that takes 10 minute window around data interested in (so that can remove instrument response correctly)
            except TypeError:
                continue
            if len(st_tmp) > 0:
                if st_tmp[0].stats.station in stations_not_to_use:
                    continue # Skip reading this stream if the station should not be used
            for i in range(len(st_tmp)):
                try:
                    st.append(st_tmp[i])
                    del st_tmp
                    gc.collect()
                except UnboundLocalError:
                    continue      
        # And check if obtaind data:
        if len(st) == 0:
            print('Found no data for year: ', noise_sample_starttime_curr.year, ' day: ', noise_sample_starttime_curr.julday, 'Therefore skipping day.')
            continue

        # Remove instrument response:
        st_instrument_resp_removed = moment.remove_instrument_response(st, inventory_fname=inventory_fname)

        # And trim data again to exact window to use:
        st_instrument_resp_removed.trim(starttime=noise_sample_starttime_curr, endtime=noise_sample_starttime_curr+secs_to_sample)
        
        # Calculate noise spectrum:
        Sigma_0s_curr = []
        for tr in st_instrument_resp_removed:
            Sigma_0, f_c, t_star, Sigma_0_stdev, f_c_stdev, t_star_stdev = moment.get_displacement_spectra_coeffs(tr)
            Sigma_0s_curr.append(Sigma_0)
        av_Sigma_0_curr = np.average(np.array(Sigma_0s_curr))

        # And append to data stores:
        noise_LP_spectral_levels.append(av_Sigma_0_curr)
        time_labels.append(noise_sample_starttime_curr.datetime)
    
    # And do any final processing:
    noise_LP_spectral_levels = np.array(noise_LP_spectral_levels)

    # And save data to file, if specified:
    if len(out_fname) > 0:
        LP_noise_spect_level_outdict = {}
        LP_noise_spect_level_outdict['noise_LP_spectral_levels'] = noise_LP_spectral_levels
        LP_noise_spect_level_outdict['time_labels'] = time_labels
        outfile = open(out_fname, 'wb')
        pickle.dump(LP_noise_spect_level_outdict, outfile)
        outfile.close()

    return time_labels, noise_LP_spectral_levels



        