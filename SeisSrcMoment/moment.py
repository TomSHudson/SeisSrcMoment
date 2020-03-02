#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor inversion result (e.g. from MTFIT) and mseed data and calculate moment magnitude from specified stations.
# The code currently relies on a moment tensor inversion of the event (for the radiation pattern) based on the output of MTFIT (see: https://djpugh.github.io/MTfit/)
# Script returns moment magnitude as calculated at each station, as well as combined value with uncertainty estimate.

# Input variables:

# Output variables:

# Created by Tom Hudson, 10th January 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import sys, os
import scipy.io as sio # For importing .mat MT solution data
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall # For getting nodal planes for unconstrained moment tensors
from obspy.core.event.source import farfield # For calculating MT radiation patterns
import glob
# from scipy import fft
from scipy.signal import periodogram
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth # Function to calc distance and azimuth between two coordinates (see help(gps2DistAzimuth) for more info)
from obspy import UTCDateTime as UTCDateTime
import subprocess
import gc
from NonLinLocPy import read_nonlinloc # For reading NonLinLoc data (can install via pip)
from scipy.optimize import curve_fit 
from mtspec import mtspec # For multi-taper spectral analysis


# ------------------- Define generally useful functions -------------------
def load_MT_dict_from_file(matlab_data_filename):
    data=sio.loadmat(matlab_data_filename)
    i=0
    while True:
        try:
            # Load data UID from matlab file:
            if data['Events'][0].dtype.descr[i][0] == 'UID':
                uid=data['Events'][0][0][i][0]
            if data['Events'][0].dtype.descr[i][0] == 'Probability':
                MTp=data['Events'][0][0][i][0] # stored as a n length vector, the probability
            if data['Events'][0].dtype.descr[i][0] == 'MTSpace':
                MTs=data['Events'][0][0][i] # stored as a 6 by n array (6 as 6 moment tensor components)
            i+=1
        except IndexError:
            break
    
    try:
        stations = data['Stations']
    except KeyError:
        stations = []
        
    return uid, MTp, MTs, stations
    
    
def load_mseed(mseed_filename, filt_freqs=[]):
    """Function to load mseed from file. Returns stream, detrended.
    filt_freqs - The high pass and low pass filter values to use (if specified)."""
    st = obspy.read(mseed_filename)
    st.detrend("demean")
    if len(filt_freqs)>0:
        st.filter('bandpass', freqmin=filt_freqs[0], freqmax=filt_freqs[1], corners=4)
    return st


def force_stream_sample_alignment(st):
    """Function to force alignment if samples are out by less than a sample."""
    for i in range(1, len(st)):
        if np.abs(st[i].stats.starttime - st[0].stats.starttime) <= 1./st[i].stats.sampling_rate:
            st[i].stats.starttime = st[0].stats.starttime # Shift the start time so that all traces are alligned.
            # if st[i].stats.sampling_rate == st[0].stats.sampling_rate:
            #     st[i].stats.sampling_rate = st[0].stats.sampling_rate 
    return st


def check_st_n_samp(st):
    """Function to check number of samples."""
    min_n_samp = len(st[0].data)
    for i in range(len(st)):
        if len(st[i].data) < min_n_samp:
            min_n_samp = len(st[i].data)
    for i in range(len(st)):
        if len(st[i].data) > min_n_samp:
            st[i].data = st[i].data[0:min_n_samp]
    return st


def relabel_one_and_two_channels_with_N_E(st):
    """Function to relabel channel ??1 and ??2 labels as ??N and ??E.
    Note: This is not a formal correction, just a relabelling procedure."""
    for i in range(len(st)):
        if st[i].stats.channel[-1] == '1':
            st[i].stats.channel = st[i].stats.channel[0:-1] + 'N'
        elif st[i].stats.channel[-1] == '2':
            st[i].stats.channel = st[i].stats.channel[0:-1] + 'E'
    return st


def get_inst_resp_info_from_network_dataless(inventory_fname):
    """Function to get instrument response dictionary from network .dataless file.
    Arguments:
    inventory_fname - Inventory filename (.dataless) that containing all the netowrk information. (str)
    
    Returns:
    inst_resp_dict - A dictionary containing all the instrument response information for each instrument
                     channel in the network. (dict)
    """
    inv = obspy.read_inventory(inventory_fname) 
    inst_resp_dict = {}
    for a in range(len(inv)):
        network_code = inv.networks[a].code
        inst_resp_dict[network_code] = {}
        for b in range(len(inv.networks[a])):
            station = inv.networks[a].stations[b].code
            inst_resp_dict[network_code][station] = {}
            for c in range(len(inv.networks[a].stations[b])):
                channel = inv.networks[a].stations[b].channels[c].code
                inst_resp_dict[network_code][station][channel] = {}
                paz = inv.networks[a].stations[b].channels[c].response.get_paz()
                inst_resp_dict[network_code][station][channel]['poles'] = paz.poles
                inst_resp_dict[network_code][station][channel]['zeros'] = paz.zeros
                inst_resp_dict[network_code][station][channel]['sensitivity'] = inv.networks[a].stations[b].channels[c].response.instrument_sensitivity.value
                inst_resp_dict[network_code][station][channel]['gain'] = paz.normalization_factor #1.0

    return inst_resp_dict


def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    

def numerically_integrate(x_data, y_data):
    """Function to numerically integrate (sum area under a) function, given x and y data. Method is central distances, so first and last values are not calculated."""
    int_y_data = np.zeros(len(y_data),dtype=float)
    for i in range(1,len(y_data)-1):
        int_y_data[i] = int_y_data[i-1] + np.average(np.array(y_data[i-1],y_data[i+1]))*(x_data[i+1] - x_data[i-1])/2. # Calculates half area under point beyond and before it and sums to previous value
    int_y_data[-1] = int_y_data[-2] # And set final value
    return int_y_data


def Brune_model(f, Sigma_0, f_c, t_star):
    """Brune source model. Uses t_star = travel-time/Q.
    For further details see: Brune1970, Stork2014."""
    Sigma = Sigma_0 * np.exp(-np.pi * f * t_star) / ( 1. + ((f / f_c)**2) )
    return Sigma
    

def remove_instrument_response(st, inventory_fname=None, instruments_gain_filename=None, pre_filt=None):
    """Function to remove instrument response from mseed.
    Note: Only corrects for full instrument response if inventory_fname is specified."""
    st_out = st.copy()

    # Check that not given multiple data sources (which would be ambiguous):
    if None not in (inventory_fname, instruments_gain_filename):
        print("Error: Ambiguous instrument response info. as both inventory_fname or instruments_gain_filename are provided, therefore not correcting for instrument response. Exiting.")
        sys.exit()

    # Correct for full instrument response, if network inventory file is provided by user:
    elif inventory_fname:
        inst_resp_dict = get_inst_resp_info_from_network_dataless(inventory_fname)
        tr_info_to_remove = [] # List of trs to remove (that cannot remove response for)
        for i in range(len(st_out)):
            try:
                st_out[i].simulate(paz_remove=inst_resp_dict[st[i].stats.network][st[i].stats.station][st[i].stats.channel], pre_filt=pre_filt)
            except KeyError:
                print("Station ("+st[i].stats.station+") or channel ("+st[i].stats.channel+") not in instrument inventory, therefore not correcting for this component and removing it.")
                tr_info_to_remove.append([ st[i].stats.station, st[i].stats.channel ])
                continue
        # And remove any traces that didn't process:
        for i in range(len(tr_info_to_remove)):
            tr_tmp = st_out.select(station=tr_info_to_remove[i][0], channel=tr_info_to_remove[i][1])[0]
            st_out.remove(tr_tmp)
            del tr_tmp
        
    # Else import instrument gain file and convert to dict (if specified by user):
    elif instruments_gain_filename:
        # Note: This method doesn't correct for instrument frequency response
        instruments_gain_array = np.loadtxt(instruments_gain_filename, dtype=str)
        instruments_response_dict = {}
        for i in range(len(instruments_gain_array[:,0])):
            station = instruments_gain_array[i,0]
            instruments_response_dict[station] = {}
            instruments_response_dict[station]["Z"] = {}
            instruments_response_dict[station]["N"] = {}
            instruments_response_dict[station]["E"] = {}
            instruments_response_dict[station]["Z"]["overall_gain"] = np.float(instruments_gain_array[i,1])*np.float(instruments_gain_array[i,4]) # instrument x digitiliser gain
            instruments_response_dict[station]["N"]["overall_gain"] = np.float(instruments_gain_array[i,2])*np.float(instruments_gain_array[i,5]) # instrument x digitiliser gain
            instruments_response_dict[station]["E"]["overall_gain"] = np.float(instruments_gain_array[i,3])*np.float(instruments_gain_array[i,6]) # instrument x digitiliser gain
            
        # And loop over stream, correcting for instrument gain response on each station and channel:
        for j in range(len(st_out)):
            station = st_out[j].stats.station
            component = st_out[j].stats.channel[2].upper()
            # correct for gain:
            try:
                st_out[j].data = st_out[j].data/instruments_response_dict[station][component]["overall_gain"]
            except KeyError:
                continue # Skip correcting this trace as no info to correct it for

    else:
        print("Error: No inventory_fname or instruments_gain_filename provided, therefore cannot correct for instrument response. Exiting.")
        sys.exit()
        
    return st_out


def get_P_and_S_arrival_location_from_NLLoc_hyp_file(nlloc_hyp_filename):
    """Function to get lat lon depth location from NLLoc hyp file."""
    # Get lat, lon and depth from basal icequake:
    os.system("grep 'GEOGRAPHIC' "+nlloc_hyp_filename+" > ./tmp_GEO_line.txt")
    GEO_line = np.loadtxt("./tmp_GEO_line.txt", dtype=str)
    lat_lon_depth = np.array([float(GEO_line[9]), float(GEO_line[11]), float(GEO_line[13])])
    # And remove temp files:
    os.system("rm ./tmp_GEO_line.txt")
    return lat_lon_depth
    
    
def rotate_ZNE_to_LQT_axes(st, NLLoc_event_hyp_filename, stations_not_to_process=[]):
    '''Function to return rotated mssed stream, given st (=stream to rotate) and station_coords_array (= array of station coords to obtain back azimuth from).'''
        
    # Create new stream to store rotated data to:
    st_to_rotate = st
    st_rotated = obspy.core.stream.Stream()
    statNamesProcessedList = stations_not_to_process.copy() # Array containing station names not to process further
    
    # Get information from NLLoc file:
    nonlinloc_hyp_file_data = read_nonlinloc.read_hyp_file(NLLoc_event_hyp_filename)
    
    # Loop over stations in station_coords_array and if the station exists in stream then rotate and append to stream:
    for i in np.arange(len(list(nonlinloc_hyp_file_data.phase_data.keys()))):
        # Get station name and station lat and lon:
        statName = list(nonlinloc_hyp_file_data.phase_data.keys())[i]
        
        # Check station is not in list not to use (and not repeated):
        if statName.upper() not in statNamesProcessedList:
            # Append station to list so that dont process twice:
            statNamesProcessedList.append(statName.upper())
    
            # Find station in stream (if exists, this will be longer than 0):
            st_spec_station = st_to_rotate.select(station=statName)
    
            # Check if station exists in stream (i.e. if station stream is equal to 3) and if it does then continue processing:
            if len(st_spec_station) == 3:
                # Find the back azimuth and inclinaation angle at the station from nonlinloc data:
                phases = list(nonlinloc_hyp_file_data.phase_data[statName].keys())
                station_current_azimuth_event_to_sta = nonlinloc_hyp_file_data.phase_data[statName][phases[0]]['SAzim']
                station_current_toa_event_to_sta = nonlinloc_hyp_file_data.phase_data[statName][phases[0]]['RDip']
                if station_current_azimuth_event_to_sta > 180.:
                    station_current_azimuth_sta_to_event = 180. - (360. - station_current_azimuth_event_to_sta)
                elif station_current_azimuth_event_to_sta <= 180.:
                    station_current_azimuth_sta_to_event = 360. - (180. - station_current_azimuth_event_to_sta)
                if station_current_toa_event_to_sta > 180.:
                    station_current_toa_sta_inclination = station_current_toa_event_to_sta - 180.
                elif station_current_toa_event_to_sta <= 180:
                    station_current_toa_sta_inclination = 180. - station_current_toa_event_to_sta
                
                stat_event_back_azimuth = station_current_azimuth_sta_to_event
                stat_event_inclination = station_current_toa_sta_inclination

                # Make rotated radial and transverse stream from trace N and E components:
                st_spec_station_rotated = st_spec_station.rotate('ZNE->LQT', back_azimuth=stat_event_back_azimuth, inclination=stat_event_inclination) # Back azimuth is azimuth from station to event, inclination=inclination of the ray at the sation, in degrees.
                tr_L = st_spec_station_rotated.select(component="L")[0]
                tr_Q = st_spec_station_rotated.select(component="Q")[0]
                tr_T = st_spec_station_rotated.select(component="T")[0]

                # Apend traces to stream:
                st_rotated.append(tr_L)
                st_rotated.append(tr_Q)
                st_rotated.append(tr_T)
                    
    return st_rotated
    

def get_displacement_integrated_over_time(tr, plot_switch=False):
    """Function to get integrated displacement over time, for calculating moment. Takes trace of component L as input, 
    which has had the instrument response corrected and been rotated such that the trace is alligned with the P wave 
    arrival direction."""
    
    # Take FFT to get peak frequency:
    freq, Pxx = periodogram(tr.data, fs=tr.stats.sampling_rate)
    tr_data_freq_domain = np.sqrt(Pxx)

    peak_freq_idx = np.argmax(tr_data_freq_domain)
    peak_freq = freq[peak_freq_idx]
    
    # Integrate trace to get displacement:
    x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
    y_data = tr.data # Velocity data
    tr_displacement = numerically_integrate(x_data, y_data)
    
    # Integrate displacement to get the long period spectral level (total displacement*time):
    x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
    y_data = np.abs(tr_displacement) # Absolute displacement data (Note: Absolute so that obtains area under displacement rather than just indefinite integral, to account for any negative overall displacement)
    tr_displacement_area_with_time = numerically_integrate(x_data, y_data)
    #tr_displacement_area_with_time = np.abs(tr_displacement_area_with_time) # And take the absolute values, to account for any negative overall displacement
    
    # And take FFT of displacement:
    freq_displacement, Pxx = periodogram(tr_displacement, fs=tr.stats.sampling_rate) 
    tr_displacement_data_freq_domain = np.sqrt(Pxx)

    # And get maximum displacement integrated with time (To find spectral level):
    long_period_spectral_level = np.max(tr_displacement_area_with_time)
    
    # And plot some figures if specified:
    if plot_switch:
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0,0].plot(freq[1:], tr_data_freq_domain[1:])
        axes[0,0].set_xscale("log", nonposx='clip')
        axes[0,0].set_yscale("log", nonposy='clip')
        axes[0,0].set_xlabel("Frequency (Hz)")
        axes[0,0].set_ylabel("Amplitude (m/s/Hz)")
        axes[0,0].set_title("Velocity spectrum")
        axes[0,1].plot(x_data, tr_displacement/np.max(np.abs(tr_displacement)), label="Displacement")
        axes[0,1].plot(x_data, tr.data/np.max(np.abs(tr.data)), label="Velocity")
        axes[0,1].set_xlabel("Time (s)")
        axes[0,1].set_ylabel("Amplitude (normallised)")
        axes[0,1].set_title("Comparison of velocity and displacement")
        # And take FFT of displacement:
        freq_displacement, Pxx = periodogram(tr_displacement, fs=tr.stats.sampling_rate)
        tr_displacement_data_freq_domain = np.sqrt(Pxx)
        axes[1,0].plot(freq_displacement[1:], tr_displacement_data_freq_domain[1:])
        axes[1,0].set_xscale("log", nonposx='clip')
        axes[1,0].set_yscale("log", nonposy='clip')
        axes[1,0].set_xlabel("Frequency (Hz)")
        axes[1,0].set_ylabel("Amplitude (m/Hz)")
        axes[1,0].set_title("Displacement spectrum")
        axes[1,1].plot(x_data, tr_displacement_area_with_time)
        axes[1,1].set_xlabel("Time (s)")
        axes[1,1].set_ylabel("Integrated displacement (m . s)")
        axes[1,1].set_title("Integrated dispacement")
        plt.show()
    
    return long_period_spectral_level


def get_displacement_spectra_coeffs(tr, plot_switch=False):
    """Function to get long-period spectral level, for calculating moment. Also finds the corner frequency and 
    travel-time / Q. Takes trace of component L as input, which has had the instrument response corrected and 
    been rotated such that the trace is alligned with the P wave arrival direction."""
    
    # Take FFT to get peak frequency:
    freq, Pxx = periodogram(tr.data, fs=tr.stats.sampling_rate)
    tr_data_freq_domain = np.sqrt(Pxx)
    
    # Integrate trace to get displacement:
    x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
    y_data = tr.data # Velocity data
    tr_displacement = numerically_integrate(x_data, y_data)
    
    # Get spectra of displacement (using multi-taper spectral analysis method):
    # (See: Thomson1982, Park1987, Prieto2009, Pozgay2009a for further details)
    Pxx, freq_displacement = mtspec(tr_displacement, delta=1./tr.stats.sampling_rate, time_bandwidth=2, number_of_tapers=5)
    disp_spect_amp = np.sqrt(Pxx)

    # And fit Brune model:
    param, param_cov = curve_fit(Brune_model, freq_displacement[1:], disp_spect_amp[1:], p0=[np.max(disp_spect_amp), 10., 1./250.])
    Sigma_0 = param[0]
    f_c =  param[1]
    t_star = param[2] # t_star = 
    Brune_model_spect_amp_fit = Brune_model(freq_displacement[1:], Sigma_0, f_c, t_star)
    
    # And plot some figures if specified:
    if plot_switch:
        # Print fitting parameters:
        print("Sigma_0:", Sigma_0, "f_c:", f_c, "t_star:", t_star)
        # And take FFT of displacement:
        # (Note: for comparison only)
        freq_displacement2, Pxx = periodogram(tr_displacement, fs=tr.stats.sampling_rate) 
        tr_displacement_data_freq_domain = np.sqrt(Pxx)
        # Integrate displacement to get the long period spectral level (total displacement*time):
        # (Note: Only used for plotting purposes, for comparision)
        x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
        y_data = np.abs(tr_displacement) # Absolute displacement data (Note: Absolute so that obtains area under displacement rather than just indefinite integral, to account for any negative overall displacement)
        tr_displacement_area_with_time = numerically_integrate(x_data, y_data)
        # And plot data:        
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0,0].plot(freq[1:], tr_data_freq_domain[1:])
        axes[0,0].set_xscale("log", nonposx='clip')
        axes[0,0].set_yscale("log", nonposy='clip')
        axes[0,0].set_xlabel("Frequency (Hz)")
        axes[0,0].set_ylabel("Amplitude (m/s/Hz)")
        axes[0,0].set_title("Velocity spectrum")
        axes[0,1].plot(x_data, tr_displacement/np.max(np.abs(tr_displacement)), label="Displacement")
        axes[0,1].plot(x_data, tr.data/np.max(np.abs(tr.data)), label="Velocity")
        axes[0,1].set_xlabel("Time (s)")
        axes[0,1].set_ylabel("Amplitude (normallised)")
        axes[0,1].set_title("Comparison of velocity and displacement")
        # And take FFT of displacement:
        freq_displacement, Pxx = periodogram(tr_displacement, fs=tr.stats.sampling_rate)
        tr_displacement_data_freq_domain = np.sqrt(Pxx)
        axes[1,0].plot(freq_displacement2[1:], tr_displacement_data_freq_domain[1:])
        axes[1,0].plot(freq_displacement[1:], disp_spect_amp[1:], c='r')
        axes[1,0].plot(freq_displacement[1:], Brune_model_spect_amp_fit, c='g')
        axes[1,0].set_xscale("log", nonposx='clip')
        axes[1,0].set_yscale("log", nonposy='clip')
        axes[1,0].set_xlabel("Frequency (Hz)")
        axes[1,0].set_ylabel("Amplitude (m/Hz)")
        axes[1,0].set_title("Displacement spectrum")
        axes[1,1].plot(x_data, tr_displacement_area_with_time)
        axes[1,1].set_xlabel("Time (s)")
        axes[1,1].set_ylabel("Integrated displacement (m . s)")
        axes[1,1].set_title("Integrated dispacement")
        plt.show()
    
    return Sigma_0, f_c, t_star


    
def get_theta_and_phi_for_stations_from_NLLoc_event_hyp_data(nonlinloc_hyp_file_data, station):
    """Function to get theta and phi, in terms of lower hemisphere plot, for calculating radiation pattern components in other functions. Takes in NLLoc hyp file and station name, and returns theta and phi spherical coords."""
    # Get theta and phi in terms of lower hemisphere plot (as can get radiation pattern from lower or upper)
    phases = list(nonlinloc_hyp_file_data.phase_data[station].keys())
    azi = (nonlinloc_hyp_file_data.phase_data[station][phases[0]]['SAzim']/360.)*2*np.pi - np.pi # - pi as lower hemisphere plot so azimuth reflected by 180 degrees
    toa = (nonlinloc_hyp_file_data.phase_data[station][phases[0]]['RDip']/360.)*2*np.pi
    # And get 3D coordinates for station (and find on 2D projection):
    theta = np.pi - toa # as +ve Z = down
    phi = azi
    if theta>np.pi/2.:
        theta = theta - np.pi
        phi=phi+np.pi
    return theta, phi
    
    
def get_normallised_rad_pattern_point_amplitude(theta, phi, full_MT_norm_in):
    """Function to get normallised point amplitude on radiation plot from spherical coords theta and phi (in radians),
     as well as the full normallised moment tensor solution."""
    # Get x_1,x_2,X_3 (cartesian vector components from spherical coords):
    x_1 = np.sin(theta)*np.cos(phi)
    x_2 = np.sin(theta)*np.sin(phi)
    x_3 = np.cos(theta)
    # And get A_rad_point:
    A_rad_point = (2*x_1*x_2*full_MT_norm_in[0,1]) + (2*x_1*x_3*full_MT_norm_in[0,2]) + (2*x_2*x_3*full_MT_norm_in[1,2]) + ((x_1**2)*full_MT_norm_in[0,0]) + ((x_2**2)*full_MT_norm_in[1,1]) + ((x_3**2)*full_MT_norm_in[2,2])
    return A_rad_point


def get_event_hypocentre_station_distance(NLLoc_event_hyp_filename, station):
    """Function to get event hypocentre to station distance, in km. Inputs are NLLoc hyp filename and station name. 
    Output is STRAIGHT LINE distance of event from station in km."""
    # Import nonlinloc data:
    nonlinloc_hyp_file_data = read_nonlinloc.read_hyp_file(NLLoc_event_hyp_filename)
    # Get event coords (in km grid):
    event_x = nonlinloc_hyp_file_data.max_prob_hypocenter['x']
    event_y = nonlinloc_hyp_file_data.max_prob_hypocenter['y']
    event_z = nonlinloc_hyp_file_data.max_prob_hypocenter['z']
    # Get station coords (in km grid):
    phases = list(nonlinloc_hyp_file_data.phase_data[station].keys())
    sta_x = nonlinloc_hyp_file_data.phase_data[station][phases[0]]['StaLoc']['x']
    sta_y = nonlinloc_hyp_file_data.phase_data[station][phases[0]]['StaLoc']['y']
    sta_z = nonlinloc_hyp_file_data.phase_data[station][phases[0]]['StaLoc']['z']
    # Calculate station-event distance:
    sta_event_dist_km = np.sqrt(((event_x-sta_x)**2) + ((event_y-sta_y)**2) + ((event_z-sta_z)**2))
    return sta_event_dist_km


def calc_constant_C(rho, alpha, r_m, A_rad_point, surf_inc_angle_rad=0.):
    """rho=density,alpha=Vp,r in metres, A is point radiation amplitude. Returns value of constant C.
    surf_inc_angle_rad is the surface arrival angle of incidence from the vertical, in radians (for correcting for free surface effect).
    Set surf_inc_angle_rad to np.pi/2. if station is not at free surface."""
    # Get constant
    C = 4 * np.pi * rho * (alpha**3) * r_m / A_rad_point
    # Correct for free surface effect
    C = C / 2. * np.cos(surf_inc_angle_rad)
    return C


def calc_const_freq_atten_factor(f_const_freq, Q, Vp, r_m):
    """Function to calculate attenuation factor based on constant attentuation with frequency assumption (Peters et al 2012)"""
    atten_fact = np.exp(np.pi*f_const_freq*r_m/(Vp*Q)) # Note that positive since getting back to atteniation at source from amplitude at reciever
    return atten_fact


def calc_moment(mseed_filename, NLLoc_event_hyp_filename, stations_to_calculate_moment_for, density, Vp, inventory_fname=None, instruments_gain_filename=None, window_before_after=[0.004, 0.196], Q=150, filt_freqs=[], stations_not_to_process=[], MT_data_filename=None, MT_six_tensor=[], no_MT_phase='P', surf_inc_angle_rad=0., st=None, use_full_spectral_method=True, verbosity_level=0, plot_switch=False):
    """Function to calculate seismic moment, based on input params, using the P wave arrivals, as detailed in Shearer et al 2009.
    Arguments:
    Required:
    mseed_filename - Filename of the mseed data to use (str)
    NLLoc_event_hyp_filename - Filename of the nonlinloc location file containing the phase arrival information for the event. (str)
    stations_to_calculate_moment_for - List of station names to use for calculating the moment for (list of strs)
    density - Density of the medium (float)
    Vp - The velocity of the medium at teh source (could be P or S phase velocity) (float)
    Note: Must also specify one of inventory_fname or instruments_gain_filename (see optional args for more info)
    Optional:
    inventory_fname - Filename of a .dataless file containing the network instrument info (dataless information), for correcting for instrument response (str)
    instruments_gain_filename - Filenmae of a text file containing the instrument gains, with the columns: [Station instrument_gain(Z N E) digitaliser_gain(Z N E)].
                                Not required if specified inventory_fname. Note: Will not correct for the gains if instruments_gain_filename is specified. (str)
    window_before_after - The time in seconds, for the window start before and after the phase arrival, in the format [t_before, t_after] (list of floats of len 2)
    Q - The seismic attenuation Q value for the medium (defualt = 150) (float)
    filt_freqs - A list of two values, specifying the high pass and low pass filter values to apply to the imported data [high_pass, low_pass]. Default is to not apply
    a filter (filt_freqs = []) (list of two floats)
    stations_not_to_process - List of station names not to process (list of strs)
    MT_data_filename - Filename of the moment tensor inversion output, in MTFit format. If None, will use specified MT_six_tensor instead (default = None) (str)
    MT_six_tensor - 6 moment tensor to use if MT_data_filename not specified.
    no_MT_phase - If no moment tensor information is specified, will use average radiation pattern values for the phase specified here (either P or S). 
    Average values are 0.44 for P and 0.6 for S (Stork et al 2014) (str)
    surf_inc_angle_rad - The angle of incidence (in radians) of the ray with the surface used for the free surface correction. A good approximation (default) 
    is to set to zero. If the instruments are not at the free surface, set this to pi/2. (float)
    st - Obspy stream to process. If this is specified, the parameter mseed_filename will be ignored and this stream used instead. Default is None. (obspy Stream object)
    use_full_spectral_method - If use_full_spectral_method=True, uses a full spectral method, fitting multi-tapered spectra for the long-period displacement level, 
                                corner frequency, and travel-time / Q. If this is used, Q will be set from the fitted value rather than from the user input value. Otherwise, 
                                will use a time-domain method to find area under displacement of phase arrival only. (default is True) (bool)
    verbosity_level - Verbosity level of output. 1 for moment only, 2 for major parameters, 3 for plotting of traces. (defualt = 0) (int)
    plot_switch - Switches on plotting of analysis if True. Default is False (bool)
    Returns:
    seis_M_0 - The seismic moment in Nm (float)
    """

    # Import mseed data, nonlinloc data and MT data:
    # Import nonlinloc data:
    nonlinloc_hyp_file_data = read_nonlinloc.read_hyp_file(NLLoc_event_hyp_filename)
    # Import mseed data:
    if not st:
        st = load_mseed(mseed_filename, filt_freqs)
    # Check if any components are labelled ??1 or ??2, and relabel N and E just for magnitude processing (for rotation below):
    st = relabel_one_and_two_channels_with_N_E(st)
    # Try and trim the stream to a smaller size (if using database mseed):
    origin_time = nonlinloc_hyp_file_data.origin_time
    st.trim(starttime=origin_time - 10.0, endtime=origin_time + 80.0)
    # And force allignment of traces (if out by less than 1 sample):
    st = force_stream_sample_alignment(st)
    st = check_st_n_samp(st)
    # Import MT data:
    if MT_data_filename:
        uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)
        index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
        MT_max_prob = MTs[:,index_MT_max_prob]
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
    elif len(MT_six_tensor)>0:
        full_MT_max_prob = get_full_MT_array(MT_six_tensor)
    else:
        print("Warning: Need to specify MT_six_tensor or MT_data_filename for accurate radiation pattern correction. \n Using average radiation pattern value instead.")

    # Setup some data outputs:
    event_obs_dict = {}

    # 1. Correct for instrument response:
    st_inst_resp_corrected = remove_instrument_response(st, inventory_fname, instruments_gain_filename) # Note: Only corrects for full instrument response if inventory_fname is specified

    # 2. Rotate trace into LQT:
    st_inst_resp_corrected_rotated = rotate_ZNE_to_LQT_axes(st_inst_resp_corrected, NLLoc_event_hyp_filename, stations_not_to_process)

    # Define station to process for:
    seis_M_0_all_stations = [] # For appending results for each station to
    for station in stations_to_calculate_moment_for:
        if verbosity_level>0:
            print("Processing data for station:", station)
        # Check that station available and continue if not:
        if len(st_inst_resp_corrected_rotated.select(station=station,component="L")) == 0:
            continue
        
        # 3. Get distance to station, r:
        r_km = get_event_hypocentre_station_distance(NLLoc_event_hyp_filename, station)
        r_m = r_km*1000.
        if verbosity_level>=2:
            print("r (m):", r_m)

        # 4. Get displacement and therefroe long-period spectral level:
        # Get and trim trace to approx. event only
        tr = st_inst_resp_corrected_rotated.select(station=station, component="L")[0] #, component="Z")[0] # NOTE: MUST ROTATE TRACE TO GET TOTAL P!!!
        # Trim trace:
        try:
            P_arrival_time = nonlinloc_hyp_file_data.phase_data[station]['P']['arrival_time'] #get_P_arrival_time_at_station_from_NLLoc_event_hyp_file(NLLoc_event_hyp_filename, station)
        except KeyError:
            if verbosity_level > 0:
                print("Cannot find P arrival phase information for station:",station,"therefore skipping this station.")
            continue
        tr.trim(starttime=P_arrival_time-window_before_after[0], endtime=P_arrival_time+window_before_after[1]).detrend('demean')
        if verbosity_level>=3:
            tr.plot()
        # Check if there is no data in trace after trimming:
        if len(tr.data) == 0:
            continue
        # And get spectral level from trace:
        if use_full_spectral_method:
            Sigma_0, f_c, t_star = get_displacement_spectra_coeffs(tr, plot_switch=plot_switch)
            long_period_spectral_level = Sigma_0
            approx_travel_time = r_m / Vp
            Q = np.abs(approx_travel_time / t_star)
            if verbosity_level>=2:
                print("Spectral method fitted parameters:", "Sigma_0 =", Sigma_0, ", f_c = ", f_c, ", travel-time / Q = ", t_star, "Approx. Q = ", Q)
        else:
            long_period_spectral_level = get_displacement_integrated_over_time(tr, plot_switch=plot_switch)
            if verbosity_level>=2:
                print("Long period spectral level:", long_period_spectral_level)
        
    
        # 5. Use event NonLinLoc solution to find theta and phi (for getting radiation pattern variation terms):
        theta, phi = get_theta_and_phi_for_stations_from_NLLoc_event_hyp_data(nonlinloc_hyp_file_data, station)

        # 6. Get normallised radiation pattern value for station (A(theta, phi, unit_matrix_M)):
        if MT_data_filename or len(MT_six_tensor)>0:
            A_rad_point = get_normallised_rad_pattern_point_amplitude(theta, phi, full_MT_max_prob)
        else:
            if no_MT_phase == 'P':
                A_rad_point = 0.44
            elif no_MT_phase == 'S':
                A_rad_point = 0.6
            else:
                print("Error: No moment tensor info specified and cannot recognise no_MT_phase: "+no_MT_phase+" therefore exiting.")
                sys.exit()
        if verbosity_level>=2:
            print("Amplitude value A (normallised radiation pattern value):", A_rad_point)
    
        # 7. Calculate constant, C:
        C = calc_constant_C(density,Vp,r_m,A_rad_point, surf_inc_angle_rad=surf_inc_angle_rad)
    
        # 8. Find attenuation factor:
        # Take FFT to get peak frequency:
        freq, Pxx = periodogram(tr.data, fs=tr.stats.sampling_rate)
        tr_data_freq_domain = np.sqrt(Pxx)
        f_peak_idx = np.argmax(tr_data_freq_domain)
        f_peak = freq[f_peak_idx]
        # And calculate attentuation factor:
        atten_fact = calc_const_freq_atten_factor(f_peak,Q,Vp,r_m)
        if verbosity_level>=2:
            print("Attenuation factor (based upon constant frequency):", atten_fact)
    
        # 9. Calculate sesimic moment:
        seis_M_0 = np.abs(C*long_period_spectral_level*atten_fact)
        if verbosity_level>=1:
            print("Overall seismic moment (Nm):", seis_M_0)
        seis_M_0_all_stations.append(seis_M_0)
        # and write data:
        event_obs_dict[station] = {}
        event_obs_dict[station]['M_0'] = seis_M_0
        if use_full_spectral_method:
            event_obs_dict[station]['Sigma_0'] = Sigma_0
            event_obs_dict[station]['f_c'] = f_c
            event_obs_dict[station]['t_star'] = t_star
            event_obs_dict[station]['Q'] = Q

    # 10. Get overall average seismic moment and associated uncertainty:
    seis_M_0_all_stations = np.array(seis_M_0_all_stations)
    av_seis_M_0 = np.mean(seis_M_0_all_stations)
    n_obs = len(seis_M_0_all_stations)
    std_err_seis_M_0 = np.std(seis_M_0_all_stations)/np.sqrt(n_obs)
    
    if verbosity_level>=1:
        print("Average seismic moment for event:", av_seis_M_0, "+/-", std_err_seis_M_0)

    # Clean up:
    del st_inst_resp_corrected_rotated, st_inst_resp_corrected, st
    gc.collect()

    return av_seis_M_0, std_err_seis_M_0, n_obs, event_obs_dict
    

# ------------------- Main script for running -------------------
if __name__ == "__main__":

    # # Specify variables:
    # stations_to_calculate_moment_for = ["SKR01", "SKR02", "SKR03", "SKR04", "SKR05", "SKR06", "SKR07"]
    # stations_not_to_process = ["SKG08", "SKG09", "SKG10", "SKG11", "SKG12", "SKG13", "GR01", "GR02", "GR03","GR04","BARD"]
    # mseed_filename = "data/mseed_data/20140629184210331.m"
    # inventory_fname = None
    # instruments_gain_filename = "data/instrument_gain_data.txt" # File with instrument name, instrument gains (Z,N,E) and digitaliser gains (Z,N,E)
    # NLLoc_event_hyp_filename = "data/NLLoc_data/loc.Tom__RunNLLoc000.20140629.184210.grid0.loc.hyp"
    # window_before_after = [0.004, 0.196] # The time before and after the phase pick to use for calculating the magnitude within
    # filt_freqs = []
    # # MT_data_filename = "data/MT_data/20140629184210363MT.mat"
    # MT_six_tensor = np.array([1.,1.,1.,0.,0.,0.])
    # density = 917. # Density of medium, in kg/m3
    # Vp = 3630. # P-wave velocity in m/s
    # Q = 150. # Quality factor for the medium
    # verbosity_level = 1 # Verbosity level (1 for moment only) (2 for major parameters) (3 for plotting of traces)
    # plot_switch = False

    mseed_filename = "/Users/eart0504/data/mseed/Uturuncu/XP_network/2010/131/*.m"
    inventory_fname = "/Users/eart0504/data/mseed/Uturuncu/XP_network/dataless/IRISDMC-Plutons_dataless.dataless"  # The inventory fname, pointing to the dataless file for the network
    instruments_gain_filename = None
    NLLoc_event_hyp_filename = "/Users/eart0504/nonlinloc/loc/Uturuncu_XP_run002/loc.Tom_RunNLLoc000.20100511.013132.grid0.loc.hyp"
    stations_to_calculate_moment_for = list(read_nonlinloc.read_hyp_file(NLLoc_event_hyp_filename).phase_data.keys()) # ["PLSM", "PLLO", "PLLA", "PL03"]
    stations_not_to_process = []
    window_before_after = [0.0, 0.5] # The time before and after the phase pick to use for calculating the magnitude within
    filt_freqs = [0.5, 49.0]
    MT_six_tensor = []
    density = 2000. # Density of medium, in kg/m3
    Vp = 5000. # P-wave velocity in m/s
    Q = 50. # Quality factor for the medium
    verbosity_level = 1 # Verbosity level (1 for moment only) (2 for major parameters) (3 for plotting of traces)
    plot_switch = True


    # Get seismic moment:
    seis_moment, seis_moment_std_err, n_obs = calc_moment(mseed_filename, NLLoc_event_hyp_filename, stations_to_calculate_moment_for, density, Vp, inventory_fname=inventory_fname, instruments_gain_filename=instruments_gain_filename, Q=Q, window_before_after=window_before_after, filt_freqs=filt_freqs, stations_not_to_process=stations_not_to_process, MT_six_tensor=MT_six_tensor, verbosity_level=verbosity_level, plot_switch=plot_switch)

    print("Finished")
        
