#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor inversion result (from MTFIT) and mseed data and calculate moment magnitude from specified stations.
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
from scipy import fft
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth # Function to calc distance and azimuth between two coordinates (see help(gps2DistAzimuth) for more info)
from obspy import UTCDateTime as UTCDateTime
import subprocess

# Specify variables:
stations_to_calculate_moment_for = ["SKR01", "SKR02", "SKR03", "SKR04", "SKR05", "SKR06", "SKR07"]
stations_not_to_process = ["SKG08", "SKG09", "SKG10", "SKG11", "SKG12", "SKG13", "GR01", "GR02", "GR03","GR04","BARD"]
mseed_filename = "mseed_data/mseed_data_multi_events/20140619135945000.m"#"mseed_data/x.m"
instruments_gain_filename = "instrument_gain_data.txt" # File with instrument name, instrument gains (Z,N,E) and digitaliser gains (Z,N,E)
#station_coords_filename = 'input_data/ALLSTA_with_BARW.ssim' # Filename of array containing station name, latitude, longitude and altitude (altitude in km)
NLLoc_event_hyp_filename = "NLLoc_data/NLLoc_data_multi_events/loc.Tom__RunNLLoc000.20140619.135945.grid0.loc.hyp" #"NLLoc_data/loc.Tom__RunNLLoc000.20140629.184210.grid0.loc.hyp"
MT_data_filename = "MT_data/MT_data_multi_events/20140619135945526MT.mat" #"MT_data/20140629184210363MT.mat"
density = 917. # Density of medium, in kg/m3
Vp = 3630. # P-wave velocity in m/s
Q = 150. # Quality factor for the medium
verbosity_level = 1 # Verbosity level (1 for moment only) (2 for major parameters) (3 for plotting of traces)


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
    
    
def load_mseed(mseed_filename):
    """Function to load mseed from file. Returns stream, detrended."""
    st = obspy.read(mseed_filename)
    st.detrend("demean")
    return st


def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    

def time_to_freq_domain_unnormallised(tr_data, sampling_rate):
    """Function to convert a trace from the time domain to the frequency domain."""
    # Find data in freq. domain:
    lengthTrace = len(tr_data) # Length of data
    k = np.arange(lengthTrace, dtype=float)
    T = lengthTrace/sampling_rate # Time duration of data
    freqTemp = k/T # Positive and negative frequency range
    freq = freqTemp[range(lengthTrace/2)] # Positive frequency only 
    spectrumTraceTemp = fft(tr_data)/lengthTrace # Perform FFT on trace to find frequency component
    spectrumTrace = spectrumTraceTemp[range(lengthTrace/2)] # Only take positive half of spectrum
    spectrumTraceAbs = abs(spectrumTrace) # To give absolute spectrum for plotting
    tr_data_freq_domain = spectrumTraceAbs
    return tr_data_freq_domain, freq
    

def numerically_integrate(x_data, y_data):
    """Function to numerically integrate (sum area under a) function, given x and y data. Method is central distances, so first and last values are not calculated."""
    int_y_data = np.zeros(len(y_data),dtype=float)
    for i in range(1,len(y_data)-1):
        int_y_data[i] = int_y_data[i-1] + np.average(np.array(y_data[i-1],y_data[i+1]))*(x_data[i+1] - x_data[i-1])/2. # Calculates half area under point beyond and before it and sums to previous value
    int_y_data[-1] = int_y_data[-2] # And set final value
    return int_y_data
    

def remove_instrument_response(st, instruments_gain_filename, instruments_poles_zeros_filename):
    """Function to remove instrument response from mseed."""
    st_out = st.copy()
    
    # Import instrument gain file and convert to dict:
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
        
    # And loop over stream, correcting for gain and instrument response on each station and channel:
    for j in range(len(st_out)):
        station = st_out[j].stats.station
        component = st_out[j].stats.channel[2].upper()
        # correct for gain:
        try:
            st_out[j].data = st_out[j].data/instruments_response_dict[station][component]["overall_gain"]
        except KeyError:
            continue # Skip correcting this trace as no info to correct it for
        # Correct for instrument response
        # Need to do this!!! ....................
        
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
    

def get_arrival_time_data_from_NLLoc_hyp_files(nlloc_hyp_filename):
    """Function to get phase data from NLLoc file."""
    
    # Get event origin times:
    # Get event time from NLLoc file for basal icequake:
    os.system("grep 'GEOGRAPHIC' "+nlloc_hyp_filename+" > ./tmp_event_GEO_line.txt")
    GEO_line = np.loadtxt("./tmp_event_GEO_line.txt", dtype=str)
    event_origin_time = UTCDateTime(GEO_line[2]+GEO_line[3]+GEO_line[4]+GEO_line[5]+GEO_line[6]+GEO_line[7])
    # And remove temp files:
    os.system("rm ./tmp_*GEO_line.txt")
    
    # And get station arrival times and azimuth+takeoff angles for each phase, for event:
    os.system("awk '/PHASE ID/{f=1;next} /END_PHASE/{f=0} f' "+nlloc_hyp_filename+" > ./tmp_event_PHASE_lines.txt") # Get phase info and write to tmp file
    PHASE_lines = np.loadtxt("./tmp_event_PHASE_lines.txt", dtype=str) # And import phase lines as np str array
    arrival_times_dict = {} # Create empty dictionary to store data (with keys: event_origin_time, station_arrivals {station {station_P_arrival, station_S_arrival}}})
    arrival_times_dict['event_origin_time'] = event_origin_time
    arrival_times_dict['station_arrival_times'] = {}
    arrival_times_dict['azi_takeoff_angles'] = {}
    # Loop over stations:
    for i in range(len(PHASE_lines[:,0])):
        station = PHASE_lines[i, 0]
        station_current_phase_arrival = UTCDateTime(PHASE_lines[i,6]+PHASE_lines[i,7]+PHASE_lines[i,8])
        station_current_azimuth_event_to_sta = float(PHASE_lines[i,22])
        if station_current_azimuth_event_to_sta > 180.:
            station_current_azimuth_sta_to_event = 180. - (360. - station_current_azimuth_event_to_sta)
        elif station_current_azimuth_event_to_sta <= 180.:
            station_current_azimuth_sta_to_event = 360. - (180. - station_current_azimuth_event_to_sta)
        station_current_toa_event_to_sta = float(PHASE_lines[i,24])
        station_current_toa_sta_inclination = 180. - station_current_toa_event_to_sta
        # See if station entry exists, and if does, write arrival to array, otherwise, create entry and write data to file:
        # For station arrival times:
        try:
            arrival_times_dict['station_arrival_times'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And for azimuth and takeoff angle:
        try:
            arrival_times_dict['azi_takeoff_angles'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station] = {}
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_sta_to_event
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_sta_to_event
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination
                
                
    # And clean up:
    os.system("rm ./tmp*PHASE_lines.txt")
            
    return arrival_times_dict

    
def rotate_ZNE_to_LQT_axes(st, NLLoc_event_hyp_filename, stations_not_to_process=[]):
    '''Function to return rotated mssed stream, given st (=stream to rotate) and station_coords_array (= array of station coords to obtain back azimuth from).'''
        
    # Create new stream to store rotated data to:
    st_to_rotate = st.copy()
    st_rotated = obspy.core.stream.Stream()
    statNamesProcessedList=stations_not_to_process # Array containing station names not to process further
    
    # Get information from NLLoc file:
    arrival_times_dict = get_arrival_time_data_from_NLLoc_hyp_files(NLLoc_event_hyp_filename)
    
    # Loop over stations in station_coords_array and if the station exists in stream then rotate and append to stream:
    for i in np.arange(len(arrival_times_dict['station_arrival_times'].keys())):
        # Get station name and station lat and lon:
        statName = arrival_times_dict['station_arrival_times'].keys()[i]
        
        # Check station is not in list not to use (and not repeated):
        if statName.upper() not in statNamesProcessedList:
            # Append station to list so that dont process twice:
            statNamesProcessedList.append(statName.upper())
    
            # Find station in stream (if exists, this will be longer than 0):
            st_spec_station = st_to_rotate.select(station=statName)
    
            # Check if station exists in stream (i.e. if station stream is equal to 3) and if it does then continue processing:
            if len(st_spec_station) == 3:
                # Find back azimuth (angle from station to event) from event location (from NLLoc or CMM):
                ##stat_event_back_azimuth = gps2dist_azimuth(statLat,statLon,eventLat,eventLon)[1] # Returns 2nd value of tuple (distance, azimuth A->B, azimuth B->A)
                stat_event_back_azimuth = arrival_times_dict['azi_takeoff_angles'][statName]["P_azimuth_sta_to_event"]
                stat_event_inclination = arrival_times_dict['azi_takeoff_angles'][statName]["P_toa_sta_inclination"]
                # Make rotated radial and transverse stream from trace N and E components:
                st_spec_station_rotated = st_spec_station.rotate('ZNE->LQT', back_azimuth=stat_event_back_azimuth, inclination=stat_event_inclination) # Back azimuth is azimuth from station to event, inclination=None as not used for 'NE->RT' method
                tr_L = st_spec_station_rotated.select(component="L")[0]
                tr_Q = st_spec_station_rotated.select(component="Q")[0]
                tr_T = st_spec_station_rotated.select(component="T")[0]
                # Apend traces to stream:
                st_rotated.append(tr_L)
                st_rotated.append(tr_Q)
                st_rotated.append(tr_T)
                    
    return st_rotated
    

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


def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    

def get_displacement_integrated_over_time(tr, plot_switch=False):
    """Function to get integrated displacement over time, for calculating moment. Takes trace of component L as input, which has had the instrument response corrected and been rotated such that the trace is alligned with the P wave arrival direction."""
    
    # Take FFT to get peak frequency:
    tr_data_freq_domain, freq = time_to_freq_domain_unnormallised(tr.data, tr.stats.sampling_rate)
    peak_freq_idx = np.argmax(tr_data_freq_domain)
    peak_freq = freq[peak_freq_idx]
    
    # Integrate trace to get displacement:
    x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
    y_data = tr.data # Velocity data
    tr_displacement = numerically_integrate(x_data, y_data)
    
    # Integrate displacement to get the long period spectral level (total displacement*time):
    x_data = np.arange(0.0,len(tr.data)/tr.stats.sampling_rate,1./tr.stats.sampling_rate) # Get time data
    y_data = tr_displacement # Displacement data
    tr_displacement_area_with_time = numerically_integrate(x_data, y_data)
    
    # And take FFT of displacement:
    tr_displacement_data_freq_domain, freq_displacement = time_to_freq_domain_unnormallised(tr_displacement, tr.stats.sampling_rate)
    
    # And get maximum displacement integrated with time (To find spectral level):
    long_period_spectral_level = np.max(tr_displacement_area_with_time)
    
    # And plot some figures if specified:
    if plot_switch:
        plt.figure()
        plt.plot(freq, tr_data_freq_domain)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (unnormallised)")
        plt.show()
        plt.figure()
        plt.plot(x_data, tr_displacement/np.max(tr_displacement), label="Displacement")
        plt.plot(x_data, tr.data/np.max(tr.data), label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normallised)")
        plt.title("Comparison of velocity and displacement")
        plt.show()
        plt.figure()
        plt.plot(freq_displacement, tr_displacement_data_freq_domain)
        plt.xlabel("Frequnecy (Hz)")
        plt.ylabel("Amplitude (unnormallised)")
        plt.show()
        plt.figure()
        plt.plot(x_data, tr_displacement_area_with_time)
        plt.xlabel("Time (s)")
        plt.ylabel("Integrated displacement")
        plt.show()
    
    return long_period_spectral_level
    
def get_P_arrival_time_at_station_from_NLLoc_event_hyp_file(NLLoc_event_hyp_filename, station):
    """Function to get theta and phi, in terms of lower hemisphere plot, for calculating radiation pattern components in other functions. Takes in NLLoc hyp file and station name, and returns theta and phi spherical coords."""
    # Get theta and phi in terms of lower hemisphere plot (as can get radiation pattern from lower or upper)
    var = subprocess.check_output('grep "'+station+'" '+NLLoc_event_hyp_filename+' | grep "? P" | '+"awk '{print $7$8$9}'", shell=True)
    P_arrival_time = UTCDateTime(var)
    return P_arrival_time
    
    
def get_theta_and_phi_for_stations_from_NLLoc_event_hyp_file(NLLoc_event_hyp_filename, station):
    """Function to get theta and phi, in terms of lower hemisphere plot, for calculating radiation pattern components in other functions. Takes in NLLoc hyp file and station name, and returns theta and phi spherical coords."""
    # Get theta and phi in terms of lower hemisphere plot (as can get radiation pattern from lower or upper)
    var = subprocess.check_output('grep "'+station+'" '+NLLoc_event_hyp_filename+' | grep "? P" | '+"awk '{print $23,$25}'", shell=True)
    azi = (float(var.split(" ")[0])/360.)*2*np.pi - np.pi # - pi as lower hemisphere plot so azimuth reflected by 180 degrees
    toa = (float(var.split(" ")[1])/360.)*2*np.pi
    # And get 3D coordinates for station (and find on 2D projection):
    theta = np.pi - toa # as +ve Z = down
    phi = azi
    if theta>np.pi/2.:
        theta = theta - np.pi
        phi=phi+np.pi
    return theta, phi
    
    
def get_normallised_rad_pattern_point_amplitude(theta, phi, full_MT_norm_in):
    """Function to get normallised point amplitude on radiation plot from spherical coords theta and phi (in radians), as well as the full normallised moment tensor solution."""
    
    # Get x_1,x_2,X_3 (cartesian vector components from spherical coords):
    x_1 = np.sin(theta)*np.cos(phi)
    x_2 = np.sin(theta)*np.sin(phi)
    x_3 = np.cos(theta)
    
    # And get A_rad_point:
    A_rad_point = (2*x_1*x_2*full_MT_norm_in[0,1]) + (2*x_1*x_3*full_MT_norm_in[0,2]) + (2*x_2*x_3*full_MT_norm_in[1,2]) + ((x_1**2)*full_MT_norm_in[0,0]) + ((x_2**2)*full_MT_norm_in[1,1]) + ((x_3**2)*full_MT_norm_in[2,2])
    
    return A_rad_point


def get_event_hypocentre_station_distance(NLLoc_event_hyp_filename, station):
    """Function to get event hypocentre to station distance, in km. Inputs are NLLoc hyp filename and station name. Output is STRAIGHT LINE distance of event from station in km."""
    # Get event coords (in km grid):
    var = subprocess.check_output('grep "HYPOCENTER" '+NLLoc_event_hyp_filename+" | awk '{print $3,$5,$7}'", shell=True)
    event_x = float(var.split(" ")[0])
    event_y = float(var.split(" ")[1])
    event_z = float(var.split(" ")[2])
    # Get station coords (in km grid):
    var = subprocess.check_output('grep "'+station+'" '+NLLoc_event_hyp_filename+' | grep "? P" | '+"awk '{print $19,$20,$21}'", shell=True)
    sta_x = float(var.split(" ")[0])
    sta_y = float(var.split(" ")[1])
    sta_z = float(var.split(" ")[2])    
    # Calculate station-event distance:
    sta_event_dist_km = np.sqrt(((event_x-sta_x)**2) + ((event_y-sta_y)**2) + ((event_z-sta_z)**2))
    return sta_event_dist_km


def calc_constant_C(rho,alpha,r_m,A_rad_point):
    """rho=density,alpha=Vp,r in metres, A is point radiation amplitude. Returns value of constant C."""
    C = 4*np.pi*rho*(alpha**3)*r_m/A_rad_point
    return C


def calc_const_freq_atten_factor(f_const_freq,Q,Vp,r_m):
    """Function to calculate attenuation factor based on constant attentuation with frequency assumption (Peters et al 2012)"""
    atten_fact = np.exp(np.pi*f_const_freq*r_m/(Vp*Q)) # Note that positive since getting back to atteniation at source from amplitude at reciever
    return atten_fact

# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Import mseed data and MT data:
    st = load_mseed(mseed_filename)
    uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)
    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
    MT_max_prob = MTs[:,index_MT_max_prob]
    full_MT_max_prob = get_full_MT_array(MT_max_prob)
        
    # 1. Correct for instrument response:
    instruments_poles_zeros_filename = []
    st_inst_resp_corrected = remove_instrument_response(st, instruments_gain_filename, instruments_poles_zeros_filename) # Note: NEED TO ADD DECONVOLUTION OF INST TRANSFER FUNC!!! (Although decided not to for this case, as majority of energy within flat response part of spectrum)
    
    # 2. Rotate trace into LQT:
    st_inst_resp_corrected_rotated = rotate_ZNE_to_LQT_axes(st_inst_resp_corrected, NLLoc_event_hyp_filename, stations_not_to_process)
        
    # Define station to process for:
    #station = "SKR01"
    for station in stations_to_calculate_moment_for:
        print "Processing data for station:", station
        # Check that station available and continue if not:
        if len(st_inst_resp_corrected_rotated.select(station=station,component="L")) == 0:
            continue
        
        # 3. Get displacement and therefroe long-period spectral level:
        # Get and trim trace to approx. event only
        tr = st_inst_resp_corrected_rotated.select(station=station, component="L")[0] #, component="Z")[0] # NOTE: MUST ROTATE TRACE TO GET TOTAL P!!!
        # Trim trace:
        P_arrival_time = get_P_arrival_time_at_station_from_NLLoc_event_hyp_file(NLLoc_event_hyp_filename, station)
        starttime = P_arrival_time-0.004
        tr.trim(starttime=starttime, endtime=starttime+0.2)
        if verbosity_level>=3:
            tr.plot()
        # And get spectral level from trace:
        long_period_spectral_level = get_displacement_integrated_over_time(tr)
        if verbosity_level>=2:
            print "Long period spectral level:", long_period_spectral_level
    
        # 4. Use event NonLinLoc solution to find theta and phi (for getting radiation pattern variation terms):
        theta, phi = get_theta_and_phi_for_stations_from_NLLoc_event_hyp_file(NLLoc_event_hyp_filename, station)

        # 5. Get normallised radiation pattern value for station (A(theta, phi, unit_matrix_M)):
        A_rad_point = get_normallised_rad_pattern_point_amplitude(theta, phi, full_MT_max_prob)
        if verbosity_level>=2:
            print "Amplitude value A (normallised radiation pattern value):", A_rad_point
    
        # 6. Get distance to station, r:
        r_km = get_event_hypocentre_station_distance(NLLoc_event_hyp_filename, station)
        r_m = r_km*1000.
        if verbosity_level>=2:
            print "r (m):", r_m
    
        # 7. Calculate constant, C:
        C = calc_constant_C(density,Vp,r_m,A_rad_point)
    
        # 8. Find attenuation factor:
        # Take FFT to get peak frequency:
        tr_data_freq_domain, freq = time_to_freq_domain_unnormallised(tr.data, tr.stats.sampling_rate)
        f_peak_idx = np.argmax(tr_data_freq_domain)
        f_peak = freq[f_peak_idx]
        # And calculate attentuation factor:
        atten_fact = calc_const_freq_atten_factor(f_peak,Q,Vp,r_m)
        if verbosity_level>=2:
            print "Attenuation factor (based upon constant frequency):", atten_fact
    
        # 9. Calculate sesimic moment:
        seis_M_0 = C*long_period_spectral_level*atten_fact
        if verbosity_level>=1:
            print "M_0 (Nm):", np.absolute(seis_M_0)
    
    print "Finished"
        