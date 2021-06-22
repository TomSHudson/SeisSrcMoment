#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Functions to calculate stress drop using moment magnitude results.

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



# ------------------- Define generally useful functions -------------------
def calc_stress_drop(M_0, f_c, src_model_shear_vel, phase_to_use="P", uncertainty_M_0=[], uncertainty_f_c=[]):
    """Function to calculate stress drop for one or a number of events.
    This function currently uses the source model of Kaneko and Shearer (2014).
    This model is based upon a Brune model. The model is specifically for a 
    circular, symetrical fault with a singular crack model with spontaneous 
    healing of slip.
    Note that the values of k used to calculate the stress drop are a spherical 
    average, and so therefore do not fully account for geometrical variations 
    in apparent corner frequency.
    Note that this function currently assumes a Poissonian solid (i.e. v = 0.25).
    
    Arguments:
    Required:
    M_0 - Value/s of seismic moment for an earthquake or earthquakes. (float 
            or array of floats)
    f_c - Value/s of corner frequency for an earthquake or earthquakes. (float 
            or array of floats)
    src_model_shear_vel - The shear-wave velocity at the earthquake source 
                            location, in m/s. (float)
    Optional:
    phase_to_use - The phase used to find M_0 and f_c values. Default is P. Can 
                    be P or S. (str)

    Returns:
    stress_drop - The stress drop/s of the earthquake data inputs. (float or array 
                    of floats)
    fault_radius - The fault radius/radii of the earthquake data inputs. (float or 
                    array of floats)
    uncertainty_M_0 - The uncertainty in the M_0 value/s provided. If specified, 
                        the uncertainty in stress-drop/s and fault radii will be 
                        returned. Default is [], and will not return uncertainties.
                        Note that both uncertainty_M_0 and uncertainty_f_c are 
                        required in order to calculate the uncertainties. (float or 
                        array of floats)
    uncertainty_f_c - The uncertainty in the f_c value/s provided. If specified, 
                        the uncertainty in stress-drop/s and fault radii will be 
                        returned. Default is [], and will not return uncertainties.
                        Note that both uncertainty_M_0 and uncertainty_f_c are 
                        required in order to calculate the uncertainties. (float or 
                        array of floats)
    """
    # Get inputs into appropriate format for further calculation:
    try:
        len(M_0)
        len(f_c)
        M_0 = np.array(M_0)
        f_c = np.array(f_c)
    except TypeError:
        M_0 = np.array(M_0)
        f_c = np.array(f_c)
    if (len(uncertainty_M_0) > 0) and (len(uncertainty_f_c) > 0):
        try:
            len(uncertainty_M_0)
            len(uncertainty_f_c)
            uncertainty_M_0 = np.array(uncertainty_M_0)
            uncertainty_f_c = np.array(uncertainty_f_c)
        except TypeError:
            uncertainty_M_0 = np.array(uncertainty_M_0)
            uncertainty_f_c = np.array(uncertainty_f_c)
    
    # Calculate fault radius from corner frequency:
    # (Note: Uses the Kaneko and Shearer (2014) source model)
    if phase_to_use == "P":
        k = 0.38
    elif phase_to_use == "S":
        k = 0.26
    fault_radius = k * src_model_shear_vel / f_c # a = k * beta / f_c
    # And calculate uncertainty in fault radius, if specified:
    if (len(uncertainty_M_0) > 0) and (len(uncertainty_f_c) > 0):
        fault_radius_plus_uncert = k * src_model_shear_vel / (f_c + uncertainty_f_c)
        fault_radius_minus_uncert = k * src_model_shear_vel / (f_c - uncertainty_f_c)
        uncertainty_fault_radius = ( np.abs(fault_radius_plus_uncert - fault_radius) + np.abs(fault_radius_minus_uncert - fault_radius) ) / 2.

    # Calculate stress drop from fault radius:
    # (Note that the following equation is for the special case of a circular source 
    # within a Poissonian solid, i.e. v=0.25)
    stress_drop = (7/16) * M_0 / (fault_radius**3)
    # And calculate uncertainty in stress-drop, if specified:
    if (len(uncertainty_M_0) > 0) and (len(uncertainty_f_c) > 0):
        uncertainty_stress_drop = np.sqrt( (((7/16) / (fault_radius**3))**2) * (uncertainty_M_0**2) + (((-21/16) * M_0 / (fault_radius ** 4))**2) * (uncertainty_fault_radius**2) ) # (Note that this is only a calculus-based approximation)

    # And tidy outputs if neccessary:
    if len(stress_drop) == 1:
        stress_drop = stress_drop[0]
        fault_radius = fault_radius[0]

    if (len(uncertainty_M_0) > 0) and (len(uncertainty_f_c) > 0):
        return stress_drop, fault_radius, uncertainty_stress_drop, uncertainty_fault_radius
    else:
        return stress_drop, fault_radius



        
