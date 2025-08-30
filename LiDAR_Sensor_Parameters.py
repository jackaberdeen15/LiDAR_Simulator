# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 14:28:36 2025

@author: jackm
"""

import json
import math

Lidar = {
    # Pixel Parameters
    "PDP" : 0.04, #Photon detection probability %
    "FF" : 0.163, #Fill Factor %
    "Apix" : 114e-6*54e-6, # Pixel area m**2
    "Tdead" : 5e-9, # SPAD dead time
    "DCR" : 6.8e3, #Dark Count Rate Hz
    "SPADnum" : 16, #Number of SPADs per macropixel
    "QuenchType" : "passive", # quench type of the SPADs
    
    # Emitter Parameters
    "lambdae" : 940e-9, #Laser wavelength m
    "Epulse" : 640e-9, #Pulse Energy J
    "Tpulse" : 10e-9, #Pulse Width FWHM s
    "thetae" : math.radians(12.5), #Beam divergence rad
    
    # Optical Element Parameters
    "bw" : 10e-9, #Filter Bandwidth FWHM m
    "tau_opt" : 0.66, #Lens transmittance %
    "FL" : 7e-3, #Focus Length m
    "dlens" : 5e-3, #Lens Diameter m
    
    #System parameters
    "Bin_num" : 512, # number of bins in histogram
    "Bin_width" : 1e-9, # temporal width of histogram bins
    "Jitter" : 100e-12, #System jitter s
    "Laser_cycles" : 10 # number of laser cycles per exposure
    
    }

# writing dictionary to a file as JSON
with open('LiDAR_params.json', 'w') as f:
    json.dump(Lidar, f)