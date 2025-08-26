# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:38:49 2025

Creating a SPAD-based LiDAR sensor simulation to generate histogram data
Model being implemented is based on the methodology outlined in https://doi.org/10.1038/s41598-022-27012-1

@author: jackaberdeen15
"""

#import scipy
import json
import numpy as np


def generate_scene_conditions(data):
    
    Rmax = data["Bin_num"] * data["Bin_width"] * 3e8 / 2
    
    # generate surfaces (max four) to be present in macropixel 
    R = np.random.random((np.random.randint(1,5)))*Rmax
    ref = np.random.random((len(R)))

    # distribute SPADs among the surfaces (each SPAD can only see one surface for simplicity)
    # generate n-1 unique split points between 1 and total SPAD number - 1
    splits = np.sort(np.random.choice(range(1, data["SPADnum"]), size=len(R)-1, replace=False))

    # add 0 and total SPAD number to the ends of the splits list and compute differences between points
    SPAD_dist = np.diff([0] + splits.tolist() + [data["SPADnum"]])
    
    return R, ref, SPAD_dist

def photon_numbers(data, R, ref, SPAD_dist, ambient):
    h = 3.626e-34 # plancks constant in Js
    c = 3e8 # speed of light in m/s
    Catm = 10e3 # atmospheric attenuation length in meters
    #Ptx = 0.94 * data["Epulse"] / data["Tpulse"]
    Fnum = data["FL"] / data["dlens"]
    Penergy = h * c / data["lambdae"] #energy of a photon in J
    Wbckg=0.4*ambient / 100e3 # solar background at 940 nm
    
    Psig = []
    Pnoise = []
    
    # estimate signal and noise photon numbers for each surface within macropixel
    for i in range(len(R)):
        Psource = data["Epulse"] * np.exp(-2*R[i]/Catm) * ref[i] * data["PDP"] * data["FF"] * data["Apix"] / 8 / Fnum**2 / np.pi / R[i]**2 / np.tan(data["thetae"])**2
        Psource = Psource / Penergy
        
        Psig.append(float(Psource * SPAD_dist[i] / data["SPADnum"]))
    
        Pback = Wbckg * np.exp(-R[i]/Catm) * ref[i] * data["PDP"] * data["FF"] * data["Apix"] / 8 / Fnum**2
        Pback = Pback * data["Bin_width"] * data["Bin_num"] / Penergy
    
        Pnoise.append(float(Pback * SPAD_dist[i] / data["SPADnum"] + data["DCR"] * data["Bin_width"] * data["Bin_num"]))
    
    return Psig, Pnoise

#Load SPAD-based LiDAR parameters from .json
with open("LiDAR_params.json","r") as file:
    data = json.load(file)

R, ref, SPAD_dist = generate_scene_conditions(data)

Psig, Pnoise = photon_numbers(data,R,ref,SPAD_dist,20e3)

print(Psig)
print(Pnoise)