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
import scipy
import matplotlib.pyplot as plt


dt = 50e-12

def generate_scene_conditions(data, batch_size):
    
    Rmax = 20#data["Bin_num"] * data["Bin_width"] * 3e8 / 2
    batch_R = []
    batch_ref = []
    batch_SPAD_dist = []
    
    max_spad = data["SPADnum"]
    
    for _ in range(batch_size):
        # generate surfaces (max four) to be present in macropixel 
        R = np.random.random((np.random.randint(1,5)))*Rmax
        ref = np.random.random((len(R)))
        
        if len(R)==1:
            SPAD_dist = np.array([max_spad])
        else:
            # distribute SPADs among the surfaces (each SPAD can only see one surface for simplicity)
            # generate n-1 unique split points between 1 and total SPAD number - 1
            splits = np.sort(np.random.choice(range(1, max_spad), size=len(R)-1, replace=False))
        
            # add 0 and total SPAD number to the ends of the splits list and compute differences between points
            SPAD_dist = np.diff([0] + splits.tolist() + [max_spad])
        
        batch_R.append(R)
        batch_ref.append(ref)
        batch_SPAD_dist.append(SPAD_dist)
    
    return batch_R, batch_ref, batch_SPAD_dist

def photon_numbers_scene(data, R, ref, SPAD_dist, ambient):
    h = 6.626e-34 # plancks constant in Js
    c = 3e8 # speed of light in m/s
    Catm = 10e3 # atmospheric attenuation length in meters
    #Ptx = 0.94 * data["Epulse"] / data["Tpulse"]
    Fnum = data["FL"] / data["dlens"]
    Penergy = h * c / data["lambdae"] #energy of a photon in J
    Wbckg=0.4*ambient / 100e3 # solar background at 940 nm
    
    # mask for valid surfaces
    mask = R > 0
    
    #prevent divide by zero or exp(inf) by safely replacing 0s
    R_safe = np.where(mask, R, 1.0)
    
    # estimate the number of incident signal photons per surface per SPAD
    Psource = data["Epulse"] * np.exp(-2*R_safe/Catm) * ref * data["PDP"] * data["FF"] * data["Apix"] / 8 / Fnum**2 / np.pi / R_safe**2 / np.tan(data["thetae"])**2
    Psource = Psource / Penergy
    
    Psig = Psource / data["SPADnum"]
    
    # estimate the number of incident noise photons per surface per SPAD (from ambient and DCR)
    Pback = Wbckg * np.exp(-R_safe/Catm) * ref * data["PDP"] * data["FF"] * data["Apix"] / 8 / Fnum**2
    Pback = Pback * data["Bin_width"] * data["Bin_num"] / Penergy

    Pnoise = Pback / data["SPADnum"] + data["DCR"] * data["Bin_width"] * data["Bin_num"]
    
    return np.where(mask,Psig,0.0), np.where(mask, Pnoise, 0.0)

def photon_numbers_batch(data, R_batch, ref_batch, SPAD_dist_batch, ambient):
    return zip(*[photon_numbers_scene(data, R, ref, SPAD_dist, ambient)
                 for R, ref, SPAD_dist in zip(R_batch, ref_batch, SPAD_dist_batch)])

def expand_to_SPADs(data,Psig, Pnoise, R, SPAD_dists):
    n_spads = data["SPADnum"]
    Psig_expand = np.zeros((len(Psig), n_spads))
    Pnoise_expand = np.zeros((len(Pnoise), n_spads))
    R_expand = np.zeros((len(R), n_spads))
    
    for i in range(len(Psig)):
        Psig_expand[i] = np.repeat(Psig[i], SPAD_dists[i])
        Pnoise_expand[i] = np.repeat(Pnoise[i], SPAD_dists[i])
        R_expand[i] = np.repeat(R[i], SPAD_dists[i])

    return Psig_expand, Pnoise_expand, R_expand


def SPAD_Pulse_Model(data, timestamps):
    Tdead = int(data["Tdead"] / dt) 
    Pulse = np.zeros(timestamps.shape)
    for t in range(len(Pulse)):
        if timestamps[t] > 0:
            Pulse[t:t+Tdead] = 1.0
    return Pulse

def histogram_generation_cycle(data, R, Psig, Pnoise, pdf):
    c = 3e8 # speed of light in m/s
    Laser_cycles = data["Laser_cycles"]
    Tpulse = data["Tpulse"]
    Bin_num = data["Bin_num"]
    Bin_width = data["Bin_width"]
    cycle_len = int(Bin_num*Bin_width / dt)
    N_SPADs = data["SPADnum"]
    pulse_bins = 2*int(Tpulse / dt)           # pulse length in bins
    bin_to_dt = int(Bin_width / dt)
    Tdead = int(data["Tdead"] / dt)
    tstmp_buffer_merged = np.zeros((cycle_len,))
    Hist = np.zeros((Bin_num,), dtype=np.uint16)

    # Initialize buffers
    tstmp_buffer = np.zeros((N_SPADs, cycle_len), dtype=int)

    for i in range(N_SPADs):
        # Scale signal pulse shape to photon rate per bin
        sig_rate = pdf * Psig[i]

        # Generate signal photons with Poisson draw
        sig_counts = np.random.poisson(sig_rate)

        # Time delay due to range
        tof = 2 * R[i] / c   # in seconds
        shift_bins = int(tof / dt - pulse_bins / 2)

        # Shift signal into histogram
        start = shift_bins
        end = shift_bins + pulse_bins
        if end > cycle_len:
            end = cycle_len
            sig_counts = sig_counts[:end - start]  # truncate to fit
        tstmp_buffer[i, start:end] += sig_counts

        # Add ambient noise uniformly across entire cycle
        ambient_rate = Pnoise[i] / cycle_len
        noise_counts = np.random.poisson(ambient_rate, size=cycle_len)
        tstmp_buffer[i] += noise_counts
        
        SPAD_pulses = SPAD_Pulse_Model(data, tstmp_buffer[i])
        tstmp_buffer_merged = np.logical_or(tstmp_buffer_merged,SPAD_pulses)
    
    for bin_i in range(Bin_num):
        Hist[bin_i] = np.sum(np.diff(tstmp_buffer_merged[bin_i * bin_to_dt:(bin_i+1) * bin_to_dt]) > 0)
    
    return Hist

def histogram_generation_exposure(data, R_s_batch, Psig_batch, Pnoise_batch):
    batch_size = len(Psig_batch)
    Histograms = np.zeros((batch_size,data["Bin_num"]))
    cycle_num = data["Laser_cycles"]
    
    samples = np.linspace(-data["Tpulse"],data["Tpulse"], int(2*data["Tpulse"] / dt))
    pdf = scipy.stats.norm.pdf(samples, 0, data["Tpulse"]/2.355)
    
    # normalise pdf
    pdf /= np.sum(pdf)
    
    for b in range(batch_size):
        for cycle in range(cycle_num):
            Histograms[b] += histogram_generation_cycle(data, R_s_batch[b], Psig_batch[b], Pnoise_batch[b], pdf)
    
    return Histograms

#def timestamp_generation_batch(data, Psig_batch, Pnoise_batch, SPAD_dist_batch):
    

#Load SPAD-based LiDAR parameters from .json
with open("LiDAR_params.json","r") as file:
    data = json.load(file)

R_batch, ref_batch, SPAD_dist_batch = generate_scene_conditions(data,2)

Psig_batch, Pnoise_batch = photon_numbers_batch(data,R_batch,ref_batch, SPAD_dist_batch, 20e3)

Psig_s_batch, Pnoise_s_batch, R_s_batch = expand_to_SPADs(data, Psig_batch, Pnoise_batch, R_batch, SPAD_dist_batch)

histograms = histogram_generation_exposure(data, R_s_batch, Psig_s_batch, Pnoise_s_batch)

print(R_batch[0])
print(ref_batch[0])
print(Psig_batch[0])
print(Pnoise_batch[0])
print(np.sum(histograms[0]))
