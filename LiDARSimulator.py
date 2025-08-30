# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 14:26:12 2025

Lidar Sim Class

@author: jackm
"""

import numpy as np
import json
from scipy.stats import norm

class LiDARSimulator:
    def __init__(self, param_file: str, dt: float = 50e-12):
        self.data = self._load_params(param_file)
        self.dt = dt
        self.c = 3e8
        self.n_spads = self.data["SPADnum"]
        self.bin_num = self.data["Bin_num"]
        self.bin_width = self.data["Bin_width"]
        self.tpulse = self.data["Tpulse"]
        self.tdead = self.data["Tdead"]
        self.l_cycles = self.data["Laser_cycles"]
        self.pdf = self._generate_pulse_pdf()
        self.pulse_len = len(self.pdf)
        self.cycle_len = int(self.bin_num * self.bin_width / self.dt)
        
        
    def _load_params(self, filepath):
        with open("LiDAR_params.json","r") as file:
            return json.load(file)
    
    def _generate_pulse_pdf(self):
        samples = np.linspace(-self.tpulse, self.tpulse, int(2 * self.tpulse / self.dt))
        pdf = norm.pdf(samples, 0, self.tpulse / 2.355)
        return pdf / np.sum(pdf)

    def generate_scene(self, batch_size):
        
        Rmax = self.bin_num * self.bin_width * self.c / 2
        batch_R = []
        batch_ref = []
        batch_SPAD_dist = []
        
        for _ in range(batch_size):
            # generate surfaces (max four) to be present in macropixel 
            R = np.random.random((np.random.randint(1,5)))*Rmax
            ref = np.random.random((len(R)))
            
            if len(R)==1:
                SPAD_dist = np.array([self.n_spads])
            else:
                # distribute SPADs among the surfaces (each SPAD can only see one surface for simplicity)
                # generate n-1 unique split points between 1 and total SPAD number - 1
                splits = np.sort(np.random.choice(range(1, self.n_spads), size=len(R)-1, replace=False))
            
                # add 0 and total SPAD number to the ends of the splits list and compute differences between points
                SPAD_dist = np.diff([0] + splits.tolist() + [self.n_spads])
                
            batch_R.append(R)
            batch_ref.append(ref)
            batch_SPAD_dist.append(SPAD_dist)
        
        return batch_R, batch_ref, batch_SPAD_dist
    
    def _photon_numbers(self, R, ref, SPAD_dist, ambient):
        h = 6.626e-34 # plancks constant in Jsc = 3e8 # speed of light in m/s
        Catm = 10e3 # atmospheric attenuation length in meters
        #Ptx = 0.94 * data["Epulse"] / data["Tpulse"]
        Fnum = self.data["FL"] / self.data["dlens"]
        Penergy = h * self.c / self.data["lambdae"] #energy of a photon in J
        Wbckg=0.4*ambient / 100e3 # solar background at 940 nm
        
        # mask for valid surfaces
        mask = R > 0
        
        #prevent divide by zero or exp(inf) by safely replacing 0s
        R_safe = np.where(mask, R, 1.0)
        
        # estimate the number of incident signal photons per surface per SPAD
        Psource = self.data["Epulse"] * np.exp(-2*R_safe/Catm) * ref * self.data["PDP"] * self.data["FF"] * self.data["Apix"] / 8 / Fnum**2 / np.pi / R_safe**2 / np.tan(self.data["thetae"])**2
        Psource = Psource / Penergy
        
        Psig = Psource / self.n_spads
        
        # estimate the number of incident noise photons per surface per SPAD (from ambient and DCR)
        Pback = Wbckg * np.exp(-R_safe/Catm) * ref * self.data["PDP"] * self.data["FF"] * self.data["Apix"] / 8 / Fnum**2
        Pback = Pback * self.bin_width * self.bin_width / Penergy

        Pnoise = Pback / self.n_spads + self.data["DCR"] * self.bin_width * self.bin_num
        
        return np.where(mask,Psig,0.0), np.where(mask, Pnoise, 0.0)
    
    def photon_numbers(self, R_batch, ref_batch, SPAD_dist_batch, ambient):
        return zip(*[self._photon_numbers(R, ref, SPAD_dist, ambient)
                     for R, ref, SPAD_dist in zip(R_batch, ref_batch, SPAD_dist_batch)])
    
    # change from ragged lists to consistently sized arrays
    def expand_to_SPADs(self, Psig, Pnoise, R, SPAD_dists):
        Psig_expand = np.zeros((len(Psig), self.n_spads))
        Pnoise_expand = np.zeros((len(Pnoise), self.n_spads))
        R_expand = np.zeros((len(R), self.n_spads))
        
        for i in range(len(Psig)):
            Psig_expand[i] = np.repeat(Psig[i], SPAD_dists[i])
            Pnoise_expand[i] = np.repeat(Pnoise[i], SPAD_dists[i])
            R_expand[i] = np.repeat(R[i], SPAD_dists[i])

        return Psig_expand, Pnoise_expand, R_expand
    
    def _SPAD_Pulse_Model(self, timestamps):
        Tdead = int(self.tdead / self.dt) 
        Pulse = np.zeros(timestamps.shape)
        for t in range(len(Pulse)):
            if timestamps[t] > 0:
                Pulse[t:t+Tdead] = 1.0
        return Pulse
    
    def _histogram_generation_cycle(self, R, Psig, Pnoise):
        pulse_bins = 2*int(self.tpulse / self.dt)           # pulse length in bins
        bin_to_dt = int(self.bin_width / self.dt)
        tstmp_buffer_merged = np.zeros((self.cycle_len,))
        Hist = np.zeros((self.bin_num,), dtype=np.uint16)

        # Initialize buffers
        tstmp_buffer = np.zeros((self.n_spads, self.cycle_len), dtype=int)

        for i in range(self.n_spads):
            # Scale signal pulse shape to photon rate per bin
            sig_rate = self.pdf * Psig[i]

            # Generate signal photons with Poisson draw
            sig_counts = np.random.poisson(sig_rate)

            # Time delay due to range
            tof = 2 * R[i] / self.c   # in seconds
            start = int(tof / self.dt - pulse_bins / 2)
            
            end = start + pulse_bins
            
            
            # Ensure start and end are within bounds
            if start >= self.cycle_len or end <= 0:
                continue  # Skip this SPAD if signal would land entirely out of range
            
            # close distances can cause start of pulse to lie before TDC starts, set to 0 if start is negative
            if start < 0:
                sig_counts = sig_counts[-start:]  # truncate to fit
                start = 0
            
            if end > self.cycle_len:
                end = self.cycle_len
                sig_counts = sig_counts[:end - start]  # truncate to fit
                
            #print(start,end, sig_counts.shape, tstmp_buffer[i,start:end].shape)
            tstmp_buffer[i, start:end] += sig_counts

            # Add ambient noise uniformly across entire cycle
            ambient_rate = Pnoise[i] / self.cycle_len
            noise_counts = np.random.poisson(ambient_rate, size=self.cycle_len)
            tstmp_buffer[i] += noise_counts
            
            SPAD_pulses = self._SPAD_Pulse_Model(tstmp_buffer[i])
            tstmp_buffer_merged = np.logical_or(tstmp_buffer_merged,SPAD_pulses)
        
        for bin_i in range(self.bin_num):
            Hist[bin_i] = np.sum(np.diff(tstmp_buffer_merged[bin_i * bin_to_dt:(bin_i+1) * bin_to_dt]) > 0)
        
        return Hist
    
    def histogram_generation(self, R_s_batch, Psig_batch, Pnoise_batch):
        batch_size = len(Psig_batch)
        Histograms = np.zeros((batch_size,self.bin_num))
        
        #cycle through each proposed waveform and generate the histogram
        for b in range(batch_size):
            #build histogram up over multiple laser cycles
            for cycle in range(self.l_cycles):
                Histograms[b] += self._histogram_generation_cycle(R_s_batch[b], Psig_batch[b], Pnoise_batch[b])
        
        return Histograms
    