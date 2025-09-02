# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:38:49 2025

Creating a SPAD-based LiDAR sensor simulation to generate histogram data
Model being implemented is based on the methodology outlined in https://doi.org/10.1038/s41598-022-27012-1

@author: jackaberdeen15
"""

#import scipy
import numpy as np
import LiDARSimulator

sim = LiDARSimulator.LiDARSimulator("LiDAR_params.json")

R_batch, ref_batch, SPAD_dist_batch = sim.generate_scene(10)

Psig, Pnoise = sim.photon_numbers(R_batch, ref_batch, 20e3)

Psig_SPAD, Pnoise_SPAD, R_SPAD = sim.expand_to_SPADs(Psig, Pnoise,  R_batch, SPAD_dist_batch)

Histograms = sim.histogram_generation(R_SPAD, Psig_SPAD, Pnoise_SPAD)

#%%
sel = 1
print(R_batch[sel])
print(ref_batch[sel])
print(SPAD_dist_batch[sel])
print(Psig_SPAD[sel])
print(Pnoise_SPAD[sel])
print(np.sum(Histograms[sel]))
