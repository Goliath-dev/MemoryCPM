# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:55:46 2024

@author: Admin
"""

# This file preprocesses the resting state data from the https://openneuro.org/datasets/ds003838/versions/1.0.6
# dataset. 

import mne
import os
import glob
import autoreject
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
import source_utils

raw_path = 'D:\Memory\EEG\DigitSpan'

conn_dir = 'Matrices\\Span Rest'
labels_dir = 'Labels\\Span Rest'
source_ts_dir = 'Source time series\\Span Rest'
clean_epochs_dir = 'Preprocessed data\\Span Rest'

conn_methods = ['wPLI', 'imCoh', 'PLV']
atlas = 'Destrieux'
fmin = (4, 8,  10, 13, 20, 30)
fmax = (8, 10, 13, 20, 30, 45)

raw_files = glob.glob(f'{raw_path}\\sub-*\\eeg\\sub-*_task-rest_eeg.set')
for file in raw_files:
    subject = file.split('\\')[-1].split('_')[0]
    raw = mne.io.read_raw_eeglab(file, preload=True)
    raw.resample(100.0)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    raw.filter(1, 45)
    
    epochs = mne.make_fixed_length_epochs(raw, duration=6.,
                                             preload=True, overlap=1.)
    
    ica = mne.preprocessing.ICA(method = 'infomax',
                                random_state=42)
    ica.fit(epochs)
    eog_channel = 'Fp2'  
    eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name = eog_channel, 
                                    measure = 'correlation', threshold = 0.4)
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs)
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(muscle_inds)
    ica.apply(epochs)
    
    ar = autoreject.AutoReject(random_state=42)
    clean_epochs, rej_log = ar.fit_transform(epochs, return_log=True)
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    # A dirty trick to prevent source rocenstruction from crash because of wrong 
    # reference projector due to removing bads in autorject.
    clean_epochs.set_eeg_reference('average', projection=True)
    clean_epochs.apply_proj()
    
    labels, ts = source_utils.fsaverage_time_courses(clean_epochs, clean_epochs.info, method='eLORETA', parc = 'aparc.a2009s')
    
    for method in conn_methods:
        # Compute the conncetivity matrices, given method and frequency bands.
        conn = spectral_connectivity_epochs(ts, method=method.lower(), sfreq=epochs.info['sfreq'], 
                                            fmin=fmin, fmax=fmax, 
                                            faverage=True, n_jobs=-1)
        os.makedirs(f'{conn_dir}\\{method}\\{atlas}', exist_ok=True)
        np.save(f'{conn_dir}\\{method}\\{atlas}\\{subject}', np.swapaxes(conn.get_data('dense'), 0, 1))
    os.makedirs(f'{labels_dir}\\{atlas}', exist_ok=True)
    os.makedirs(f'{source_ts_dir}\\{atlas}', exist_ok=True)
    os.makedirs(f'{clean_epochs_dir}/{atlas}', exist_ok=True)
    np.save(f'{labels_dir}/{atlas}/{subject}_labels', labels) 
    np.save(f'{source_ts_dir}/{atlas}/{subject}_source_ts', ts) 
    clean_epochs.save(f'{clean_epochs_dir}/{atlas}/{subject}_clean_epo.fif', fmt='double', overwrite=True)