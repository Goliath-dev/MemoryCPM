# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:03:36 2024

@author: Admin
"""

# This file preprocesses the task data from the https://openneuro.org/datasets/ds003838/versions/1.0.6
# dataset. 

import mne
import os
import glob
import autoreject
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from mne import compute_covariance, setup_source_space, make_forward_solution 
from mne import read_labels_from_annot, extract_label_time_course
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.datasets import fetch_fsaverage
import time
import os.path as op

start = time.time()

raw_path = 'D:\Memory\EEG\DigitSpan'

span_length = 5

conn_dir = f'Matrices\\Span Task\\stim-{span_length}'
labels_dir = 'Labels\\Span Task'
source_ts_dir = f'Source time series\\Span Task\\stim-{span_length}'
clean_epochs_dir = f'Preprocessed data\\Span Task\\stim-{span_length}'


conn_methods = ['wPLI', 'imCoh', 'PLV']
atlases = {'DK': 'aparc', 'Destrieux': 'aparc.a2009s'}

fmin = (4, 8,  10, 13, 20, 30)
fmax = (8, 10, 13, 20, 30, 45)

raw_files = glob.glob(f'{raw_path}\\sub-*\\eeg\\sub-*_task-memory_eeg.set')
for file in raw_files:
    subject = file.split('\\')[-1].split('_')[0]
    # if os.path.exists(f'{conn_dir}\\{conn_methods[-1]}\\{list(atlases.keys())[-1]}\\{subject}.npy'): 
    #     print(f'Subject {subject} has already been calculated.')
    #     continue
    try:
        raw = mne.io.read_raw_eeglab(file)
    except OSError:
        print('File is corrupted.')
        continue
    
    raw.resample(100)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    raw.filter(1, 45)
    events = mne.events_from_annotations(raw)
    digit_appeared_keys = ['6001050', '6001051', '6002050', '6002051', '6003050', 
                            '6003051', '6004050', '6004051','6005050', '6005051']
    # digit_appeared_keys = ['6001090', '6001091', '6002090', '6002091', '6003090', 
    #                         '6003091', '6004090', '6004091','6005090', '6005091',
    #                         '6006090', '6006091', '6007090','6007091', '6008090',
    #                         '6008091', '6009090', '6009091']
    # digit_appeared_keys = ['6001130', '6001131', '6002130', '6002131', '6003130', 
    #                         '6003131', '6004130', '6004131','6005130', '6005131',
    #                         '6006130', '6006131', '6007130','6007131', '6008130',
    #                         '6008131', '6009130', '6009131', '6010130', '6010131',
    #                         '6011130', '6011131', '6012130', '6012131', '6013130',
    #                         '6013131']
    digits_appeared_ids = [events[1][key] for key in digit_appeared_keys if key in events[1].keys()]
    
    epochs = mne.Epochs(raw, events[0], event_id=digits_appeared_ids,
                        tmin=-0.5, tmax=2.0,baseline=(-0.5, 0), preload=True)
    ica = mne.preprocessing.ICA(random_state=42)
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
    
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, 
                            min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    # A dirty trick to prevent source rocenstruction from crash because of wrong 
    # reference projector due to removing bads in autorject.
    clean_epochs.set_eeg_reference('average', projection=True)
    clean_epochs.apply_proj()
    
    os.makedirs(f'{clean_epochs_dir}', exist_ok=True)
    clean_epochs.save(f'{clean_epochs_dir}/{subject}_clean_epo.fif', fmt='double', overwrite=True)
    # Solve the inverse problem and extract the time series of the sources.
    # Also extract the label objects.
    for atlas in atlases:
        noise_cov = compute_covariance(clean_epochs, tmax=0., method=['shrunk', 'empirical'])
        fs_dir = fetch_fsaverage()
        subjects_dir = os.path.dirname(fs_dir)
        fs_subject = 'fsaverage'
        trans = 'fsaverage'
        src = setup_source_space(subject=fs_subject, subjects_dir=subjects_dir, n_jobs=-1)
        bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        fwd = make_forward_solution(clean_epochs.info, trans=trans, src=src, 
                                        eeg=True, bem=bem, mindist=5.0, n_jobs=-1)
        inverse = make_inverse_operator(clean_epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
        snr = 3
        lambda2 = 1. / snr ** 2
        stc = apply_inverse_epochs(clean_epochs, inverse, lambda2, method='eLORETA')
        labels = read_labels_from_annot(subject=fs_subject, subjects_dir=subjects_dir, parc = atlases[atlas])
        offset = 1 if atlas == 'DK' else 2
        time_courses = extract_label_time_course(stc, labels[:-offset], src)
        os.makedirs(f'{labels_dir}\\{atlas}', exist_ok=True)
        os.makedirs(f'{source_ts_dir}\\{atlas}', exist_ok=True)
        np.save(f'{labels_dir}\\{atlas}\\{subject}_labels', labels)
        np.save(f'{source_ts_dir}\\{atlas}\\{subject}_source_ts', time_courses)
            
        for method in conn_methods:
            # Compute the conncetivity matrices, given method and frequency bands.
            conn = spectral_connectivity_epochs(time_courses, method=method.lower(), sfreq=epochs.info['sfreq'], 
                                                fmin=fmin, fmax=fmax, 
                                                faverage=True, n_jobs=-1)
            os.makedirs(f'{conn_dir}\\{method}\\{atlas}', exist_ok=True)
            np.save(f'{conn_dir}\\{method}\\{atlas}\\{subject}', np.swapaxes(conn.get_data('dense'), 0, 1))
    
stop = time.time()
print(f'Seconds elapsed: {stop - start}')
