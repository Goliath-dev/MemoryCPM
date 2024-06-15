# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:09:28 2024

@author: Admin
"""

# This script calculates p-values of r-squares and saves them to a given directory.

import csv
import numpy as np
from sklearn.metrics import r2_score
import glob
import os

def perm_test(behav_arr, pred, n_perms = 10000):
    R2_arr = np.zeros(n_perms)
    rng = np.random.default_rng()
    for i in range(n_perms):
        N = len(pred)
        rand_idcs = np.arange(0, N)
        rng.shuffle(rand_idcs)
        permuted_pred = pred[rand_idcs]
        R2_arr[i] = r2_score(behav_arr, permuted_pred)
        
    true_R2 = r2_score(behav_arr, pred)
    p = len(R2_arr[R2_arr > true_R2]) / n_perms
    return p

# res_dir = f'Results\\CPMResults\\validation {validation}\\{mode}\\stim-{stimuli_len}\\{method}\\{atlas}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
res_dir = 'Results\\CPMResults'
p_dir = 'Results\\R2PValues'

# intel_files = glob.glob(f'{res_dir}\\**\\intel_arr_*.npy', recursive=True)
pred_files = glob.glob(f'{res_dir}\\**\\*_pred_*.npy', recursive=True)

for pred_file in pred_files:
    # Directory is ordered in the following way: Results\CPMResults\[validation type]\
    # [mode, rest or task]\[stimuli length]\[FC method]\[atlas]\[correlation type]\
    # [CPM order]\[p-value threshold]\[files are here], so that's where numbers below
    # come from. 
    parsed_filename = pred_file.split('\\')
    mode = parsed_filename[3]
    span_len = parsed_filename[4]
    method = parsed_filename[5]
    atlas = parsed_filename[6]
    corr_sign = parsed_filename[10].split('_')[0]
    freq = parsed_filename[10].split('.')[0].split('_')[-1]
    
    behav_file = pred_file.replace(f'{corr_sign}_pred', f'intel_arr_{corr_sign}')
    pred = np.load(pred_file)
    behav = np.load(behav_file)
    p = perm_test(behav, pred)
    R2 = r2_score(behav, pred)
    p_file_name = f'{p_dir}\\{mode}_{span_len}_{method}_{atlas}_{corr_sign}_{freq}.csv'
    os.makedirs(p_dir, exist_ok=True)
    with open(p_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['R2', 'p'])
        writer.writerow([R2, p])
    
    
