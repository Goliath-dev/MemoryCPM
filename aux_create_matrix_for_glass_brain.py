# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:34:59 2024

@author: Admin
"""

# This script creates a matrix of p-values in lesioning approach. 

import numpy as np
import glob
import csv
import netplotbrain
# import matplotlib.pyplot as plt
import templateflow.api as tf
import pandas as pd
from scipy.spatial.distance import squareform

CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # imCoh, wPLI or PLV
corr_mode = 'partial2' # corr, partial or partial2
atlas = 'Destrieux' # Destrieux or DK
validation = 'LOO' # k-fold or LOO
mode = 'Span Rest' # Span Rest or Span Task
stimuli_len = 13 # 5, 9 or 13
freq = '30-45' # 4-8, 8-13, 8-10, 10-13, 13-20, 20-30 or 30-45
prefix = 'pos' # posedges or negedges

# sample_number = {'Chel': '1st', 'Cuba': '2nd', 'German': '3rd'}
res_dir = f'Results\\CPMResults\\validation {validation}\\{mode}\\stim-{stimuli_len}\\{method}\\{atlas}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
label_dir = f'Labels\\{mode}\\{atlas}\\'

labels = np.load(f'{label_dir}\\sub-032_labels.npy', allow_pickle=True)

# edges_f = open(f'{res_dir}\\{sample_number[sample]} sample_{method}_{freq} Hz_{prefix}_{atlas}_edges.txt')
# p_f = open(f'{res_dir}\\{sample_number[sample]} sample_{method}_{freq} Hz_{prefix}_{atlas}_p_values.txt')
# p_values = list(filter(lambda s: s != '', p_f.read().split('\n')))

# edges = list(filter(lambda s: ''.join(s.split()) != 'to' and ''.join(s.split()) != '', edges_f.read().split('\n')))
# edges = list(''.join(edge.split()) for edge in edges)
# edges = np.array(edges)

# edges_from = edges[0::2]
# edges_to = edges[1::2]

ROI_offset = 2 if atlas == 'Destrieux' else 1
ROI_labels = np.load(label_dir + 'sub-032_labels.npy', allow_pickle=True)[:-ROI_offset]
ROI_names = [label.name for label in ROI_labels]

edges = np.load(f'{res_dir}\\{prefix}edges_{freq} Hz.npy')
edge_matrix = squareform(edges)
upper_tri_idcs = np.triu_indices(len(ROI_labels), 1)
idcs = np.where(edge_matrix)
unique_idcs = set([(i, j) if i > j else (j, i) for i, j in zip(idcs[0], idcs[1])])
edges_from = [f'{ROI_names[i]}' for i, j in unique_idcs] 
edges_to = [f'{ROI_names[j]}' for i, j in unique_idcs] 

# I read label names from a file file rather than [label.name for label in labels]
# because I use the matrix later on in an external R script that plots the glass brain.
# Therefore, the order of labels here matters (it defines the order of nodes in the
# adjacency matrix and it must fit the order of them in the R lib so that the nodes
# would be placed correctly), and I copied the nodes from R to a txt file to preserve
# the order.
# P. S.: To be fully precise, this is only true for the DK atlas, as it is built-in
# in the R library I use. The Destrieux atlas is custom and built by me (it is not built-in there),
# so technically I could just use [label.name for label in labels], as this is exactly the 
# way I built Destrieux atlas for the R script. I used a txt file instead, though, 
# for the sake of standartization (and a bit of laziness). 
label_names = open(f'{atlas}_label_names.txt').read()
label_names = label_names.split('\n')
label_names = np.array([''.join(name.split()) for name in label_names])
idcs_from = [np.argwhere(label_names == el)[0, 0] for el in edges_from]
idcs_to = [np.argwhere(label_names == el)[0, 0] for el in edges_to]

N = 148 if atlas == 'Destrieux' else 68
matrix = np.zeros((N, N))
for i, j in zip(idcs_from, idcs_to):
    matrix[i, j] = 1
    matrix[j, i] = 1


with open('matrix_glass_brain.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = np.arange(0, N).astype(str)
    writer.writerow(header)
    for row in matrix:
        writer.writerow(row)

