# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 21:45:17 2023

@author: Admin
"""

# This script prepares pictures for the memory article. 

import numpy as np
import scipy as sp
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import pandas as pd
import csv
from scipy.spatial.distance import squareform

def plot_circle(ROI_labels, edges, ax):
    ROI_names = [label.name for label in ROI_labels]
    lh_labels = [name for name in ROI_names if name.endswith('lh')]
    label_ypos = list()
    for name in lh_labels:
        idx = ROI_names.index(name)
        ypos = np.mean(ROI_labels[idx].pos[:, 1])
        label_ypos.append(ypos)
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)  
    node_angles = circular_layout(ROI_names, node_order, start_pos=90,
                              group_boundaries=[0, len(ROI_names) / 2])
    plot_connectivity_circle(sp.spatial.distance.squareform(edges.astype(float)), 
                             ROI_names, node_angles=node_angles,
                             ax = ax, fontsize_title=20, fontsize_names=20,
                             colorbar=False, facecolor='white', textcolor='black',
                             node_edgecolor='black')

def plot_scatter(intel_arr, pred, axis_labels, title, ax):
    arr = np.array([intel_arr, pred]).T
    data = pd.DataFrame(arr, columns=axis_labels)
    sns.regplot(data=data, x=axis_labels[0], y=axis_labels[1], ax=ax)
    # intel_utils.plot_scatter_against_function([intel_arr], [pred], [],
    #                                           axis_labels=axis_labels, 
    #                                           title=title, 
    #                                           ax = ax)
CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
corr_mode = 'partial2' # corr, partial or partial2
atlas = 'Destrieux' # Destrieux or DK
freq = '30-45' # 4-8, 8-13, 8-10, 10-13, 13-20, 20-30 or 30-45
prefix = 'pos' # posedges or negedges
validation = 'LOO' # k-fold or LOO
mode = 'Span Rest' # Span Rest or Span Task
stimuli_len = 13 # 5, 9 or 13

label_dir = f'Labels\\{mode}\\{atlas}\\'
res_dir = f'Results\\CPMResults\\validation {validation}\\{mode}\\stim-{stimuli_len}\\{method}\\{atlas}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
edge_file = f'{res_dir}{prefix}edges_{freq} Hz.npy'
pred_file = f'{res_dir}{prefix}_pred_{freq} Hz.npy'
intel_file = f'{res_dir}intel_arr_{prefix}_{freq} Hz.npy'

ROI_offset = 2 if atlas == 'Destrieux' else 1
ROI_labels = np.load(label_dir + 'sub-032_labels.npy', allow_pickle=True)[:-ROI_offset]
ROI_names = [label.name for label in ROI_labels]

edges = np.load(edge_file)
pred = np.load(pred_file)
intel = np.load(intel_file)

matrix = sp.spatial.distance.squareform(edges)
idcs = np.argwhere(matrix)
tril_idcs = np.array(np.tril_indices(matrix.shape[0])).T
# I guess, not the most obvious way to remove duplicating edges, but whatever, it works.
idcs = set([tuple(x) for x in idcs]) & set([tuple(x) for x in tril_idcs])
# To form a list of valuable edges, see Supplementary section. 
edge_names = [f'{ROI_names[idx[0]]} to {ROI_names[idx[1]]}'for idx in idcs]

fig = plt.figure(figsize = (40, 20))
# fig.set_facecolor('white')
grid_spec = fig.add_gridspec(1, 2)

circle_ax = fig.add_subplot(grid_spec[0, 0], projection='polar')
# circle_ax.set_facecolor('white')
# circle_title = f'{method}, {freq} Hz'

scatter_ax = fig.add_subplot(grid_spec[0, 1])
axis_labels = ['Observed memory score', 'Predicted memory score']
scatter_title = ''
R2 = r2_score(intel, pred)
mae = mean_absolute_error(intel, pred)
scatter_ax.text(x=0.5, y=0.1, s=f'R² = {round(R2, 2)}\nMAE = {round(mae, 2)}', transform=scatter_ax.transAxes)
scatter_ax.set_facecolor('white')
scatter_ax.grid()
# scatter_ax.grid(c='k')
sns.set(font_scale=3)

plot_scatter(intel, pred, axis_labels, scatter_title, scatter_ax)
plot_circle(ROI_labels, edges, circle_ax)

# with open('matrix.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     matrix = squareform(edges).astype(int)
#     for row in matrix:
#         writer.writerow(row)














