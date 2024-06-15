# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:40:29 2023

@author: Admin
"""

# This script contains some general purpose functions and pieces of code used across 
# the working memory project. 

import numpy as np
import csv
import glob
import networkx as nx
import seaborn as sns
import pandas as pd
from dataclasses import dataclass

def read_csv(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        arr = np.array([line for line in reader])
    return arr

def process_Sternberg(subj, data):
    """
    Processes the data containing results of Sternberg test and return the
    correctness rate and the response time. 

    Parameters
    ----------
    subj : str
        Subject which data is processed. For logging purposes only.
    data : np.array
        Sternberg test data. Must be read from corresponding .csv file. 

    Returns
    -------
    (float, float)
        A 2-element tuple with the 1st element being the correctness rate and
        the 2nd element being the response time.

    """
    if data.shape[0] < 15: 
        print(f'Participant {subj} did not get to the main part of the test.')
        return np.nan
    # For some reason one participant has an extra 'marker2' field. We do not use
    # it so this case might probably be worked out better than just rejecting the
    # participant, but hey, who cares about that one poor guy?
    if data.shape[1] != 28 and subj.startswith('chcon'): 
        print(f'Participant {subj} has inconsistent data.')
        return np.nan
    # The numbers of column and starting row of response correctness.
    resp_col = 18 if subj.startswith('chcon') else 34
    resp_start_row = 15 if subj.startswith('chcon') else 16
    
    # The number of column of response time.
    resp_time_col = 19 if subj.startswith('chcon') else 35
    
    responses = data[resp_start_row:, resp_col] # 1 for correct response and 0 for incorrect one
    responses = responses[responses != ''].astype(int)
    
    response_times = data[resp_start_row:, resp_time_col]
    response_times = response_times[response_times != ''].astype(float)
    response_times = response_times[responses == 1]
    
    corr_rate = len(responses[responses == 1]) / len(responses)
    mean_resp_time = np.mean(response_times)
    return corr_rate, mean_resp_time

def create_memory_dict():
    """
    Creates memory dictionary that maps subjects to the memory performance. 

    Returns
    -------
    mem_dict : {str -> (float, float)}
        A dictionary whose keys are subjects and values are 2-element tuples
        according to the process_Sternberg function.

    """
    files = glob.glob('D:\поведенческие\Pre-COVID data\Sternberg_ready\data\chcon_s_*.csv')
    mem_dict = {(subj:="_".join(file.split('\\')[-1].split('.')[0].split('_')[0:3])): 
                process_Sternberg(subj, read_csv(file)) for file in files}
    return mem_dict

def process_DS(file, N): 
    with open(file) as file: # A bit of a misuse, gotta add that to read_csv later. 
        tsv_file = csv.reader(file, delimiter="\t")
        arr = np.array([line for line in tsv_file])
    len_idx = 2 # The index of stimuli length; 5, 9 and 13 are available. 
    arr_N = arr[1:][arr[1:, len_idx] == str(N)]
    correct_arr = arr_N[:, 6].astype(int)
    return np.mean(correct_arr) / N
    
def DS_create_memory_dict(N):
    files = glob.glob('D:\\Memory\\EEG\\DigitSpan\\sub-*\\beh\\sub-*_task-memory_beh.tsv')
    print(len(files))
    mem_dict = {(subj:=file.split('\\')[-1].split('_')[0]): 
                process_DS(file, N) for file in files}
    return mem_dict

def DS_create_age_dict():
    with open('D:\\Memory\\EEG\\DigitSpan\\participants.tsv') as file: # A bit of a misuse, gotta add that to read_csv later. 
        tsv_file = csv.reader(file, delimiter="\t")
        arr = np.array([line for line in tsv_file])
    id_idx = 0
    age_idx = 1
    age_dict = {el[id_idx]: el[age_idx].astype(int) for el in arr[1:]}
    return age_dict

def DS_create_sex_dict():
    with open('D:\\Memory\\EEG\\DigitSpan\\participants.tsv') as file: # A bit of a misuse, gotta add that to read_csv later. 
        tsv_file = csv.reader(file, delimiter="\t")
        arr = np.array([line for line in tsv_file])
    id_idx = 0
    sex_idx = 2
    sex_dict = {el[id_idx]: el[sex_idx] for el in arr[1:]}
    return sex_dict
    
def modularity_curried(community_method):
    def func(G, weight):
        return nx.community.modularity(G, community_method(G), weight)
    return func

def apply_topo(conn_data, topo_method, threshold_perc):
    result = np.full((conn_data.shape[2],), np.nan)
    for i in range(conn_data.shape[2]):
        inv = np.zeros_like(conn_data[:,:,i])
        inv[conn_data[:,:,i] != 0] = 1 / np.abs(conn_data[:,:,i][conn_data[:,:,i] != 0])
        matrix = inv
        percentile = np.percentile(matrix, threshold_perc)
        matrix[matrix < percentile] = 0
        G = nx.from_numpy_array(matrix, create_using=nx.Graph)
        if nx.is_connected(G):
            result[i] = topo_method(G, weight='weight')
        else:
            result[i] = np.nan
    return result
    
def plot_scatter(x, y, axis_labels, ax):
    arr = np.array([x, y]).T
    data = pd.DataFrame(arr, columns=axis_labels)
    sns.regplot(data=data, x=axis_labels[0], y=axis_labels[1], ax=ax)

@dataclass
class TestCase:
    conn_method: str
    atlas: str
    net: str
    fmin: int
    fmax: int
    topo_method: str
    behav_parameter: str
    r: float
    p_value: float
    behav_values: list
    topo_values: list

