# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:53:45 2021

@author: baoxiaoge
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pickle
from scipy.io import loadmat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

seed = 13648  # Seed random number generators for reproducibility
pic_label = [1,2,3]
for pic_chose in pic_label:
    if pic_chose == 1:
        A = loadmat('PPI2')['A']
        G = nx.from_numpy_matrix(A)
    elif pic_chose == 2:
        G = nx.generators.connected_watts_strogatz_graph(100, k=10, p = 0.1, tries=100, seed=seed)
    else:
        G = nx.generators.random_graphs.erdos_renyi_graph(100, p=0.1, seed=seed, directed=False)
        nx.write_gpickle(G,'G_three.gpl') #write ER

    line_num = 0
    tri_num= 0
    square_num= 0
    five_num= 0
    for edge in (list(G.edges())):
        t1 = time.time()
        H = G.copy()
        H.remove_edge(edge[0],edge[1])
        try:        
            aa = nx.shortest_path_length(H,source=edge[0],target=edge[1])
            if aa == 2:
                tri_num +=1 
            elif aa == 3:
                square_num += 1
            elif aa == 4:
                five_num += 1
        except:
            line_num += 1
        
        t2 = time.time()
        print('time',t2- t1)

    motif_dicone = dict()
    motif_dicone.setdefault('line',line_num)
    motif_dicone.setdefault('triangle',tri_num)
    motif_dicone.setdefault('square',square_num)
    motif_dicone.setdefault('five',five_num)

    save_dict(motif_dicone, 'motif_one_'+str(pic_chose))
    motif_dicone = load_dict('motif_one_'+str(pic_chose))
    print('load pic:',pic_chose)
