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
from collections import Counter
import pickle
from scipy.io import loadmat
import networkx as nx

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
        
    motif_dic = dict()
    motif_dic.setdefault('triangle',dict())
    motif_dic.setdefault('square',dict())
    motif_dic.setdefault('five',dict())
    line_num = 0

    for edge in (list(G.edges())):
        t1 = time.time()
        H = G.copy()
        H.remove_edge(edge[0],edge[1])
        try:        
            short_list = [len(p)-2 for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])]
            result = Counter(short_list)
            if min(short_list) == 1:
                try: 
                    motif_dic['triangle'][str(result[min(short_list)])] += 1
                except:
                    motif_dic['triangle'].setdefault(str(result[min(short_list)]), 1)
            elif min(short_list) == 2:
                try: 
                    motif_dic['square'][str(result[min(short_list)])] += 1
                except:
                    motif_dic['square'].setdefault(str(result[min(short_list)]), 1)
            elif min(short_list) == 3:
                try: 
                    motif_dic['five'][str(result[min(short_list)])] += 1
                except:
                    motif_dic['five'].setdefault(str(result[min(short_list)]), 1)
        except:
            line_num += 1
        
        t2 = time.time()
        print('time',t2- t1)
        
    save_dict(motif_dic, 'motif_two_'+str(pic_chose))
    motif_two = load_dict('motif_two_'+str(pic_chose))
  
