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

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

G=nx.read_gpickle('G_three.gpl')
G.add_edge(99,100)
nx.write_gpickle(G,'G_three.gpl')
line_num = 0
find_triangle = dict()
find_line = dict()
find_direct = dict()

for edge in (list(G.edges())):
    t1 = time.time()
    H = G.copy()
    H.remove_edge(edge[0],edge[1])
    try:        
        short_list = [len(p)-2 for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])]
        result = Counter(short_list)
        if min(short_list) == 1:
            if (result[min(short_list)]) == 5: 
                print(edge[0],edge[1], [p[1] for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])])
                try:
                    find_triangle['m'].append(edge[0])
                except:
                    find_triangle.setdefault('m',[])
                    find_triangle['m'].append(edge[0])
                try:
                    find_triangle['i'].append(edge[1])
                except:
                    find_triangle.setdefault('i',[])
                    find_triangle['i'].append(edge[1])
                try:
                    find_triangle['j'].append([p[1] for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])])
                except:
                    find_triangle.setdefault('j',[])
                    find_triangle['j'].append([p[1] for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])])
            
            elif (result[min(short_list)]) == 1: 
                try:
                    find_line['m'].append(edge[0])
                except:
                    find_line.setdefault('m',[])
                    find_line['m'].append(edge[0])
                try:
                    find_line['i'].append(edge[1])
                except:
                    find_line.setdefault('i',[])
                    find_line['i'].append(edge[1])
                try:
                    find_line['j'].append([p[1] for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])])
                except:
                    find_line.setdefault('j',[])
                    find_line['j'].append([p[1] for p in nx.all_shortest_paths(H,source=edge[0],target=edge[1])])
    except:
        find_direct.setdefault('m',edge[0])
        find_direct.setdefault('i',edge[1])
    
    t2 = time.time()
    #print('time',t2- t1)
    
save_dict(find_triangle, 'find_tri')
find_triangle = load_dict('find_tri')
print(find_triangle)
save_dict(find_line, 'find_line')
find_line = load_dict('find_line')
print(find_line)
save_dict(find_direct, 'find_direct')
find_direct = load_dict('find_direct')
print(find_direct)

