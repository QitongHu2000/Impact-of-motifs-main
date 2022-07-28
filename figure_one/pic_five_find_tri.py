from matplotlib.ticker import MaxNLocator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import odeint
from matplotlib import font_manager as fm, rcParams
import os
import pickle
import time 
from collections import Counter

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
eta = 0.3
gamma=0.1
t_0=1500
n=1000000

def sim_network():
    H = nx.read_gpickle('G_three.gpl')
    A=nx.to_numpy_matrix(H)
    return A, H

def F(x):
    B = 0.01
    alpha = 0.01
    a=1.2
    b=1.1
    return np.mat(-B*np.power(x,a)+alpha*A*np.power(x,b))

def Fun(x,t):
    x=np.mat(x).T
    dx=F(x).tolist()
    dx=[dx[i][0] for i in range(len(dx))]
    return dx

def sim_first(A):
    x_0=np.ones(np.shape(A)[0])
    t=np.linspace(0,t_0,n)
    xs=odeint(Fun,x_0,t)
    x=xs[np.shape(xs)[0]-1,:].tolist()
    return x

def simulation_time(x,m):
    def Fun_1(x,t):
        #print('m',m)
        
        x=np.mat(x).T
        dx=F(x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        
        dx[m]=0
        return dx
    
    def sim_second(x):
        #print('m',m)
    
        x[m]*=(1+gamma)
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun_1,x,t)
        return np.mat(xs)
    
    def time_new(xs,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(indexs[i]+len_2/len_1)
        return np.mat(times)*t_0/n
    
    xs=sim_second(x_1.copy())
    times=time_new(xs.copy(),eta).tolist()[0]
    return times

def find_pairs(H):
    line_num = 0
    motif_dic = dict()
    for edge in (list(H.edges())):
        t1 = time.time()
        HH = H.copy()
        HH.remove_edge(edge[0],edge[1])
        try:        
            short_list = [len(p)-2 for p in nx.all_shortest_paths(HH,source=edge[0],target=edge[1])]
            result = Counter(short_list)
            if min(short_list) == 1:
                try:
                    motif_dic.setdefault(str(result[min(short_list)]),{})[str(edge[0])].append(edge[1])
                except:
                    motif_dic.setdefault(str(result[min(short_list)]),{})[str(edge[0])] = []
                    motif_dic.setdefault(str(result[min(short_list)]),{})[str(edge[0])].append(edge[1])
        except:
            line_num += 1
        
        t2 = time.time()
        #print('time',t2- t1)
    return (motif_dic)

if __name__=='__main__':
    A,H=sim_network()
    x_1=sim_first(A)
    motif = find_pairs(H)
    
    final_time = dict()
    
    for key in motif: 
        for i in motif[key]:
            #t_zero = time.time()
            times_new= simulation_time(x_1, int(i))
            for j in motif[key][i]:
                final_time.setdefault(key,[]).append(times_new[int(j)])
                print(i,j,times_new[int(j)])
    save_dict(final_time,'pic_five')
    final_time = load_dict('pic_five')
    print(final_time)
   
