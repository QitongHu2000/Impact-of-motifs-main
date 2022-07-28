# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:54:24 2021

@author: baoxiaoge
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import warnings
import pickle
warnings.filterwarnings('ignore')

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
a=1.0
b=0.2
eta = 0.3

J_i_list = np.load('edge_J_i_list_'+str(int(100*a))+'_'+str(int(100*b))+'.npy')
Eim_list = np.load('edge_Eim_list_'+str(int(100*a))+'_'+str(int(100*b))+'.npy')

final_times = load_dict('edge_P_'+str(int(100*a))+'_'+str(int(100*b)))
weights = load_dict('edge_P_weight_'+str(int(100*a))+'_'+str(int(100*b)))
#our_time = np.power(weights, 1.0/a-1.0-b/a)*100
our_time = (- J_i_list * np.log(1 - eta) ) / (1 + (eta * Eim_list) / ( (1 - eta) * np.log(1 - eta) ))
print(our_time)

colors = ['#EECB8E', '#DC8910','#83272E']
fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)

times=(final_times[:,0]).T.tolist()[0]
ax.scatter(weights,times,s=250,marker = 's',c=colors[0],label = 'Simulation',alpha =0.8)
ax.loglog(weights,times,c=colors[0],linewidth=4,linestyle='-',alpha = 0.7)

times=np.array((final_times[:,2]).T.tolist()[0])
ax.scatter(weights,times,s=200,c=colors[1],label = r'$d_i^{\theta}$')
ax.loglog(weights,times,c=colors[1],linewidth=4,linestyle='-',alpha = 1.0)
    
ax.scatter(weights,our_time,s=100,c=colors[2],label = 'Theory',alpha = 0.8)
ax.loglog(weights,our_time,c=colors[2],linewidth=4,linestyle='-',alpha = 0.7)

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xscale('log')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlabel(r"$d_i$",fontsize=35)
plt.ylabel(r"$\tau_{im},\tau_{i}$",fontsize=35)
#plt.legend(['simulation','scaling_hens','scaling_ours'],fontsize=15)
plt.tight_layout()
plt.savefig('edge_P_'+str(int(100*a))+'_'+str(int(100*b))+'.pdf',dpi=300)
plt.show()