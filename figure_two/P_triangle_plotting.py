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
a=1.2
b=0.5#0.8#0.4#0.8#1.0#0.6
eta = 0.3

J_i_list = np.load('tri_J_i_list_'+str(int(100*a))+'_'+str(int(100*b))+'.npy')
f_list = np.load('tri_f_list_'+str(int(100*a))+'_'+str(int(100*b))+'.npy')
Edelta_list = np.load('tri_Edelta_list_'+str(int(100*a))+'_'+str(int(100*b))+'.npy')


final_times = load_dict('tri_P_'+str(int(100*a))+'_'+str(int(100*b)))
weights = load_dict('tri_P_weight_'+str(int(100*a))+'_'+str(int(100*b)))
#our_time = np.power(weights, 1.0/a)/50
our_time = -np.log(1 - eta) * J_i_list * (1 + Edelta_list)/(1 + (1 - f_list) * Edelta_list)
print('Eq 12:',our_time)
print('Ji:', J_i_list)
print('degree:', weights)
print('f:',f_list)
print('Eim:',Edelta_list)
print('our time:', our_time)
#colors = ['#5681B4', '#72A559','#AC685F']
colors = ['#6A468F', '#004C65','#92A8D7']

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)

times=(final_times[:,0]).T.tolist()[0]
ax.scatter(weights,times,s=300,marker = '^',c=colors[0],label = 'Simulation',alpha = 0.8)
ax.loglog(weights,times,c=colors[0],linewidth=4,linestyle='-',alpha = 0.7)

ax.scatter(weights,our_time,s=300,c=colors[2],marker = '^',label = 'Theory',alpha = 0.8)
ax.loglog(weights,our_time,c=colors[2],linewidth=4,linestyle='-',alpha = 0.7)

times= np.power(weights, 1.0/a -1.0)#np.array((final_times[:,2]).T.tolist()[0])#*5
ax.scatter(weights,times,s=300,marker='^',c=colors[1],label = r'$d_i^{\theta}$')
ax.loglog(weights,times,c=colors[1],linewidth=4,linestyle='-',alpha = 1.0)

#times= -np.log(1 - eta) * J_i_list * (1 / (1- f_list))
#ax.scatter(weights,times,s=300,marker='^',c='gray',label = r'$-ln(1-\eta) J * \frac{1}{1-f}$')
#ax.loglog(weights,times,c='gray',linewidth=4,linestyle='-',alpha = 1.0)

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.legend(fontsize = 15)
plt.xlabel(r"$d_i$",fontsize=35)
plt.ylabel(r"$\tau_{im},\tau_{i}$",fontsize=35)
#plt.legend(['simulation','scaling_hens','scaling_ours'],fontsize=15)
plt.tight_layout()
plt.savefig('tri_P_'+str(int(100*a))+'_'+str(int(100*b))+'.pdf',dpi=300)
plt.show()