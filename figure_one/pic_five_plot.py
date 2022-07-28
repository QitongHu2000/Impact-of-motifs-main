# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:25:15 2021

@author: baoxiaoge
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
final_dict = load_dict('pic_five')
print(final_dict)

fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
x_data = list(final_dict.keys())
x_data = [int(i) for i in x_data]
interval = np.arange(0.5,6.5,1)
print(interval)
pick_avg= []
pick_var= []

for j in final_dict:
    pick_avg.append(np.mean(np.array(final_dict[j])))
    pick_var.append(np.std(np.array(final_dict[j])))

error_params=dict(elinewidth=4,ecolor='#113d96',capsize=5)
plt.bar(np.arange(1,6),pick_avg,color='#99c8e6',yerr=pick_var,error_kw=error_params)
plt.ylim(0.4,0.85)

ax.tick_params(axis='both',which='both',direction='in',width=4,length=10)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(range(1,6),fontsize=25)
plt.yticks(fontsize=25)

#plt.xlabel("Number of triangels",fontsize=30)
plt.ylabel(r"$\tau_{im}$",fontsize=30)
#plt.legend(['simulation','scaling_hens','scaling_ours','std'],fontsize=15,bbox_to_anchor=(1.1,1.1,0,0))
plt.tight_layout()
plt.savefig('pic_five.pdf',dpi=300)
plt.show()

