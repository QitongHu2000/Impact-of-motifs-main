import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from bs4 import BeautifulSoup
from scipy.integrate import odeint
import warnings
warnings.filterwarnings("ignore")
import xlwt
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import MaxNLocator
import os
import gc
import bezier
import time
import pickle
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
fpath=os.path.join('SimHei.ttf')
prop=fm.FontProperties(fname=fpath)

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

tri_def = 'select'
time_dict = load_dict('time_dict_' + tri_def)
times_1 = np.load('times_1.npy')
times_2 = np.load('times_2.npy')
times_3 = np.load('times_3.npy')

pair_keys = list(time_dict.keys())
layer_list = [a[0] for a in pair_keys]
tri_list = [a[1] for a in pair_keys]
time_list = []

mean_matrix = np.zeros((max(layer_list)+1, max(tri_list)+1))
for row in range((max(layer_list)+1)):
    for col in range((max(tri_list)+1)):
        try:
            mean_matrix[row][col] = np.mean(time_dict[(row, col)])
            time_list.append(np.mean(time_dict[(row, col)]))
            #print('layer',row, 'tri',col, 'time', np.mean(time_dict[(row,col)]))
        except:
            mean_matrix[row][col] = 0 

mean_matrix = mean_matrix.T
mean_matrix_select = mean_matrix[2:7, 2:5]
print(mean_matrix_select)

fig,ax=plt.subplots(figsize=(6,6))
sns.set(style="white")
#sns.set(font_scale=2)

bounds = [2.9, 4, 4.7, 5.6, 6.3, 7.2]
nodes = [(v-min(bounds))/(max(bounds)-min(bounds)) for v in bounds]
#colors = [ '#34A9E1','#22499D', '#58F400','#00942E', '#F2EB2D','#EC6619','#E7211A']
colors = [ '#C0D5E0','#65A3C8', '#65A3C8','#1E4981', '#1E4981','#28384E','#E7211A']
#colors = ['#DDEAFF','#A2C6E7','orange','blue','yellow','pink','purple']
cmap = LinearSegmentedColormap.from_list('mymap', list(zip(nodes, colors)))

sns.heatmap(mean_matrix_select,mask=mean_matrix_select == 0, cmap=cmap,vmax=max(bounds),vmin = min(bounds),square=True, linewidths=1,
            cbar_kws={"shrink":.5},xticklabels=4,yticklabels=4,ax=ax)

ax.set_xlabel('Layer',fontsize=30)
ax.set_ylabel('Number of triangles',fontsize=30)
plt.yticks(np.array([0,1,2,3,4])+0.5, [2,3,4,5,6], va= 'center',rotation = 90)
ax.set_xticks(np.array([0,1,2])+0.5)
ax.set_xticklabels([2,3,4], ha='center')
ax.invert_yaxis()
plt.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig('matrix.pdf',dpi=300)
plt.show()


for name in ['PPI22','PPI2','PPI21']:
    A=loadmat('matlab/'+name+'.mat')['A']
    G=nx.from_numpy_matrix(A)
    print('triangles:', sum(nx.triangles(G).values()) / 3.0)
    print('transitivity:', nx.transitivity(G))
    print('clustering:', nx.average_clustering(G))

times_1[np.isnan(times_1)]=0
times_2[np.isnan(times_2)]=0
times_3[np.isnan(times_3)]=0
times_1[np.isinf(times_1)]=0
times_2[np.isinf(times_2)]=0
times_3[np.isinf(times_3)]=0
times_1_list = list(times_1[0])
times_2_list = list(times_2[0])
times_3_list = list(times_3[0])

data= np.hstack((times_3_list, times_2_list, times_1_list)).reshape(3*len(times_1_list),1)
data_time = pd.DataFrame(data, columns = ['T'])
data_time.insert(data_time.shape[1], 'N', 0)

for i in range(len(times_1_list)):
    data_time['N'][i] = '0.018'
    data_time['N'][len(times_1_list)+i] = '0.034'
    data_time['N'][2*len(times_1_list)+i] = '0.050'

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)
#sns.set_theme(style="whitegrid")
sns.set(style="white")
ax = sns.violinplot(x="N", y="T", data= data_time, palette=['#629BBD','#629BBD','#629BBD'])
plt.tick_params(labelsize=25)
plt.xlabel("Clustering coefficient",fontsize=30)
plt.ylabel("Propagation time", fontsize=30)
plt.tight_layout()
plt.savefig('violinplot.pdf',dpi=300,bbox_inches='tight')
plt.show()


