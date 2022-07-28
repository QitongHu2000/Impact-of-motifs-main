import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.io import loadmat
import networkx as nx

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

pic_label = [1,2,3]
for p in pic_label:
    motif = load_dict('motif_one_'+str(p))
    data = []
    for keys in motif:
        data.append(motif[keys])
    plt.figure(figsize = (6,3))
    ax=plt.gca()
    bwith=1
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.bar(range(len(data)), np.array(data)/np.sum(np.array(data)), color=['#a14847','#006fae','#3c9a3b','#b4724c'],alpha = 0.8) # or `color=['r', 'g', 'b']`
    #plt.xticks(np.arange(0.5,5.5), ['Line','Triangle','Square','Pentagon'],fontsize = 30,rotation = 315)
    plt.ylim(0,0.81)
    plt.xticks([])
    plt.yticks(np.arange(0,1,0.2),fontsize = 30)
    #plt.ylabel('Number',fontsize = 30)
    #plt.xlabel('Motif',fontsize = 25)
    plt.tight_layout()
    plt.savefig('hist_'+str(int(p*100))+'.pdf',dpi=300)
    plt.show()

seed = 13648 
net_size = []
for pic_chose in pic_label:
    if pic_chose == 1:
        A = loadmat('PPI2')['A']
        G = nx.from_numpy_matrix(A)
        net_size.append(len(G.nodes()))
        print('network size of'+str(pic_chose)+'is', len(G.nodes()))
    elif pic_chose == 2:
        G = nx.generators.connected_watts_strogatz_graph(100, k=10, p = 0.1, tries=100, seed=seed)
        net_size.append(len(G.nodes()))
        print('network size of'+str(pic_chose)+'is', len(G.nodes()))
    else:
        G = nx.generators.random_graphs.erdos_renyi_graph(100, p=0.1, seed=seed, directed=False)
        net_size.append(len(G.nodes()))
        print('network size of'+str(pic_chose)+'is', len(G.nodes()))

colors = ['#006fae','#3c9a3b','#b4724c']
for p in pic_label:
    motif2 = load_dict('motif_two_'+str(p))
    plt.figure(figsize = (6,3))
    x_max = 6
    y_max = 0
    x_data = [str(j) for j in range(1,x_max)]
    x = range(len(x_data))
    m = 0
    for keys in motif2:
        y_data = []
        for j in x_data:
            try:     
                y_data.append(motif2[keys][j])
            except:
                y_data.append(0)
        #plt.plot(x_data, y_data, '-o',color = colors[m],label = keys) # or `color=['r', 'g', 'b']`
        plt.bar(x=[i + 0.2*m for i in x], height=np.array(y_data)/ net_size[int(p-1)], width=0.2, alpha=0.8, color=colors[m])
        m += 1
        if max(y_data) > y_max:
            y_max = max(y_data)
    ax=plt.gca()
    bwith=1
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(x,[i+1 for i in x],fontsize = 30)
    plt.yticks(fontsize = 30)
    #plt.ylabel('Number of Pairs',fontsize = 30)
    #plt.xlabel('Number of motifs',fontsize = 30)
    #plt.legend()
    plt.tight_layout()
    plt.savefig('line_'+str(int(p*100))+'.pdf',dpi=300)
    plt.show()
   
    