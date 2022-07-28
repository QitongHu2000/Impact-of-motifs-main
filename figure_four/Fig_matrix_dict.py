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

fpath=os.path.join('SimHei.ttf')
prop=fm.FontProperties(fname=fpath)

gamma=0.3
eta=0.3
t_0= 500#3000 #500
n= 100000#300000 #100000

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_graph_info(name):
    A=loadmat(name)['A']
    G=nx.from_numpy_matrix(A)
    S=dict(G.degree)
    A=sparse.coo_matrix(A)
    print('load information done!')
    return A,G,S

def simulation(A,active_node,fun):
    def F(A,x):
        if(fun=='R'):
            return np.mat(-1*np.power(x,0.8)+A*(np.power(x,0.5)/(1+np.power(x,0.5))))
            #return np.mat(-0.01*np.power(x,0.8)+0.01*A*(np.power(x,0.5)/(1+np.power(x,0.5))))
        if(fun=='P'):
            return np.mat(-1*np.power(x,1)+A*np.power(x,0.2))
        if(fun=='E'):
            return np.mat(-np.power(x,1)+np.multiply(1-x,A*x))
        if(fun=='N'):
            return np.mat(-3*x+3*np.tanh(x)+0.1*A*np.tanh(x))
    
    def Fun(x,t,A):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t,A,active_node):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[active_node]=0
        return dx
    
    def sim_first(A):
        x_0=np.random.rand(np.shape(A)[0])*0.1
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun,x_0,t,args=(A,))
        x=xs[np.shape(xs)[0]-1,:].tolist()
        return x
    
    def sim_second(A,x,source):
        x[source]*=(1+gamma)
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun_1,x,t,args=(A,source))
        return np.mat(xs)
    
    def time(xs,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(indexs[i]+len_2/len_1)
        times=np.mat(times)*t_0/n
        return times
    
    x_1=sim_first(A)
    xs=sim_second(A,x_1.copy(),active_node)
    times=time(xs.copy(),eta)
    times[0,active_node]=0
 
    return xs,times

def cal_mat(A): # calculate number of triangles between i and j
    C=np.zeros(shape=np.shape(A))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if(A[i,j]>0):
                set_1=np.where(A[:,i]>0)[0].tolist()
                set_2=np.where(A[:,j]>0)[0].tolist()
                C[i,j]=len(list(set(set_1).intersection(set(set_2))))
    print('calculate triangle number done!')
    return C


def time_dict_func(A,G,active_node,times):
    C=cal_mat(np.mat(A.toarray()))

    time_dict = dict()  # save dict: key is (layer, tri_num), value is time list
    time_dict_select = dict()  # save dict: key is (layer, tri_num > 0.5* degree_num), value is time list
    select_percent = 0.02

    for i in G.nodes:
        print('node:',i)
        paths=[j for j in nx.all_shortest_paths(G,active_node,i)]

        tri_list = []
        degree_list = []
        for path_index in range(len(paths)): # different shortest path
            tri_num = 0
            degree_num = G.degree[paths[path_index][0]]
            for k in range(1, len(paths[path_index])):
                tri_num += C[paths[path_index][k-1], paths[path_index][k]]  # sum tri_num between each layer along the shortest path
                degree_num += G.degree[paths[path_index][k]]
            tri_list.append(tri_num)
            degree_list.append(degree_num)

        tri_num_insert = int(sum(tri_list) / len(paths)) #int(min(tri_list))  #int(max(tri_list)) #int(sum(tri_list) / len(paths)) #int(max(tri_list))  
        degree_num_insert = int(sum(degree_list) / len(paths)) #int(min(degree_list)) #int(max(degree_list)) #int(sum(degree_list) / len(paths))

        if tri_num_insert >= degree_num_insert * select_percent:
            try:
                time_dict[(len(paths[0])-1, tri_num_insert)].append(times[0][i])
            except:
                time_dict.setdefault((len(paths[0])-1, tri_num_insert), [])
                time_dict[(len(paths[0])-1, tri_num_insert)].append(times[0][i])


    #for pair_key in time_dict:
    #    time_dict[pair_key] = np.mean(time_dict[pair_key]) # value is mean time

    save_dict(time_dict, 'time_dict_select')
    #print(time_dict_select)

if __name__=='__main__':
    model='PPI2'
    fun='R'
    active_node=1
    A,G,S=load_graph_info('matlab/'+model+'.mat')
    xs,times=simulation(A,active_node,fun)
    np.save('time_all.npy', np.array(times))
    times = np.load('time_all.npy')
    time_dict_func(A, G, active_node,times)
    time_dict = load_dict('time_dict_select')
    print(time_dict)

   