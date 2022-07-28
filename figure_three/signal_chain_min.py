# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:05:57 2020

@author: baoxiaoge
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import odeint
from random import shuffle
from matplotlib.ticker import MaxNLocator,FuncFormatter
import math
from sklearn.linear_model import LinearRegression
gamma=0.3
t_0=1000
n=50000

def sim(degrees):
    #n=7
    def sim_network(N,degrees):
        G=nx.Graph()
        k=1
        for i in range(N):
            G.add_edge(i,i+1)
            k+=1
        for i in range(1,N):
            for j in range(degrees[i-1]):
                G.add_edge(i,k)
                k+=1
        for i in range(20):
            G.add_edge(0,k)
            k+=1
        return G,nx.to_numpy_matrix(G)

    def F(x):
        # return np.mat(-np.power(x,a)+A*np.power(x,b))
        return np.mat(-np.power(x,a)+A*(np.power(x,b)/(1+np.power(x,b))))
    
    def Fun(x,t):
        x=np.mat(x).T
        dx=F(x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t):
        x=np.mat(x).T
        dx=F(x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[0]=0
        return dx
    
    def sim_first():
        x_0=np.ones(np.shape(A)[0])
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun,x_0,t)
        x=xs[np.shape(xs)[0]-1,:].tolist()
        return x
    
    def sim_second(x):
        x[0]*=(1+gamma)
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun_1,x,t)
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
    G,A=sim_network(N,degrees)
    x_1=sim_first()
    xs=sim_second(x_1.copy())
    times=time(xs.copy(),eta).tolist()[0]
    times[0]=0
    return x_1,G,times

def plotting(total_std,total_times,path):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    plt.scatter(total_std,total_times)
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',width=2,length=10)
    bwith=4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.xlabel("std",fontsize=35)
    plt.ylabel("time",fontsize=35)
    # plt.legend(['N='+str(N_1),'N='+str(N_2)],fontsize=20,bbox_to_anchor=(0,0,0,0))
    plt.tight_layout()
    plt.savefig(path,dpi=300)
    plt.show()

def theta_J(a):
    return (1.0/a-1.0)

if __name__=='__main__':
    a=1.2
    b=2.0
    eta=0.3
    total_times=[]
    theory_times= []
    N=7
    degree_list = range(3,28)
    for k in degree_list:
        degrees=[0,0,0,k,0,0,0]
        x_1,G,times=sim(degrees)
        total_times.append(times[N])
        theory_times.append(math.pow(k,theta_J(a)))
        #print(times[N],math.pow(k,theta_J(a)))
    #plt.plot(degree_list,total_times,c="black",linewidth =4)
    #plt.plot(degree_list,np.power(degree_list,theta_J(a)),c="red",linewidth =4)
    fig=plt.figure(figsize=(6,6))
    ax=plt.gca()
    ax.tick_params(axis='both',which='both',direction='in',width=2,length=10)
    bwith=4
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter(theory_times,total_times,alpha=0.8,s = 200, color= '#E3C74C')
    plt.xlabel(r"min$(d^{\theta})$",fontsize=30)
    plt.ylabel("Propagation time",fontsize=30)
    plt.xticks([0.6,0.7,0.8],fontsize=25)
    plt.yticks(fontsize=25)
    model = LinearRegression()
    model = model.fit(np.array(theory_times).reshape(-1, 1), np.array(total_times).reshape(-1, 1))
    y_pred = model.predict(np.array(theory_times).reshape(-1, 1))
    plt.plot(np.array(theory_times).reshape(-1, 1),y_pred,linewidth = 4,c="#B63825", alpha = 0.8)
    plt.tight_layout()
    fun = 'R'
    path=fun+'_chain_'+str(int(a*10))+'_'+str(int(b*10))+'.pdf'
    plt.savefig(path,dpi=300)
    plt.show()