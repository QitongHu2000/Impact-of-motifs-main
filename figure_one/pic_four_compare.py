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

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
fpath=os.path.join('C:/Windows/Fonts/SimHei.ttf')
prop=fm.FontProperties(fname=fpath)
eta = 0.3
gamma=0.1
t_0=150#1500
n=1000000#500000

find_triangle = load_dict('find_tri')
print(find_triangle)
find_line = load_dict('find_line')
print(find_line)
find_direct = load_dict('find_direct')
print(find_direct)
line_try = 1
def sim_network():
    H = nx.read_gpickle('G_three.gpl')
    A=nx.to_numpy_matrix(H)
    return A

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

t_flag=0
flag=True

def sim_first():
    x_0=np.ones(np.shape(A)[0])
    t=np.linspace(0,t_0,n)
    xs=odeint(Fun,x_0,t)
    x=xs[np.shape(xs)[0]-1,:].tolist()
    return x

def Fun_1(x,t):
    m=find_triangle['m'][0]
    
    x=np.mat(x).T
    dx=F(x).tolist()
    dx=[dx[i][0] for i in range(len(dx))]
    
    dx[m]=0
    return dx

def sim_second(x):
    m=find_triangle['m'][0]

    x[m]*=(1+gamma)
    t=np.linspace(0,t_0,n)
    xs=odeint(Fun_1,x,t)
    return np.mat(xs)


def Fun_2(x,t):
    m=find_line['m'][line_try]
    
    x=np.mat(x).T
    dx=F(x).tolist()
    dx=[dx[i][0] for i in range(len(dx))]
    
    dx[m]=0
    return dx

def sim_second_2(x):
    m=find_line['m'][line_try]

    x[m]*=(1+gamma)
    t=np.linspace(0,t_0,n)
    xs=odeint(Fun_2,x,t)
    return np.mat(xs)

def Fun_3(x,t):
    m=find_direct['m']
    
    x=np.mat(x).T
    dx=F(x).tolist()
    dx=[dx[i][0] for i in range(len(dx))]
    
    dx[m]=0
    return dx

def sim_second_3(x):
    m=find_direct['m']

    x[m]*=(1+gamma)
    t=np.linspace(0,t_0,n)
    xs=odeint(Fun_3,x,t)
    return np.mat(xs)

def time_func(xs,eta):
    xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
    indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
    times=[]
    for i in range(len(indexs)):
        len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
        len_2=eta-xs[indexs[i],i]
        times.append(indexs[i]+len_2/len_1)
    return np.mat(times)*t_0/n

def plotting_line(xs,xs_new,xs_direct):
    xs=(xs-xs[0,:])/(xs[np.shape(xs)[0]-1,:]-xs[0,:])
    xs_new=(xs_new-xs_new[0,:])/(xs_new[np.shape(xs_new)[0]-1,:]-xs_new[0,:])
    xs_direct=(xs_direct-xs_direct[0,:])/(xs_direct[np.shape(xs_direct)[0]-1,:]-xs_direct[0,:])
    #xs = (xs-xs[0])/(xs[len(xs)-1]-xs[0])
    #xs_new = (xs_new-xs_new[0])/(xs_new[len(xs_new)-1]-xs_new[0])
     
    i_tri = find_triangle['i'][0]
    i_line = find_line['i'][line_try]
    i_direct = find_direct['i']

    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)

    t=[i*t_0/n for i in range(int(n/4))]
    xs_plot = xs[0:len(t),i_tri]
    xs_new_plot = xs_new[0:len(t),i_line]
    xs_direct_plot = xs_direct[0:len(t),i_direct]
    #ax.plot(t[0:-1:3000],xs_plot[0:-1:3000],marker ='$\Delta$', markersize=5,color = '#547bab',linewidth=0,alpha=0.7,label = 'triangle')
    #ax.plot(t[0:-1:3000],xs_new_plot[0:-1:3000],marker ='$\Delta$', markersize=5,color = '#97ccef',linewidth=0,alpha=0.7, label = 'line')
    #ax.plot(t[0:-1:3000],xs_plot[0:-1:3000],marker ='$\Delta$', markersize=5,color = '#547bab',linewidth=0,alpha=0.7,label = 'triangle')
    ax.plot(t[0:-1:4000],xs_new_plot[0:-1:4000],marker ='$\Delta$', markersize=8,color = '#547bab',linewidth=0,alpha=0.7, label = 'line')
    ax.plot(t[0:-1:4000],xs_direct_plot[0:-1:4000],'--', markersize=5,color = '#9C6058',linewidth=4,alpha=1, label = 'line')
    
    #ax.plot(t,xs[0:len(t),i_tri],marker ='$\Delta$',color = '#547bab',linewidth=1,alpha=0.8)
    #ax.plot(t,xs_new[0:len(t),i_line],marker ='$\Delta$',color = '#97ccef',linewidth=1,alpha=0.8)
    ax.tick_params(axis='both',which='both',direction='in',width=3,length=10)

    ax=plt.gca()
    bwith=1
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.ylabel(r'$\frac{\Delta {x_i}(t)}{\Delta {x_i}(\infty)}$',fontsize = 30)
    plt.xlabel(r'$t$',fontsize = 30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tight_layout()
    plt.savefig('pic_four.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    
if __name__=='__main__':
    A=sim_network()
    x_1=sim_first()
    xs=sim_second(x_1.copy())
    xs_new = sim_second_2(x_1.copy())
    xs_direct = sim_second_3(x_1.copy())
    plotting_line(xs,xs_new,xs_direct)
   
    