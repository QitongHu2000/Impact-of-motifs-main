import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
eta=0.3
t_0=100#1000 #10-1.2-2.0
n= 10000000 #100000000#1000000 #50000000-1.2-2.0
B = 0.01
alpha = 0.01

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def create_network_triangle(weight):
    G=nx.DiGraph()
    G.add_edge(0,1)
    G.add_edge(1,0)
    G.add_edge(1,2,weight=weight)
    G.add_edge(2,1)
    G.add_edge(0,2)
    G.add_edge(2,0)
    A=nx.to_numpy_matrix(G)
    return A

def create_network_edge(weight):
    G=nx.DiGraph()
    G.add_edge(0,1)
    G.add_edge(1,0)
    G.add_edge(1,2,weight=weight)
    G.add_edge(2,1)
    A=nx.to_numpy_matrix(G)
    return A

def simulation(A):
    def F(A,x,a,b):
        return np.mat(-B*np.power(x,a)+alpha*A*(np.power(x,b)/(1+np.power(x,b))))
    
    def Fun(x,t,A,a,b):
        x=np.mat(x).T
        dx=F(A,x,a,b).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t,A,a,b,source):
        x=np.mat(x).T
        dx=F(A,x,a,b).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(a,b,A):
        x_0=np.ones(np.shape(A)[0])
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun,x_0,t,args=(A,a,b))
        x=xs[np.shape(xs)[0]-1,:].tolist()
        return x
    
    def sim_second(a,b,A,x,source):
        x[source]*=(1+gamma)
        t=np.linspace(0,t_0,n)
        xs=odeint(Fun_1,x,t,args=(A,a,b,source))
        return np.mat(xs)
        
    def time(xs,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(indexs[i]+len_2/len_1)
        return np.mat(times)*t_0/n
    x=sim_first(a,b,A)
    xs=sim_second(a,b,A,x.copy(),source)
    times=time(xs.copy(),eta).tolist()[0]
    return times,x,xs

def theory_ours(x,degrees):
    mean_M=np.mean(np.power(x,b)/(1+np.power(x,b)))
    theory_x=np.power(degrees*mean_M/B,1/a).T
    J=1/(a*B)*np.power(theory_x,1-a)
    Q=b/(a*B)*np.power(theory_x,b-a)/np.power(np.power(theory_x,b)+1,2)
    mean_Q=np.mean(Q)
    E_L=np.multiply(Q,degrees.T-1)*mean_Q
    return -np.log(1-eta)*J/(1+(eta/((1-eta)* np.log(1-eta)))*E_L)

def theory_scaling_hens(degrees):
    return np.power(degrees,1/a-1)

def theory_scaling_ours(degrees):
    return np.power(degrees,1/a-1)

def basic_term(x):
    J_i =  np.power(x[1],1-a)/ (B * a)
    J_j =  np.power(x[2],1-a)/ (B * a)
    Q_i = ((alpha * b) / (a*B)) * np.power(x[1],b-a) / (np.power(np.power(x[1],b)+1,2))
    Q_j = ((alpha * b) / (a*B)) * np.power(x[2],b-a) / (np.power(np.power(x[2],b)+1,2))
    return J_i, J_j, Q_i, Q_j

def delta_x_plot(xs, weight):
    delta_x = (xs-xs[0])/(xs[len(xs)-1]-xs[0])
    delta_xi =  delta_x[:,1]
    delta_xj = delta_x[:,2]
    t=np.linspace(0,t_0,n)
    plt.plot(t, delta_xi, label = r'$x_i$')
    plt.plot(t, delta_xj, label = r'$x_j$')
    plt.ylabel(r'$\delta x(t)$')
    plt.xlabel('t')
    plt.title('degree='+str(weight))
    plt.legend()
    plt.show()

def plotting(weights,final_times):
    colors=['#29ABE2','r','g','orange']
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    for i in range(np.shape(final_times)[1]):
        times=(final_times[:,i]).T.tolist()[0]
        ax.scatter(weights,times,s=200,c=colors[i])
        ax.loglog(weights,times,c=colors[i],linewidth=4,linestyle='-')
    ax.tick_params(axis='both',which='both',direction='in',width=4,length=10)
    bwith=4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.xlabel(r"impact",fontsize=35)
    plt.ylabel(r"time",fontsize=35)
    plt.legend(['simulation','theory','scaling_hens','scaling_ours'],fontsize=15,bbox_to_anchor=(1.1,1.1,0,0))
    plt.tight_layout()
    #plt.savefig('R_12_20.pdf',dpi=300)
    plt.show()

source=0
final_times_R=[]
J_i_list = []
f_list = []
Edelta_list = []
weights= np.logspace(3,5,10).astype('int') #[1000, 10000, 100000]  
a=10.0
b=2.0

for weight in weights:
    print(weight)
    A_edge=create_network_triangle(weight)
    degrees=np.sum(A_edge,axis=1)
    times_simulation_edge,x_edge,xs_edge=simulation(A_edge)
    times_simulation_edge=times_simulation_edge[1]
    times_theory_edge=theory_ours(x_edge,degrees)[0,1]
    times_scaling_edge_hens=theory_scaling_hens(degrees)[1,0]
    times_scaling_edge_ours=theory_scaling_ours(degrees)[1,0]
    final_times_R.append([times_simulation_edge,times_theory_edge,
                          times_scaling_edge_hens,times_scaling_edge_ours])
    J_i, J_j, Q_i, Q_j = basic_term(x_edge)
    J_i_list.append(J_i)
    f_list.append((1 - np.exp(- times_simulation_edge / J_j) / (1 - eta)) / (times_simulation_edge / J_j + np.log(1-eta)))
    Edelta_list.append(weight * Q_j)
    #delta_x_plot(xs_edge, weight)
    print('f:',f_list)

np.save('tri_J_i_list_'+str(int(100*a))+'_'+str(int(100*b)), J_i_list)
np.save('tri_f_list_'+str(int(100*a))+'_'+str(int(100*b)), f_list)
np.save('tri_Edelta_list_'+str(int(100*a))+'_'+str(int(100*b)), Edelta_list)

final_times_R=np.mat(final_times_R)
save_dict(final_times_R, 'tri_R_'+str(int(100*a))+'_'+str(int(100*b)))
save_dict(weights, 'tri_R_weight_'+str(int(100*a))+'_'+str(int(100*b)))