# 1.eta 
# 2.degree Is big

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import warnings,time
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')

def create_network_orginal():
    G=nx.Graph()
    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(0,2)
    G.add_edge(1,3)
    A=nx.to_numpy_matrix(G)
    return A

def simulation_unperturb(A,F,source,params):
    gamma,eta,epsilon,tau,mu=params
    
    def Fun_unperturb(t,x):
        dx=F(np.mat(x).T,A).T.tolist()[0]
        return dx

    def cal_scale(xs):
        # print(xs)
        x=xs[:,np.shape(xs)[1]-1]
        dx=Fun_unperturb(0,x)
        return np.max(np.abs(dx/x)),x
        
    def sim_first():
        x=np.random.rand(np.shape(A)[0])
        t=0
        scale=1
        while(scale>epsilon):
            t_eval=np.linspace(t,t+tau,mu)
            sol=solve_ivp(Fun_unperturb,[t,t+tau],x.copy(),t_eval=t_eval,
                          rtol=1e-13,atol=1e-13,method='RK45')
            scale,x=cal_scale(sol.y)
            t+=tau
        return x
    
    x=sim_first()
    return x

def simulation_perturb(A,F,init_x,source,params):
    gamma,eta,epsilon,tau,mu=params

    def Fun_perturb(t,x):
        dx=F(np.mat(x).T,A).T.tolist()[0]
        dx[source]=0
        return dx

    def cal_scale(xs,state):
        x=xs[:,np.shape(xs)[1]-1]
        dx=Fun_perturb(0,x)
        return np.max(np.abs(dx/x)),x
    
    def sim_second(x):
        x[source]*=(1+gamma)
        t=0
        scale=1
        ts=[]
        xs=[]
        while(scale>epsilon):
            t_eval=np.linspace(t,t+tau,mu)
            sol=solve_ivp(Fun_perturb,[t,t+tau],x.copy(),t_eval=t_eval,
                          rtol=1e-13,atol=1e-13,method='RK45')
            scale,x=cal_scale(sol.y,1)
            xs.append(sol.y)
            ts.extend(sol.t)
            t+=tau
        xs=np.hstack(xs)
        ts=np.hstack(ts)
        return ts,xs
        
    def time(ts,ys):
        ys=np.mat(ys)
        scale=(ys-ys[:,0])/(ys[:,np.shape(ys)[1]-1]-ys[:,0])
        scale=np.array(scale)
        return scale

    ts,xs=sim_second(init_x.copy())
    scale=time(ts,xs)
    return ts,scale
    
def plotting(ts,scale,path):
    #colors=['#B98767','#5176A2','#3D8A39']
    colors=['#898989','#5176A2','#9C6058']
    
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    for i in range(1,4):
        ax.plot(ts[range(0,100,2)],scale[i,range(0,100,2)],'v',c=colors[i-1],label='v',markersize=10,alpha=0.95)
    ax.tick_params(axis='both',which='both',direction='in',width=4,length=10)
    bwith=1
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.xlabel(r"$t$",fontsize=30)
    plt.ylabel(r'$\frac{\Delta x(t)}{\Delta x(\infty)}$',fontsize=30)
    # plt.legend(['i','h','j'],fontsize=20,bbox_to_anchor=(1,1,0,0))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tight_layout()
    plt.savefig(path,dpi=300)
    plt.show()

if __name__=='__main__':
    beg_time=time.time()
    def Fun_R(x,A):
        B = 0.01
        alpha = 0.01
        a=1.2
        b=1.1
        return np.mat(-B*np.power(x,a)+alpha*A*np.power(x,b))

    gamma=0.3
    eta=0.3
    epsilon=1e-10
    tau=3000
    mu=1000
    params=[gamma,eta,epsilon,tau,mu]
    
    source=0
    A_orginal=create_network_orginal()
    x_orginal=simulation_unperturb(A_orginal,Fun_R,source,params)
    ts_orginal,scale_orginal=simulation_perturb(A_orginal,Fun_R,x_orginal,source,params)
    path='pic_six.pdf'
    plotting(ts_orginal,scale_orginal,path)