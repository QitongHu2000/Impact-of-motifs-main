import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import odeint
import multiprocessing as mp

gamma=0.3
t_0=1000
n=50000
eta=0.3

def sub_calculate(ranges,a,b,m,d,ker):
    def sim(degrees):
        def sim_network(n,degrees):
            G=nx.Graph()
            k=1
            for i in range(n):
                G.add_edge(i,i+1)
                k+=1
            for i in range(1,n):
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
        G,A=sim_network(m,degrees)
        x_1=sim_first()
        xs=sim_second(x_1.copy())
        times=time(xs.copy(),eta).tolist()[0]
        times[0]=0
        return x_1,G,times

    ans_total_times=[]
    ans_total_std=[]
    ans_total_mean=[]
    for N in ranges:
        for k in range(500):
            print(k)
            p=np.random.rand(1,m-1)
            p/=np.sum(p)
            degrees=np.random.multinomial(N*(m-1),p.tolist()[0])
            x_1,G,times=sim(degrees)
            std_=np.std(degrees)
            std_=int(std_/d)
            if(std_<20):
                ans_total_times.append(times[m])
                ans_total_std.append(std_)
                ans_total_mean.append(N)
    np.save('ans/'+str(ker)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_times.npy',ans_total_times)
    np.save('ans/'+str(ker)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_std.npy',ans_total_std)
    np.save('ans/'+str(ker)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_mean.npy',ans_total_mean)

if __name__=='__main__':
    mp.freeze_support()
    a=0.8
    b=0.5
    m=7
    d=1
    kernels=4
    
    total_times=[]
    total_std=[]
    total_mean=[]
    records=[]
    for k in range(kernels):
        ranges=[i*kernels+k for i in range(5)]
        process=mp.Process(target=sub_calculate,args=(ranges,a,b,m,d,k))
        process.start()
        records.append(process)
    for process in records:
        process.join()
    
    for k in range(kernels):
        total_times.append(np.load('ans/'+str(k)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_times.npy'))
        total_std.append(np.load('ans/'+str(k)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_std.npy'))
        total_mean.append(np.load('ans/'+str(k)+'_'+str(int(a*10))+'_'+str(int(b*10))+'_ans_total_mean.npy'))
    total_times=np.hstack(total_times)
    total_std=np.hstack(total_std)
    total_mean=np.hstack(total_mean)


    fun='R'
    np.save(fun+'_time_std_total_times'+str(int(a*10))+'_'+str(int(b*10))+'.npy',total_times)
    np.save(fun+'_time_std_total_std'+str(int(a*10))+'_'+str(int(b*10))+'.npy',total_std)
    np.save(fun+'_time_std_total_mean'+str(int(a*10))+'_'+str(int(b*10))+'.npy',total_mean)