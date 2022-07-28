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

fpath=os.path.join('SimHei.ttf')
prop=fm.FontProperties(fname=fpath)

gamma=0.3
eta=0.3
t_0= 500 #1000
n= 100000 #250000

def load_graph_info(name):
    A=loadmat(name)['A']
    G=nx.from_numpy_matrix(A)
    S=dict(G.degree)
    A=sparse.coo_matrix(A)
    return A,G,S

def plotting_cal(S,positions,path,edges,D,D_numbers):
    positions_new={}
    edges=np.array(edges)
    nodes=[]
    nodes_pan=[]
    Ss=[]
    Cs=[]
    rs=[]
    edges_dict={}
    for i in list(S.keys()):
        Cs.append(S[i])
        r,theta=positions[i]
        nodes_pan.append([r*np.cos(theta),r*np.sin(theta)])
        scale=D[0,i]/D_numbers[r]
        a=1
        c=1/((1-a)**3+a**3)
        b=a**3*c
        scale=c*(scale-a)**3+b
        if(scale>1):
            print(scale)
        r=10*((r-1)+scale*0.9)**1.5
        positions_new[i]=[r,theta]
        nodes.append([r*np.cos(theta),r*np.sin(theta)])
        if(S[i]==1):
            Ss.append('red')
            rs.append(r)
        else:
            #Ss.append('green')
            Ss.append('gray')
            rs.append(r)
        to_nodes=edges[edges[:,0]==i,1]
        if(len(to_nodes)>0):
            edges_dict[i]=to_nodes
    nodes_pan=np.array(nodes_pan)
    nodes=np.array(nodes)
    rs=np.array([[i,0] for i in rs])
    Ss=np.array(Ss)
    return nodes,nodes_pan,Ss,Cs,rs,edges_dict

def plotting_beautiful(locations,nodes,nodes_pan,rs,Cs,edges_dict,ax):
    cluster=LogisticRegression().fit(rs,Cs)
    k=cluster.coef_[0,0]
    b=cluster.intercept_[0]
    max_r=int(np.max(list(locations.keys())))
    avil_nodes=list(edges_dict.keys()).copy()
    for i in np.hstack((locations.values())):
        if (i not in avil_nodes):
            continue
        cluster=np.array(avil_nodes)[np.linalg.norm(nodes_pan[avil_nodes]-nodes_pan[i],axis=1)<np.linalg.norm(nodes_pan[i])*0.5]#1.5]
        cluster=cluster[np.array(rs)[cluster,0]==rs[i][0]]
        degree=10
        x_z=np.mean([nodes[j][0] for j in cluster])
        y_z=np.mean([nodes[j][1] for j in cluster])
        x_w=np.mean([nodes[i][0] for j in cluster for i in edges_dict[j]])
        y_w=np.mean([nodes[i][1] for j in cluster for i in edges_dict[j]])
        nodes_x=np.linspace(x_z,x_w,degree+1)
        nodes_y=np.linspace(y_z,y_w,degree+1)
        nodes_x=nodes_x[1:len(nodes_x)-1]
        nodes_y=nodes_y[1:len(nodes_y)-1]
        for u in cluster:
            for v in edges_dict[u]:
                if(u in locations[7] or v in locations[7]):
                    continue
                ans_nodes=[np.hstack((nodes[u][0],nodes_x,nodes[v][0])).tolist(),np.hstack((nodes[u][1],nodes_y,nodes[v][1])).tolist()]
                ans_nodes=np.asfortranarray(ans_nodes)
                curve=bezier.Curve(ans_nodes,degree=degree)
                s_vals=np.linspace(0,1,30)
                data=curve.evaluate_multi(s_vals)
                if(v in locations[6]):
                    ax.plot(data[0],data[1],c='black',alpha=0.1,linewidth=1)
                else:
                    ax.plot(data[0],data[1],c='black',alpha=0.1,linewidth=0.2)
        for i in cluster:
            avil_nodes.remove(i)
    #wed_1=Wedge(center=(0,0),r=b/k,theta1=0,theta2=360,color='orange',edgecolor=None,alpha=0.3)
    #ax.add_patch(wed_1)
    for r in range(1,max_r-1):
        wed_2=Wedge(center=(0,0),r=10*r**1.5,theta1=0,theta2=360,fill=False,color='black',linestyle='--',linewidth=2,alpha=0.4)
        ax.add_patch(wed_2)
    wed_3=Wedge(center=(0,0),r=10*6**1.5,theta1=0,theta2=360,fill=False,color='black',linestyle='--',linewidth=0,alpha=0)
    ax.add_patch(wed_3)
    return ax

def cycle_plotting(xs,active_node,t,positions,locations,path,edges,D,D_numbers):
    delta_x=np.mat(delta(xs.copy(),active_node)[t,:])

    #plt.plot(delta_x)
    #plt.title(str(t))
    #plt.show()

    S={}
    for i in range(np.shape(delta_x)[1]):
        if(np.isnan(delta_x[0,i])):
            continue
        if(delta_x[0,i]>eta):
            S[i]=1#100*np.power(delta_x[0,i],0.8)
        else:
            S[i]=0
    nodes,nodes_pan,Ss,Cs,rs,edges_dict=plotting_cal(S,positions,path,edges,D,D_numbers)
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111)
    ax=plotting_beautiful(locations,nodes,nodes_pan,rs,Cs,edges_dict,ax)
    # ax.scatter(nodes[:,0],nodes[:,1],c=Ss,s=15,alpha=0.2)
    ax.scatter(nodes[rs[:,0]<10*6**1.5,0],nodes[rs[:,0]<10*6**1.5,1],c=Ss[rs[:,0]<10*6**1.5],s=15,alpha=0.2)

    ax.scatter(0,0,c='black',s=100)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(path+'.jpg',dpi=300)
    del nodes,nodes_pan,Ss,Cs,rs,edges_dict
    gc.collect()
    plt.show()

def simulation(A,active_node,fun):
    def F(A,x):
        if(fun=='R'):
            return np.mat(-1*np.power(x,0.8)+A*(np.power(x,0.5)/(1+np.power(x,0.5))))
            #return np.mat(-0.01*np.power(x,0.8)+ 0.01*A*(np.power(x,0.5)/(1+np.power(x,0.5))))
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
    print(times)
    return xs,times

def delta(xs,active_node):
    xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
    xs[np.isnan(xs)]=1
    return xs
    
def sim(name,active_node,fun):
    A,G,S=load_graph_info('matlab/'+name+'.mat')
    xs,times=simulation(A,active_node,fun)
    return A,G,S,xs,times

def average_times(times_1,times_2,times_3,G_1,G_2,G_3,path):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    times_1[np.isnan(times_1)]=0
    times_2[np.isnan(times_2)]=0
    times_3[np.isnan(times_3)]=0
    times_1[np.isinf(times_1)]=0
    times_2[np.isinf(times_2)]=0
    times_3[np.isinf(times_3)]=0
    ax.bar(1,np.mean(times_3),color='blue',width=0.2)
    ax.bar(1.5,np.mean(times_2),color='red',width=0.2)
    ax.bar(2,np.mean(times_1),color='green',width=0.2)
    ax.scatter(1,np.mean(times_3),color='blue',s=80)
    ax.scatter(1.5,np.mean(times_2),color='red',s=80)
    ax.scatter(2,np.mean(times_1),color='green',s=80)
    ax.axhline(y=np.mean(times_1),xmin=0,xmax=0.88,ls="--",c="black",linewidth=2)
    ax.axhline(y=np.mean(times_2),xmin=0,xmax=0.5,ls="--",c="black",linewidth=2)
    ax.axhline(y=np.mean(times_3),xmin=0,xmax=0.12,ls="--",c="black",linewidth=2)
    ax.tick_params(axis='both',which='both',direction='in',width=2,length=10)
    ax=plt.gca()
    bwith=4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim([4.8,7])
    plt.xticks([1,1.5,2],['','',''],fontproperties=prop,fontsize=22)
    plt.yticks(fontproperties=prop,fontsize=30)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    #plt.xlabel(r"$Clustering\ coefficient$",fontproperties=prop,fontsize=25)
    plt.xlabel(r"$Number \ of \ triangles$",fontproperties=prop,fontsize=25)
    plt.ylabel(r"$Propagation\ time \ (a.u.)$",fontproperties=prop,fontsize=25)
    plt.tight_layout()
    plt.savefig(path,dpi=300,bbox_inches='tight')
    plt.show()
    del times_1,times_2,times_3
    
def average_activation(xs_1,xs_2,xs_3,active_node,path):
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    indexs=range(00)#np.where(delta_xs_1+delta_xs_2+delta_xs_3<2.5)[0]
    delta_x_1=delta(xs_1.copy(),active_node)
    delta_x_2=delta(xs_2.copy(),active_node)
    delta_x_3=delta(xs_3.copy(),active_node)
    average_x_1=[]
    average_x_2=[]
    average_x_3=[]
    for t in indexs:
        delta_x_1_t=delta_x_1[t,:]
        average_x_1.append(np.sum(delta_x_1_t[delta_x_1_t>eta]))
        delta_x_2_t=delta_x_2[t,:]
        average_x_2.append(np.sum(delta_x_2_t[delta_x_2_t>eta]))
        delta_x_3_t=delta_x_3[t,:]
        average_x_3.append(np.sum(delta_x_3_t[delta_x_3_t>eta]))
    ax.tick_params(axis='both',which='both',direction='in',width=2,length=10)
    ax=plt.gca()
    bwith=4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([0,500,1000,1500,2000],[0,1,2,3,4],fontproperties=prop,fontsize=30)
    plt.yticks([0,600,1200],['0','0.6e3','1.2e3'],fontproperties=prop,fontsize=30)

    plt.xlabel(r"$t$",fontproperties=prop,fontsize=45)
    plt.ylabel(r"$Average\ activation$",fontproperties=prop,fontsize=25)
    
    ax.plot(average_x_1,c='blue',linewidth=4)
    ax.plot(average_x_2,c='green',linewidth=4)
    ax.plot(average_x_3,c='red',linewidth=4)
#    ax.axvline(x=dot_1,ymin=0,ymax=0.51,ls="--",c="black",linewidth=2)
#    ax.axvline(x=dot_2,ymin=0,ymax=0.84,ls="--",c="black",linewidth=2)
#    ax.scatter(dot_1,float(delta_xs_1[dot_1]),c='blue',s=80)
#    ax.scatter(dot_1,float(delta_xs_2[dot_1]),c='green',s=80)
#    ax.scatter(dot_1,float(delta_xs_3[dot_1]),c='red',s=80)
#    ax.scatter(dot_2,float(delta_xs_1[dot_2]),c='blue',s=80)
#    ax.scatter(dot_2,float(delta_xs_2[dot_2]),c='green',s=80)
#    ax.scatter(dot_2,float(delta_xs_3[dot_2]),c='red',s=80)
    plt.legend(['Add triangles','Former','Reduce triangles'],fontsize=15,bbox_to_anchor=(1.05,1.3,0,0))
    plt.tight_layout()
    plt.savefig(path,dpi=300,bbox_inches='tight')
    plt.show()
    del delta_x_1,delta_x_2,delta_x_3,average_x_1,average_x_2,average_x_3
    gc.collect()

def activation_number(xs_1,xs_2,xs_3,active_node,path):
    delta_x_1=delta(xs_1.copy(),active_node)
    delta_x_2=delta(xs_2.copy(),active_node)
    delta_x_3=delta(xs_3.copy(),active_node)
    
    indexs=range(2000)#np.where(delta_x_1+delta_x_2+delta_x_3<2.5)[0]
    num_1=[]
    num_2=[]
    num_3=[]
    for t in indexs:
        num_1.append(np.sum(delta_x_1[t,:]>eta)/np.shape(delta_x_1)[1])
        num_2.append(np.sum(delta_x_2[t,:]>eta)/np.shape(delta_x_1)[1])
        num_3.append(np.sum(delta_x_3[t,:]>eta)/np.shape(delta_x_1)[1])
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.plot(indexs,num_3,linewidth=4,c='blue')
    ax.plot(indexs,num_2,linewidth=4,c='red')
    ax.plot(indexs,num_1,linewidth=4,c='green')
    ax.tick_params(axis='both',which='both',direction='in',width=2,length=10)
    ax=plt.gca()
    bwith=4
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([0,500,1000,1500,2000],[0,1,2,3,4],fontproperties=prop,fontsize=30)
    plt.yticks([0,0.5,1],[0,0.5,1],fontproperties=prop,fontsize=30)
    plt.xlabel(r"$t$",fontproperties=prop,fontsize=45)
    plt.ylabel(r"$Proportion\ of\ activated\ nodes$",fontproperties=prop,fontsize=25)
    
    plt.locator_params(nbins=5)
    ax.axvline(x=dot_1,ymin=0,ymax=0.36,ls="--",c="black",linewidth=2)
    ax.axvline(x=dot_2,ymin=0,ymax=0.88,ls="--",c="black",linewidth=2)
    ax.scatter(dot_1,float(num_3[dot_1]),c='blue',s=80)
    ax.scatter(dot_1,float(num_2[dot_1]),c='red',s=80)
    ax.scatter(dot_1,float(num_1[dot_1]),c='green',s=80)
    ax.scatter(dot_2,float(num_3[dot_2]),c='blue',s=80)
    ax.scatter(dot_2,float(num_2[dot_2]),c='red',s=80)
    ax.scatter(dot_2,float(num_1[dot_2]),c='green',s=80)

    # plt.legend(['C=0.009','C=0.015','C=0.068'],fontsize=15,bbox_to_anchor=(1,1.3,0,0))
    plt.tight_layout()
    plt.savefig(path,dpi=300,bbox_inches='tight')
    plt.show()
    del delta_x_1,delta_x_2,delta_x_3,num_1,num_2,num_3
    gc.collect()

def cal_mat(A):
    C=np.zeros(shape=np.shape(A))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
    #for i,j in G.edges():
            if(A[i,j]>0):
                set_1=np.where(A[:,i]>0)[0].tolist()
                set_2=np.where(A[:,j]>0)[0].tolist()
                C[i,j]=len(list(set(set_1).intersection(set(set_2))))
    return C

def Leaf(tree,num):
    if(len(list(tree.values()))==0):
        num+=1
    else:
        for i in list(tree.keys()):
            num=Leaf(tree[i],num)
    return num

def DFS(tree,k,depth,edges,locations,leaves):
    leaves[0,k]=int(Leaf(tree[k],0))
    locations[depth].append(k)
    for i in list(tree[k].keys()):
        edges.append([k,i])
        DFS(tree[k],i,depth+1,edges,locations,leaves)

def block(A,G,active_node):
    C=cal_mat(np.mat(A.toarray()))
    D=np.zeros(shape=(1,np.shape(C)[0]))
    tree={}
    length=np.zeros(shape=(1,np.shape(C)[0]))
    for i in G.nodes:
        paths=[j for j in nx.all_shortest_paths(G,active_node,i)]
        values=[]
        for path in paths:
            C_1=C[:,path]
            C_1=C_1[path,:]
            values.append(np.sum(C_1.diagonal(offset=1)))
        D[0,i]=np.min(values)
        length[0,i]=len(path)
        min_index=int(np.where(values==np.min(values))[0][0])
        min_path=paths[min_index]
        ans_tree=tree
        for j in range(len(min_path)):
            if(min_path[j] not in list(ans_tree.keys())):
                ans_tree[min_path[j]]={}
#                ans_tree[min_path[j]]={}
#                ans_tree[min_path[j]][min_path[j+1]]={}
            ans_tree=ans_tree[min_path[j]]
    ans_tree=tree
    edges=[]
    locations={}
    for i in np.unique(length):
        locations[i]=[]
    leaves=np.zeros(shape=(1,np.shape(C)[0]))
    DFS(tree,active_node,1,edges,locations,leaves)
    D_numbers={}
    for i in list(locations.keys()):
        D_numbers[i]=np.maximum(1,np.max(D[0,locations[i]]))
    leaves=leaves.astype('int64')
    return D,D_numbers,length,tree,edges,leaves,locations

def Position(tree,length,k,leaves,beg,end,positions):
    positions[k]=[length[0,k],(beg+end)/2]
    tree_keys=list(tree[k].keys())
    total_leaves=np.sum(leaves[0,tree_keys])
    cum_leaves=np.cumsum(leaves[0,tree_keys])
    thetas=cum_leaves*(end-beg)/total_leaves+beg
    thetas=np.insert(thetas,0,beg)
    for i in range(len(tree_keys)):
        Position(tree[k],length,tree_keys[i],leaves,thetas[i],thetas[i+1],positions)

def cal(model,fun,active_node):
    A,G,S,xs,times=sim(model,active_node,fun)
    
    D,D_numbers,length,tree,edges,leaves,locations=block(A,G,active_node)

    positions={}
    Position(tree,length,active_node,leaves,0,np.pi*2,positions)
    return G,xs,positions,locations,edges,D,D_numbers,times

def total_plotting(model,xs,active_node,positions,locations,edges,D,D_numbers):
    for t in ts:
        path='Figures/'+model+'_'+fun+'_'+str(t)
        cycle_plotting(xs,active_node,t,positions,locations,path,edges,D,D_numbers)

def piece(D_1,D_2,D_3):
    D_numbers={}
    for i in list(D_1.keys()):
        D_numbers[i]=np.max([D_1[i],D_2[i],D_3[i]])
    return D_numbers

if __name__=='__main__':
    model='PPI2'
    fun='R'

    dot_1=int(1000)
    dot_2=int(1500)
    ts=[0,dot_1,dot_2]
    active_node=1

    G_1,xs_1,positions_1,locations_1,edges_1,D_1,D_numbers_1,times_1=cal(model+'2',fun,active_node)
    G_2,xs_2,positions_2,locations_2,edges_2,D_2,D_numbers_2,times_2=cal(model,fun,active_node)
    G_3,xs_3,positions_3,locations_3,edges_3,D_3,D_numbers_3,times_3=cal(model+'1',fun,active_node)
    D_numbers=piece(D_numbers_1,D_numbers_2,D_numbers_3)

    np.save('times_1.npy',times_1)
    np.save('times_2.npy',times_2)
    np.save('times_3.npy',times_3)
    
