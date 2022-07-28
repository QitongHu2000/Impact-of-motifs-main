import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

a=0.8
b=0.5

def combine_data():
    fun='R'
    total_times=np.load(fun+'_time_std_total_times'+str(int(a*10))+'_'+str(int(b*10))+'.npy')
    total_std=np.load(fun+'_time_std_total_std'+str(int(a*10))+'_'+str(int(b*10))+'.npy')
    total_mean=np.load(fun+'_time_std_total_mean'+str(int(a*10))+'_'+str(int(b*10))+'.npy').astype('int')
    total_mean = list(total_mean)
    return total_times,total_std,total_mean

def F(x):
    return np.power(x,1)

def get_matrix(total_mean,total_std,total_times):
    len_x=np.max(total_mean)+1
    len_y=np.max(total_std)+1
    result=np.zeros(shape=(len_x,len_y))
    result_all = {}
    data={}
    ans_data=total_times.copy()
    ans_data=ans_data[~np.isnan(ans_data)]
    ans_data=ans_data[~np.isinf(ans_data)]
    max_c=np.max(ans_data)
    min_c=np.min(ans_data)

    for i in range(len(total_mean)):
        alpha=(total_times[i]-min_c)/(max_c-min_c)
        if(np.isnan(alpha) or np.isinf(alpha)):
            continue
        try:
            data[total_mean[i]][total_std[i]].append(alpha)
        except:
            try:
                data[total_mean[i]][total_std[i]]=[]
                data[total_mean[i]][total_std[i]].append(alpha)
            except:
                data[total_mean[i]]={}
                data[total_mean[i]][total_std[i]]=[]
                data[total_mean[i]][total_std[i]].append(alpha)
    for i in list(data.keys()):
        for j in list(data[i].keys()):
            alpha=np.array(data[i][j])
            # alpha=alpha[~ np.isnan(alpha)]
            # alpha=alpha[~ np.isinf(alpha)]
            result[i,j]=np.mean(alpha)
            try:
                result_all[(i,j)] = alpha
            except:
                result_all.set_default((i,j), array([]))
                result_all[(i,j)] = alpha
    return result, result_all

def plotting_matrix(result):
    fig,ax=plt.subplots(figsize=(6,6))
    sns.set(style="white")
    
    d=pd.DataFrame(data=F(result.T))
    indexs=np.flip(d.index)
    d.columns+=2
    d=d.reindex(index=indexs)

    cmap = "inferno"
    d_cut = d[:][:-1]
    d_cut_amend = d_cut   
    sns.heatmap(d_cut_amend,mask=d_cut_amend == 0,cmap=cmap,vmax=max(d_cut.max())   ,vmin = min(d_cut.min())  ,center=0,square=True, linewidths=1,
                cbar_kws={"shrink":.5},xticklabels=4,yticklabels=4,ax=ax)
    
    ax.set_xlabel('Mean degree',fontsize=30)
    ax.set_ylabel('Standard variance',fontsize=30)
    #ax.set_xticks(labelsize=25)
    #ax.set_yticks(labelsize=25)
    plt.tick_params(labelsize=25)
    plt.tight_layout()
    #plt.savefig('matrix_'+str(int(a*100))+'.pdf',dpi=300)
    plt.savefig('colorbar_matrix_'+str(int(a*100))+'.pdf',dpi=300)
    plt.show()

def plotting_line(result,result_all,path):
    ranges=[3,5]
    min_inx = []
    max_inx = []
    for i in ranges:
        indexs=np.where(result.T[i,:]>0)[0]
        min_inx.append(min(indexs))
        max_inx.append(max(indexs))
        #print(indexs,result.T[i,indexs])

    x_data = np.arange(6,11)
    
    #colors = ['#A393C6','#4C9C98','#DE9689']
    colors = ['#ECCD48','#B4394F','#DE9689']

    fig=plt.figure(figsize=(6,6))
    sns.set(style='white')
    m = 0
    for i in ranges: #(std) 
        indexs=np.where(result.T[i,:]>0)[0]
        y_data = []
        standard = []
        mean = []
        for j in x_data:  #(mean)
            try:     
                y_plt = result.T[i,indexs]
                indexs = list(indexs)
                val = indexs.index(j)
                y_data.append(y_plt[val])
                standard.append(np.std(result_all[(j, i)]))
                mean.append(np.mean(result_all[(j, i)]))
            except:
                y_data.append(0)
                standard.append(0)
                mean.append(0)
        
        plt.fill_between(np.arange(len(y_data)), y_data - np.array(standard), y_data + np.array(standard), color = colors[m], alpha = 0.4)
        plt.plot(np.arange(len(y_data)),y_data,color = colors[m], linewidth=4, alpha=1.0,label ='std='+str(i))
        m += 1
   
    ax=plt.gca()
    bwith=1
    #plt.ylim(0.2,0.8)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Mean degree',fontsize =30)
    plt.ylabel('Propagation time (a.u.)',fontsize =30)
    plt.xticks(np.arange(len(y_data)),x_data,fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(fontsize = 20)
    plt.tight_layout()
    plt.savefig('line_'+str(int(a*100))+'.pdf',dpi=300, bbox_inches='tight')
    plt.show()
   
total_times,total_std,total_mean=combine_data()
result, result_all=get_matrix(total_mean,total_std,total_times)
fun='R'
path=fun+'_std_'+str(int(a*10))+'_'+str(int(b*10))+'.eps'
plotting_matrix(result)
plotting_line(result,result_all,path)
