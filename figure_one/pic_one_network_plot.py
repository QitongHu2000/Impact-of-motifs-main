import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat
import numpy as np


seed = 13648  # Seed random number generators for reproducibility

G = nx.generators.random_graphs.erdos_renyi_graph(100, p=0.1, seed=seed, directed=False)

G = nx.DiGraph(G)

nnodes = G.number_of_nodes()
deg = sum(d for n, d in G.degree()) / float(nnodes)
print('average degree:',deg/2)

pos = nx.circular_layout(G,scale=1) 

plt.figure(figsize=(10, 10))

nodes = nx.draw_networkx_nodes(G, pos, node_size=80, node_color="gray")

for edge in G.edges(data=True):
    if edge[0] > edge[1]+1:
        if abs(edge[0]- edge[1]) <80:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {0.3}')
        else:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.3}')
    if edge[0] ==   edge[1]+1:
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.1}')

ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.savefig('net_three.pdf',dpi = 300)
plt.show()

G = nx.generators.connected_watts_strogatz_graph(100, k=10, p = 0.1, tries=100, seed=seed)

G = nx.DiGraph(G)

nnodes = G.number_of_nodes()
deg = sum(d for n, d in G.degree()) / float(nnodes)
print('average degree:',deg/2)

pos = nx.circular_layout(G,scale=1) 

plt.figure(figsize=(10, 10))

nodes = nx.draw_networkx_nodes(G, pos, node_size=80, node_color="gray")

for edge in G.edges(data=True):
    if edge[0] > edge[1]+1:
        if abs(edge[0]- edge[1]) <80:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {0.3}')
        else:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.3}')
    if edge[0] ==   edge[1]+1:
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.1}')

ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.savefig('net_two.pdf',dpi = 300)
plt.show()


A=loadmat('PPI2')['A']
G_whole=nx.from_numpy_matrix(A)
nodes = np.arange(200,300) # choose 100 nodes
G = G_whole.subgraph(nodes)
G = nx.DiGraph(G)

nnodes = G.number_of_nodes()
deg = sum(d for n, d in G.degree()) / float(nnodes)
print('average degree:',deg/2)

pos = nx.circular_layout(G,scale=1) 

plt.figure(figsize=(10, 10))

nodes = nx.draw_networkx_nodes(G, pos, node_size=80, node_color="gray")

for edge in G.edges(data=True):
    if edge[0] > edge[1]+1:
        if abs(edge[0]- edge[1]) <80:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {0.3}')
        else:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.3}')
    if edge[0] ==   edge[1]+1:
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.1}')

ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.savefig('net_one.pdf',dpi = 300)
plt.show()




