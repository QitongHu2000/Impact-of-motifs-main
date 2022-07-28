import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

seed = 13648  # Seed random number generators for reproducibility
pro = 0.07
G = nx.generators.random_graphs.erdos_renyi_graph(50, pro, seed=seed, directed=False)
nx.write_gexf(G,'gephi_network'+str(int(pro*100))+'.gexf')

