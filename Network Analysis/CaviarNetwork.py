import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

phases = {}
G = {}
X = {}

for i in range(1,12):
  var_name = "phase" + str(i)
  file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
  phases[i] = pd.read_csv(file_name, index_col = ["players"])
  phases[i].columns = "n" + phases[i].columns
  phases[i].index = phases[i].columns
  G[i] = nx.from_pandas_adjacency(phases[i])
  G[i].name = var_name
  X[i] = pd.DataFrame.from_dict(nx.eigenvector_centrality(G[i]), orient='index')

print(list(dict(zip(nx.eigenvector_centrality(G[1]),nx.linalg.algebraicconnectivity.fiedler_vector(G[1])), orient='index')).keys())

#plt.plot(list(nx.eigenvector_centrality(G[10]).keys()),
#               list(nx.linalg.algebraicconnectivity.fiedler_vector(G[10])))
#plt.title("Phase 10", size=18)
#plt.xlabel("Nodes", size =14)
#plt.ylabel("Eigenvector corresponding to 2nd smallest eigenvalue", size=12)
#plt.show()

#plt.plot(np.arange(1,12),[len(phases[i]) for i in range(1,12)])
#plt.xticks(np.arange(1,12))
#plt.title("Evolution of the number of nodes", size=18)
#plt.xlabel("Phase", size=14)
#plt.ylabel("# Nodes", size=14)
#plt.show()


#plt.title("Phase 11", size=18)
#nx.draw(G[11], pos=nx.drawing.nx_agraph.graphviz_layout(G[11]), with_labels=True)
#plt.show()

#start=datetime.now()
#Z = pd.concat([X[1], X[2], X[3], X[4], X[5], X[6], X[7],
#                  X[8], X[9], X[10], X[11]], axis=1)

#Z.fillna(0)

#T = Z.sum(axis=1)/3

#print(T.sort_values())
#print(datetime.now()-start)

#start=datetime.now()
#nx.betweenness_centrality(G[1], normalized = True)
#print(datetime.now()-start)

#start=datetime.now()
#nx.eigenvector_centrality(G[1])
#print(datetime.now()-start)
