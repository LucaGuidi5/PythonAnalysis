import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_dataset(n_times=50):
    # preprocess dataset

    df = pd.read_csv('dataset/heroNetwork.csv')
    # remove rows that have same hero1 and hero2
    df = df.loc[df['hero1'] != df['hero2']]
    # swap hero1 and hero2 if hero1 > hero2
    df['hero1'], df['hero2'] = np.where(df['hero1'] > df['hero2'], [df['hero2'], df['hero1']], [df['hero1'], df['hero2']])
    
    # select rows that appear more than n_times times
    df = df.groupby(['hero1', 'hero2']).size().reset_index(name='weight')
    df = df.loc[df['weight'] >= n_times]

    heroes = nx.from_pandas_edgelist(df, source = "hero1", target = "hero2")

    # get bigger connected component
    Gcc = sorted(nx.connected_components(heroes), key=len, reverse=True)
    heroes = heroes.subgraph(Gcc[0])
    return heroes

def draw(G, pos, measures, measure_name, draw_labels=False):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    
    if draw_labels:
        labels = nx.draw_networkx_labels(G, pos)
        
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()