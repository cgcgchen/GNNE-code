import networkx as nx
import copy
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np


def network_efficiency(removal_ratios, rank_list, G):
    efficiencies_sizes = []
    initial_nodes = len(G.nodes)
    efficiencies_sizes.append(nx.global_efficiency(G))
    for ratio in removal_ratios:
        num_nodes_to_remove = int(ratio * initial_nodes)
        nodes_to_remove = set()
        for node in rank_list[:num_nodes_to_remove]:
            nodes_to_remove.add(node)
        G_copy = copy.deepcopy(G)
        G_copy.remove_nodes_from(nodes_to_remove)
        efficiencies_size = nx.global_efficiency(G_copy)
        efficiencies_sizes.append(efficiencies_size)
    return efficiencies_sizes


def network_efficiency_node(rank_list, G):
    efficiencies_sizes = []
    G_copy = copy.deepcopy(G)
    initial_efficiency_size = nx.global_efficiency(G_copy)
    efficiencies_sizes.append(initial_efficiency_size)
    nodes_removed = 0
    for node in rank_list:
        G_copy.remove_node(node)
        nodes_removed += 1
        efficiencies_size = nx.global_efficiency(G_copy)
        efficiencies_sizes.append(efficiencies_size)
        if efficiencies_size <= 0.01 * initial_efficiency_size:
            print(f"nodes_removed: {nodes_removed},nx_size: {efficiencies_size}")
            break
    return efficiencies_sizes


if __name__ == '__main__':
    path = r'./power_adj.mat'
    adj = loadmat(path)
    adj = adj['power_adj']
    adj = sp.csr_matrix(adj)
    G = nx.from_scipy_sparse_array(adj)

    path_1 = r'./power_rank_list.mat'
    rank = loadmat(path_1)
    rank = rank['power_rank_list']
    rank = rank[0]

    removal_ratios = [0.02 * (i + 1) for i in range(25)]

    efficiencies_gnne = network_efficiency(removal_ratios, rank, G)

    percentages = [i * 0.02 for i in range(26)]

    plt.plot(percentages, efficiencies_gnne, marker='2', label='GNNE')
    plt.xlabel('Number of Nodes Removed')
    plt.ylabel('Global Network Efficiency')
    plt.title('Network Efficiency vs. Number of Nodes Removed')
    plt.legend()
    plt.xlim(0, 0.5)
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()
    print(f"{efficiencies_gnne}".strip("[]"))
