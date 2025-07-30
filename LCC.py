import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.io import loadmat
import copy
import scipy.io as scio


def LCC_size(G, rank_list):
    lcc_sizes = []
    initial_nodes = len(G.nodes)
    lcc_sizes.append(max(len(c) for c in nx.connected_components(G)))
    removal_ratios = [0.02 * (i + 1) for i in range(50)]

    for ratio in removal_ratios:
        num_nodes_to_remove = int(ratio * initial_nodes)
        nodes_to_remove = set()
        for node in rank_list[:num_nodes_to_remove]:
            nodes_to_remove.add(node)
        G_copy = copy.deepcopy(G)
        G_copy.remove_nodes_from(nodes_to_remove)
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) if G_copy else 0
        lcc_sizes.append(lcc_size)
    return lcc_sizes


def LCC_size_node(G, rank_list):
    lcc_sizes = []
    G_copy = copy.deepcopy(G)
    initial_lcc_size = max(len(c) for c in nx.connected_components(G))
    lcc_sizes.append(initial_lcc_size)
    nodes_removed = 0
    for node in rank_list:
        G_copy.remove_node(node)
        nodes_removed += 1
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) if G_copy else 0
        lcc_sizes.append(lcc_size)
        if lcc_size <= 0.01 * initial_lcc_size:
            print(f"nodes_removed: {nodes_removed}, LCC_size: {lcc_size}")
            break
    return lcc_sizes


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

    gnne_rank_LCC = LCC_size(G, rank)

    lcc_relative_sizes_gnne = [size / gnne_rank_LCC[0] for size in gnne_rank_LCC]

    percentages = [i * 0.02 for i in range(51)]

    plt.plot(percentages, lcc_relative_sizes_gnne, label='GNNE', marker='2')
    plt.xlabel('Percentage of nodes removed')
    plt.ylabel('Relative size of LCC')
    plt.title('LCC Relative Size Change with Node Removal')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
    print(f"{lcc_relative_sizes_gnne}".strip("[]"))
