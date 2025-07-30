import EoN
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.io import loadmat
import random


def get_odd_numbers(original_list):
    new_list = []
    n = len(original_list)
    num_to_extract = int(n * 0.05)
    for i in range(0, n, 1):
        if len(new_list) < num_to_extract:
            new_list.append(original_list[i])
        else:
            break
    return new_list


def sir(G, tau, gamma, rank):
    infection_probability = tau
    recovery_probability = gamma
    ww = [0] * 30
    for wj in range(1000):
        node_status = {node: 'susceptible' for node in G.nodes}
        for node in rank:
            node_status[node] = 'infected'

        infected_nodes = []
        recovered_nodes = []
        ir = []
        for t in range(30):
            infected_count = len([n for n, status in node_status.items() if status == 'infected'])
            recovered_count = len([n for n, status in node_status.items() if status == 'recovered'])
            infected_nodes.append(infected_count)
            recovered_nodes.append(recovered_count)
            ir.append(infected_count + recovered_count)

            for node in G.nodes:
                if node_status[node] == 'recovered':
                    node_status[node] = 'recovered'
                if node_status[node] == 'infected':
                    neighbors = list(G.neighbors(node))
                    for n in neighbors:
                        if node_status[n] == 'susceptible' and np.random.random() < infection_probability:
                            node_status[n] = 'infected'
                    if np.random.random() < recovery_probability:
                        node_status[node] = 'recovered'
        ww = [x + y for x, y in zip(ww, ir)]
    data = [x / 1000 for x in ww]
    return data


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

    gnne_zz = get_odd_numbers(rank)
    average_degree = np.mean([d for n, d in G.degree()])
    degrees = [d for _, d in G.degree()]
    second_order_moment = sum(d ** 2 for d in degrees) / len(degrees)
    propagation_threshold_probability = average_degree / (second_order_moment - average_degree)
    p = 1 * propagation_threshold_probability

    gamma = 1
    tau = p

    data1 = sir(G, tau, gamma, gnne_zz)

    time = range(30)
    plt.plot(time, data1, label='GNNE')

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('number of infected nodes')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()
    print(f"{data1}".strip("[]"))
