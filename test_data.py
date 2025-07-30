import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
from scipy.io import loadmat


def adj(G):
    adjacency_matrix = nx.to_numpy_array(G)
    a_int2 = adjacency_matrix.astype(np.int64)
    return a_int2


def degree_heterogeneity(G):
    ak = np.average([n for n in dict(G.degree()).values()])
    H = sum([abs(G.degree[i] - G.degree[j]) for i in G.nodes() for j in G.nodes()]) / (2 * ak * (len(G) ** 2))
    return round(H, 4)


def cora_data():
    path = r'./dataset\cora.mat'
    data = loadmat(path)
    cora_data = data['cora']
    G = nx.from_scipy_sparse_array(cora_data)
    return G


def usair_data():
    path = r'./dataset\USAir.mat'
    data = loadmat(path)
    USAir_data = data['USAir']
    G = nx.from_scipy_sparse_array(USAir_data)
    return G


def geom_data():
    path = r'/dataset\geom.mat'
    data = loadmat(path)
    geom_data = data['geom']
    G = nx.from_scipy_sparse_array(geom_data)
    return G


def email_data():
    path = r'./dataset\email.mat'
    data = loadmat(path)
    email_data = data['email']
    nodes = set()
    for item in email_data:
        nodes.add(item[0] - 1)
        nodes.add(item[1] - 1)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = [(item[0] - 1, item[1] - 1) for item in email_data]
    G.add_edges_from(edges)
    return G


def PB_data():
    path = r'./dataset\PB.mat'
    data = loadmat(path)
    PB_data = data['polblogs']
    G = nx.from_scipy_sparse_array(PB_data)
    return G


def power_data():
    path = r'./dataset\Power.mat'
    data = loadmat(path)
    Power_data = data['Power']
    G = nx.from_scipy_sparse_array(Power_data)
    return G


if __name__ == '__main__':

    # dataset = cora_data()
    # dataset = usair_data()
    # dataset = geom_data()
    # dataset = email_data()
    # dataset = PB_data()
    dataset = power_data()

    nx.draw(dataset, with_labels=True)
    plt.show()
    largest_cc = max(len(c) for c in nx.connected_components(dataset))
    d = dict(nx.degree(dataset))
    print("Average Degree：", sum(d.values()) / len(dataset.nodes))
    plt.figure()
    x = list(range(max(d.values()) + 1))
    y = [i / len(dataset) for i in nx.degree_histogram(dataset)]

    plt.plot(x, y, 'ro-')
    plt.xlabel("$k$")
    plt.ylabel("$p_k$")
    plt.show()

    plt.figure()
    plt.plot(x, y, 'ro-')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$k$")
    plt.ylabel("$p_k$")
    plt.show()

    plt.figure()
    new_x = []
    new_y = []
    for i in range(len(x)):
        if y[i] != 0:
            new_x.append(x[i])
            new_y.append(y[i])
    plt.plot(new_x, new_y, 'ro-')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$k$")
    plt.ylabel("$p_k$")
    plt.show()
    adj = adj(dataset)
    d = nx.average_shortest_path_length(dataset)
    C = nx.average_clustering(dataset)
    r = nx.degree_assortativity_coefficient(dataset)
    DH = degree_heterogeneity(dataset)
    scio.savemat('power_adj.mat', {'power_adj': adj})
    print(d)
    print(C)
    print(r)
    print(DH)
