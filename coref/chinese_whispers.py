import numpy as np
from random import shuffle
import networkx as nx


def chinese_whispers(start_scores, end_scores, cluster_scores, input_mask, threshold=7, iterations=20):
    """ Chinese Whispers Algorithm
    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """
    # Create graph
    nodes = []
    edges = []

    end_ids = end_scores.argmax(-1)
    mentions_mask = start_scores.argmax(-1) * input_mask

    for i in range(cluster_scores.shape[0]):
        if mentions_mask[i] == 0:
            continue

        mention = (i, end_ids[i])
        node = (i + 1, {'cluster': mention, 'mention': mention})
        nodes.append(node)
        for j in range(i + 1, cluster_scores.shape[1]):
            if mentions_mask[j] == 0:
                continue

            if cluster_scores[i, j] > threshold:
                edges.append((i + 1, j + 1, {'weight': cluster_scores[i, j]}))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = list(G.nodes())
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        #该节点邻居节点的类别的权重
                        #对应上面的字典cluster的意思就是
                        #对应的某个路径下文件的权重
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = G.node[node]['cluster']
            #将邻居节点的权重最大值对应的文件路径给到当前节点
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        mention = data['mention']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(mention)

    clusters = set(map(tuple, clusters.values()))
    clusters = sorted(clusters, key=lambda c: c[0][0])
    clusters = sorted(clusters, key=len, reverse=True)
    clusters = tuple(tuple(sorted(c, key=lambda x: x[0])) for c in clusters)

    return clusters
