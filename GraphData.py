# Author: cym
import re

import pandas as pd
import networkx as nx

def load_graph(file_name):
    data = pd.read_csv("./datasets/" + file_name)
    edge_count = {}
    for _, row in data.iterrows():
        try:
            edge_count[(row["src"], row["tgt"])] += 1
        except KeyError:
            edge_count[(row["src"], row["tgt"])] = 1
    edge_list = [(k[0], k[1], {"m": v}) for k, v in edge_count.items()]
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    edge_nums = 0
    for (u, v, m) in graph.edges.data("m"):
        edge_nums += m
    print("Graph edge nums by count m:", edge_nums)
    print("nodes number: ", graph.number_of_nodes(), " edges number: ", graph.number_of_edges())
    return graph


def ini_graph(file_name, Di = True):
    edge_count = {}
    # with open("./datasets/" + file_name) as f:
    with open("E:\\Influence Maxization\\Dataset\\boundedCostDataset\\" + file_name) as f:
        nodes_edges = f.readline().rstrip('\n').split(' ')
        nodes_num = nodes_edges[0]
        edges_num = nodes_edges[1]

        for line in f:
            line = line.rstrip('\n').split(' ')
            u, v = line[0], line[1]
            try:
                edge_count[(u, v)] += 1
            except KeyError:
                edge_count[(u, v)] = 1
            if Di is False:
                try:
                    edge_count[(v, u)] += 1
                except KeyError:
                    edge_count[(v, u)] = 1

    edge_list = [(k[0], k[1], {"m": v}) for k, v in edge_count.items()]
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    print("********    ", file_name, "    ********")
    print("nodes number: ", graph.number_of_nodes(), " edges number: ", graph.number_of_edges())
    return graph

