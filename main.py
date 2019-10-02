# Author: cym
import copy
import time
import math
import random
import networkx as nx
import Algorithms as algo
import IC
import Dynamic
import GraphData


RATIO = 10 / 1000
ITERATIONS = 500

# DATA_FILE = ["citation-hepth.txt", True]  # http://snap.stanford.edu/data/cit-HepTh.html
# DATA_FILE = ["citation-hepph.txt", True]  # http://snap.stanford.edu/data/cit-HepPh.html
# DATA_FILE = ["slashdot081106.txt", True]  # http://snap.stanford.edu/data/soc-sign-Slashdot081106.html
# DATA_FILE = ["gnutella04.txt", True]  # http://snap.stanford.edu/data/p2p-Gnutella04.html
# DATA_FILE = ["gnutella09.txt", True]  # http://snap.stanford.edu/data/p2p-Gnutella09.html
# DATA_FILE = ["enron.txt", True]  # http://snap.stanford.edu/data/email-Enron.html
# DATA_FILE = ["grqc.txt", True]  # http://snap.stanford.edu/data/ca-GrQc.html
# DATA_FILE = ["condmat.txt", True]  # http://snap.stanford.edu/data/ca-CondMat.html
# DATA_FILE = ["ca-hepph.txt", True]  # http://snap.stanford.edu/data/ca-HepPh.html
# DATA_FILE = ["ca-hepth.txt", True]  # http://snap.stanford.edu/data/ca-HepTh.html
# DATA_FILE = ["wiki-vote.txt", True]  # http://snap.stanford.edu/data/wiki-Vote.html
# DATA_FILE = ["soc-Epinions1.txt", True]  # http://snap.stanford.edu/data/soc-Epinions1.html
DATA_FILE = ["email-Eu.txt", True]  # http://snap.stanford.edu/data/email-Eu-core.html
# DATA_FILE = ["CollegeMsg.txt", True]  # http://snap.stanford.edu/data/CollegeMsg.html
# DATA_FILE = ["DBLP.txt", False]
# DATA_FILE = ["facebook.txt", False]  # http://snap.stanford.edu/data/ego-Facebook.html
# DATA_FILE = ["hep.txt", False]
# DATA_FILE = ["phy.txt", False]
# DATA_FILE = ["math.txt", True]  # http://snap.stanford.edu/data/sx-mathoverflow.html


def generate_activation_probability(graph):
    pa = 0.02
    pb = 0.06
    # activation_probability = {(u, v): (pa + (m / len(set(graph.predecessors(v)))) * pb) for (u, v, m) in graph.edges.data("m")}
    activation_probability = {(u, v): 1 / len(set(graph.predecessors(v))) for (u, v, m) in graph.edges.data("m")}
    return activation_probability

# def generate_cost(graph):
#     pagerank_values = nx.pagerank(graph)
#     max_pr = max(pagerank_values.values())
#
#     sorted_pr = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
#
#     # 激活一个节点的固定开销
#     fixed_cost = sorted_pr[int(0.6 * len(sorted_pr))][1] / max_pr
#
#     uni_pr = {x[0]: x[1]/max_pr for x in sorted_pr}
#
#     print("generate_cost done! fixed_cost: ", fixed_cost)
#     return uni_pr, fixed_cost


def init_pa(graph, act_prob):
    unact_prob = {node: 1 for node in graph}
    for u in graph:
        for v in graph.successors(u):
            unact_prob[v] *= (1 - act_prob[(u, v)])
    pa = {k: 1 - v for k, v in unact_prob.items()}
    return pa


def generate_cost(nodes_values, pa):
    alpha = 0.7
    beta = 1 - alpha
    gama = 10

    max_value = math.exp(gama * max(nodes_values.values()))
    part1 = {k: math.exp(gama * v) / max_value for k, v in nodes_values.items()}

    part2 = {k: 1 - math.log(v + 1, 2) for k, v in pa.items()}
    max_pa = max(part2.values())
    for k, v in part2.items():
        part2[k] = v / max_pa

    # costs = {k: alpha * part1[k] + beta * part2[k] for k, v in nodes_values.items()}
    costs = {k: alpha * nodes_values[k] + beta * part2[k] for k, v in nodes_values.items()}

    sorted_costs = sorted(costs.items(), key=lambda x: x[1], reverse=True)
    max_cost = sorted_costs[0][1]

    # 激活一个节点的固定开销
    fixed_cost = sorted_costs[int(0.7 * len(sorted_costs))][1] / max_cost

    uni_costs = {x[0]: x[1]/max_cost for x in sorted_costs}

    costs_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in uni_costs.values():
        costs_count[int((v * 10)) % 10] += 1
    t = sum(costs_count)
    print("网络中节点花费分布")
    print([round(100 * y / t, 3) for y in costs_count])

    print("generate_cost done! fixed_cost: ", fixed_cost)
    return uni_costs, fixed_cost


def generate_value(graph):
    pagerank_values = nx.pagerank(graph)
    max_pr = max(pagerank_values.values())
    values = {k: v/max_pr for k, v in pagerank_values.items()}

    values_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in values.values():
        values_count[int((v*10)) % 10] += 1
    t = sum(values_count)
    print("网络中节点价值分布")
    print([round(100*y/t, 3) for y in values_count])

    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
    index = [int(len(sorted_values) * 0.1 * x) for x in range(1, 11)]
    tmp = [x[1] for x in sorted_values]
    accum_values = [sum(tmp[:x]) for x in index]
    print("网络中节点价值累计分布")
    print([round(y, 3) for y in accum_values])

    print("generate_values done!")
    return values


def generate_bound(costs):
    top_k_users = max(50, int(RATIO * len(costs)))
    bound_cost = sum([x[1] for x in sorted(costs.items(), key=lambda x: x[1], reverse=True)[:top_k_users]])
    print("Total Cost:", bound_cost)
    return bound_cost


def invoke(f):
    print("######    invoking ", f.__name__, "    ######")
    start_time = time.time()
    seeds_set = f(graph, act_prob, values, costs, bound_cost, fixed_cost)
    print("seeds num: ", len(seeds_set))
    gain, count = IC.avg_IC(graph, seeds_set, act_prob, values, ITERATIONS)
    print("gain: ", gain, " active nodes num: ", count)
    print("time: ", time.time() - start_time)
    total_cost = 0
    for v in seeds_set:
        total_cost += costs[v] + fixed_cost
    print("total_cost: ", total_cost, "bound_cost: ", bound_cost)
    print()


if __name__ == "__main__":
    start = time.time()
    graph = GraphData.ini_graph(DATA_FILE[0], DATA_FILE[1])

    act_prob = generate_activation_probability(graph)
    pa = init_pa(graph, act_prob)
    values = generate_value(graph)
    costs, fixed_cost = generate_cost(values, pa)
    bound_cost = generate_bound(costs)
    print()

    # invoke(algo.min_cost)

    invoke(algo.just_max_values)
    invoke(algo.rank_values)

    # invoke(Dynamic.func2)
    #
    # invoke(algo.degree_dis_improved)
    #
    # invoke(algo.degree_dis_effic)
    #
    # invoke(algo.degree_dis_gain)
    #
    # invoke(algo.degree_dis_gain_effic)

    total_time = time.time()-start
    print("########    All Done! Using time %d min %d sec  ########" % (total_time//60, total_time % 60))















