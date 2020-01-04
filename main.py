# Author: cym
import copy
import time
import math
import random
import networkx as nx
import Algorithms as algo
import IC
import GraphData
import drawscatter


ITERATIONS = 1000


# DATA_FILE = ["konect-ca-AstroPh.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-brightkite.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-douban.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-facebook.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-gowalla.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-friendships.txt", False]  # http://konect.uni-koblenz.de/networks/
# DATA_FILE = ["konect-dblp.txt", True]  # http://konect.uni-koblenz.de/networks/dblp-cite

# DATA_FILE = ["citation-hepth.txt", True]  # http://snap.stanford.edu/data/cit-HepTh.html
# DATA_FILE = ["citation-hepph.txt", True]  # http://snap.stanford.edu/data/cit-HepPh.html
# DATA_FILE = ["slashdot081106.txt", True]  # http://snap.stanford.edu/data/soc-sign-Slashdot081106.html
# DATA_FILE = ["gnutella09.txt", True]  # http://snap.stanford.edu/data/p2p-Gnutella09.html
# DATA_FILE = ["enron.txt", True]  # http://snap.stanford.edu/data/email-Enron.html
# DATA_FILE = ["condmat.txt", True]  # http://snap.stanford.edu/data/ca-CondMat.html
# DATA_FILE = ["ca-hepph.txt", True]  # http://snap.stanford.edu/data/ca-HepPh.html
# DATA_FILE = ["wiki-vote.txt", True]  # http://snap.stanford.edu/data/wiki-Vote.html
# DATA_FILE = ["soc-Epinions1.txt", True]  # http://snap.stanford.edu/data/soc-Epinions1.html
# DATA_FILE = ["email-Eu.txt", True]  # http://snap.stanford.edu/data/email-Eu-core.html
# DATA_FILE = ["CollegeMsg.txt", True]  # http://snap.stanford.edu/data/CollegeMsg.html
# DATA_FILE = ["DBLP.txt", False]
# DATA_FILE = ["facebook.txt", False]  # http://snap.stanford.edu/data/ego-Facebook.html
# DATA_FILE = ["hep.txt", False]
# DATA_FILE = ["phy.txt", False]  # 其实这个也不错，只不过找不到数据源出处了
# DATA_FILE = ["math.txt", True]  # http://snap.stanford.edu/data/sx-mathoverflow.html


# DATA_FILE = ["gnutella04.txt", True]  # http://snap.stanford.edu/data/p2p-Gnutella04.html
# DATA_FILE = ["konect-arenas.txt", False]  # http://konect.uni-koblenz.de/networks/arenas-pgp
# DATA_FILE = ["ca-hepth.txt", True]  # http://snap.stanford.edu/data/ca-HepTh.html
DATA_FILE = ["grqc.txt", True]  # http://snap.stanford.edu/data/ca-GrQc.html


def generate_activation_probability(graph):
    pa = 0.02
    pb = 0.06
    # activation_probability = {(u, v): (pa + (m / len(set(graph.predecessors(v)))) * pb) for (u, v, m) in graph.edges.data("m")}
    activation_probability = {(u, v): 1 / len(set(graph.predecessors(v))) for (u, v, m) in graph.edges.data("m")}
    return activation_probability


def init_spa(graph, act_prob):
    unact_prob = {node: 1 for node in graph}
    for u in graph:
        for v in graph.successors(u):
            unact_prob[v] *= (1 - act_prob[(u, v)])
    spa = {k: 1 - v for k, v in unact_prob.items()}
    return spa


def generate_cost(nodes_values, spa):
    alpha = 0.7
    beta = 1 - alpha

    spa_part = {k: 1 - math.log(v + 1, 2) for k, v in spa.items()}
    # max_pa = max(spa_part.values())
    # for k, v in spa_part.items():
    #     spa_part[k] = v / max_pa

    c = {k: alpha * nodes_values[k] + beta * spa_part[k] for k, v in nodes_values.items()}

    sorted_c = sorted(c.items(), key=lambda x: x[1], reverse=True)
    max_c = sorted_c[0][1]

    # 激活一个节点的固定开销
    fixed_cost = sorted_c[int(0.7 * len(sorted_c))][1] / max_c

    norm_c = {x[0]: x[1]/max_c for x in sorted_c}
    costs = {k: v + fixed_cost for k, v in norm_c.items()}

    costs_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in norm_c.values():
        costs_count[int((v * 10)) % 10] += 1
    t = sum(costs_count)
    print("网络中节点花费分布")
    print([round(100 * y / t, 3) for y in costs_count])

    print("generate_cost done! fixed_cost: ", fixed_cost)
    return costs


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
    top_k_users = int(RATIO * len(costs))
    bound_cost = sum([x[1] for x in sorted(costs.items(), key=lambda x: x[1], reverse=True)[:top_k_users]])
    print("Total Cost:", bound_cost)
    return bound_cost


def invoke(f):
    print("######    invoking ", f.__name__, "    ######")
    start_time = time.time()
    seeds_set = f(graph, act_prob, values, costs, bound_cost)
    print("time: ", time.time() - start_time)
    print("seeds num: ", len(seeds_set))
    gain, count = IC.avg_IC(graph, seeds_set, act_prob, values, ITERATIONS)
    print("gain: ", gain, " active nodes num: ", count)

    total_cost = 0
    for k in seeds_set:
        total_cost += costs[k]
    print("total_cost: ", total_cost, "bound_cost: ", bound_cost)
    print()


# if __name__ == "__main__":
#     RATIO = 10 / 1000
#
#     start = time.time()
#     graph = GraphData.ini_graph(DATA_FILE[0], DATA_FILE[1])
#
#     act_prob = generate_activation_probability(graph)
#     pa = init_spa(graph, act_prob)
#     values = generate_value(graph)
#     costs = generate_cost(values, pa)
#     bound_cost = generate_bound(costs)
#     print()
#
#     # drawscatter.draw_scatter(DATA_FILE[0], costs, values)
#
#     invoke(algo.MCF)
#
#     invoke(algo.MIF)
#
#     invoke(algo.MIC)
#
#     invoke(algo.MGF)
#
#     invoke(algo.MGCRF)
#
#     invoke(algo.MREF)
#
#
#     total_time = time.time()-start
#     print("########    All Done! Using time %d min %d sec  ########" % (total_time//60, total_time % 60))


if __name__ == "__main__":
    graph = GraphData.ini_graph(DATA_FILE[0], DATA_FILE[1])
    act_prob = generate_activation_probability(graph)
    pa = init_spa(graph, act_prob)
    values = generate_value(graph)
    costs = generate_cost(values, pa)

    def simple_invoke(f):
        seeds_set = f(graph, act_prob, values, costs, bound_cost)
        gain, count = IC.avg_IC(graph, seeds_set, act_prob, values, ITERATIONS)
        return round(gain, 2)
    import csv

    name = "".join(list(DATA_FILE[0])[:-4])
    # 1. 创建文件对象
    result_file = open(name + '.csv', 'w', encoding='utf-8', newline="")

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(result_file)

    # 3. 构建列表头
    csv_writer.writerow([x for x in range(0, 11)])

    content = [[0 for _ in range(11)] for _ in range(6)]
    for x in range(1, 11):
        RATIO = x / 1000
        print("#"*8 + " "*2 + str(x) + " "*2 + "#"*8)
        bound_cost = generate_bound(costs)
        gain_MCF = simple_invoke(algo.MCF)
        gain_MIF = simple_invoke(algo.MIF)
        gain_MIC = simple_invoke(algo.MIC)
        gain_MGF = simple_invoke(algo.MGF)
        gain_MGCRF = simple_invoke(algo.MGCRF)
        gain_MREF = simple_invoke(algo.MREF)

        content[0][x] = gain_MCF
        content[1][x] = gain_MIF
        content[2][x] = gain_MIC
        content[3][x] = gain_MGF
        content[4][x] = gain_MGCRF
        content[5][x] = gain_MREF

    # 4. 写入csv文件内容
    for line in content:
        csv_writer.writerow(line)

    # 5. 关闭文件
    result_file.close()












