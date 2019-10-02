# Author: cym
import GraphData
import networkx as nx
import sys
import time
from networkx.algorithms.community import k_clique_communities

FILE_NAME = "Amazon.txt"
# FILE_NAME = "CollegeMsg.txt"
# FILE_NAME = "DBLP.txt"
# FILE_NAME = "facebook.txt"
# FILE_NAME = "hep.txt"
# FILE_NAME = "phy.txt"


def find_community(graph, k):
    return list(k_clique_communities(graph, k))


G = GraphData.ini_graph(FILE_NAME)

for k in range(5, 10):
    print("############# k值: %d ################" % k)
    start_time = time.clock()
    rst_com = find_community(G, k)
    end_time = time.clock()
    print("计算耗时(秒)：%.3f" % (end_time - start_time))
    print("生成的社区数：%d" % len(rst_com))
    print(rst_com)
