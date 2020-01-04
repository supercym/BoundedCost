# Author: cym
import copy
import math
import random


def MIF(graph, act_prob, values, cost, bound_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    rank = {}
    un_act_prob = {}
    act_ability = {}
    total_cost = 0
    first_time = True
    while True:
        if first_time:
            for u in graph_back:
                un_act_prob[u] = 1
                act_ability[u] = 1
                for v in graph_back.successors(u):
                    act_ability[u] += act_prob[(u, v)]
                rank[u] = un_act_prob[u] * act_ability[u]
            first_time = False
        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] > bound_cost:
            break
        total_cost += (cost[seed])
        seeds_set.add(seed)
        # Gb.remove_node(g)
        rank[seed] = -1
        change_nodes = set()
        for v in graph_back.successors(seed):
            un_act_prob[v] *= (1 - act_prob[(seed, v)])
            change_nodes.add(v)
        for v in graph_back.predecessors(seed):
            act_ability[v] -= act_prob[(v, seed)]
            change_nodes.add(v)
        for v in change_nodes.difference(seeds_set):
            rank[v] = un_act_prob[v] * act_ability[v]

    print("MIF done!")
    return seeds_set


def irie_rank(graph, act_prob, cost, bound_cost, fixed_cost):
    nodes = set(graph.nodes())
    # 入度集合
    in_nei = {}
    # 出度集合
    out_nei = {}
    nei = {}
    for u in nodes:
        in_nei[u] = set(graph.predecessors(u))
        out_nei[u] = set(graph.successors(u))
        nei[u] = in_nei[u] | out_nei[u]
    seeds_set = set()
    total_cost = 0

    while True:
        rank = {}.fromkeys(list(nodes), 1.0)
        count = 20
        while count:
            diff = 0
            for u in nodes.difference(seeds_set):
                nei_spread = 0
                un_act_prob = 1
                for v in in_nei[u] & seeds_set:
                    un_act_prob *= (1 - act_prob[(v, u)])
                for v in out_nei[u].difference(seeds_set):
                    nei_spread += act_prob[(u, v)] * rank[v]
                rank_new = un_act_prob * (1 + 0.7 * nei_spread)
                if abs(rank_new - rank[u]) > diff:
                    diff = abs(rank_new - rank[u])
                rank[u] = rank_new
            if diff < 0.0001:
                break
            count -= 1

        seed, _ = max(rank.items(), key=lambda x: x[1])
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
        seeds_set.add(seed)

    print("irie_rank done!")
    return set(seeds_set)


def MIC(graph, act_prob, values, cost, bound_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    rank = {}
    un_act_prob = {}
    act_ability = {}
    total_cost = 0
    first_time = True
    while True:
        if first_time:
            for u in graph_back:
                un_act_prob[u] = 1
                act_ability[u] = 1
                for v in graph_back.successors(u):
                    act_ability[u] += act_prob[(u, v)]
                rank[u] = (un_act_prob[u] * act_ability[u]) / cost[u]
            first_time = False

        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] > bound_cost:
            break
        total_cost += (cost[seed])
        seeds_set.add(seed)
        # Gb.remove_node(g)
        rank[seed] = -1
        change_nodes = set()
        for v in graph_back.successors(seed):
            un_act_prob[v] *= (1 - act_prob[(seed, v)])
            change_nodes.add(v)
        for v in graph_back.predecessors(seed):
            act_ability[v] -= act_prob[(v, seed)]
            change_nodes.add(v)
        for v in change_nodes.difference(seeds_set):
            rank[v] = (un_act_prob[v] * act_ability[v]) / cost[v]

    print("MIC done!")
    return seeds_set


def MCF(graph, act_prob, values, costs, bound_cost):
    seeds_set = set()
    total_cost = 0

    for u in sorted(costs.items(), key=lambda x: x[1]):
        if total_cost + u[1] > bound_cost:
            return seeds_set
        seeds_set.add(u[0])
        total_cost += u[1]


def random_seed(graph, act_prob, values, costs, bound_cost):
    random.seed()
    seeds_set = set()
    total_cost = 0

    h = sorted(costs.items(), key=lambda x: x[1])
    length = len(h)
    while True:
        index = random.randint(0, length - 1)
        seed = h[index]
        if seed[0] in seeds_set:
            continue
        if total_cost + seed[1] > bound_cost:
            return seeds_set
        seeds_set.add(seed[0])
        total_cost += seed[1]


def MGF(graph, act_prob, values, cost, bound_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    rank = {}
    un_act_prob = {}
    act_gain = {}
    total_cost = 0
    first_time = True
    while True:
        if first_time:
            for u in graph_back:
                un_act_prob[u] = 1
                act_gain[u] = values[u]
                for v in graph_back.successors(u):
                    act_gain[u] += act_prob[(u, v)] * values[v]
                rank[u] = un_act_prob[u] * act_gain[u]
            first_time = False

        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] > bound_cost:
            break
        total_cost += cost[seed]
        seeds_set.add(seed)
        # Gb.remove_node(g)
        rank[seed] = -1
        change_nodes = set()
        for v in graph_back.successors(seed):
            un_act_prob[v] *= (1 - act_prob[(seed, v)])
            change_nodes.add(v)
        for v in graph_back.predecessors(seed):
            act_gain[v] -= act_prob[(v, seed)] * values[seed]
            change_nodes.add(v)
        for v in change_nodes.difference(seeds_set):
            rank[v] = un_act_prob[v] * act_gain[v]

    print("MGF done!")
    return seeds_set


def MGCRF(graph, act_prob, values, cost, bound_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    rank = {}
    un_act_prob = {}
    act_gain = {}
    total_cost = 0
    first_time = True
    while True:
        if first_time:
            for u in graph_back:
                un_act_prob[u] = 1
                act_gain[u] = values[u]
                for v in graph_back.successors(u):
                    act_gain[u] += act_prob[(u, v)] * values[v]
                rank[u] = (un_act_prob[u] * act_gain[u]) / cost[u]
            first_time = False

        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] > bound_cost:
            break
        total_cost += (cost[seed])
        seeds_set.add(seed)
        # Gb.remove_node(g)
        rank[seed] = -1
        change_nodes = set()
        for v in graph_back.successors(seed):
            un_act_prob[v] *= (1 - act_prob[(seed, v)])
            change_nodes.add(v)
        for v in graph_back.predecessors(seed):
            act_gain[v] -= act_prob[(v, seed)] * values[seed]
            change_nodes.add(v)
        for v in change_nodes.difference(seeds_set):
            rank[v] = (un_act_prob[v] * act_gain[v]) / cost[v]

    print("MGCRF done!")
    return seeds_set


def test_algo(graph, act_prob, values, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = dict()
    rank = {}
    un_act_prob = {}
    act_gain = {}
    values_weight = {}
    total_cost = 0
    first_time = True
    while True:
        if first_time:
            for u in graph_back:
                fail_prob = 1
                for v in graph_back.successors(u):
                    fail_prob *= 1 - act_prob[(u, v)]
                values_weight[u] = 1 - fail_prob

            for u in graph_back:
                un_act_prob[u] = 1
                act_gain[u] = values[u] * values_weight[u]

                for v in graph_back.successors(u):
                    act_gain[u] += act_prob[(u, v)] * values[v] * values_weight[v]
                rank[u] = un_act_prob[u] * act_gain[u]

            first_time = False

        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
        seeds_set[seed] = 1
        # Gb.remove_node(g)
        rank[seed] = -1
        change_nodes = set()
        for v in graph_back.successors(seed):
            un_act_prob[v] *= (1 - act_prob[(seed, v)])
            change_nodes.add(v)
        for v in graph_back.predecessors(seed):
            fail_prob = 1
            for w in graph_back.successors(v):
                if w not in seeds_set.keys():
                    fail_prob *= 1 - act_prob[(v, w)]
            values_weight[v] = 1 - fail_prob

            act_gain[v] -= act_prob[(v, seed)] * values[seed] * values_weight[seed]
            change_nodes.add(v)
        for v in change_nodes.difference(set(seeds_set.keys())):
            rank[v] = un_act_prob[v] * act_gain[v]

    print("test_algo done!")
    return seeds_set


def test_algo2(graph, act_prob, values, cost, bound_cost, fixed_cost):
    nodes = set(graph.nodes())
    # 入度集合
    in_nei = {}
    # 出度集合
    out_nei = {}
    nei = {}
    for u in nodes:
        in_nei[u] = set(graph.predecessors(u))
        out_nei[u] = set(graph.successors(u))
        nei[u] = in_nei[u] | out_nei[u]
    seeds = dict()
    total_cost = 0

    graph_back = copy.deepcopy(graph)
    g = {}
    f = {}
    for u in graph_back:
        fail_prob = 1
        for v in graph_back.successors(u):
            fail_prob *= 1 - act_prob[(u, v)]
        g[u] = 1 - fail_prob
        f[u] = values[u] * g[u]

    while True:
        rank = {}.fromkeys(list(nodes), 1.0)
        count = 5
        while count:
            seeds_set = set(seeds.keys())
            f_temp = copy.deepcopy(f)
            diff = 0
            for u in nodes.difference(seeds_set):
                nei_spread = 0
                un_act_prob = 1

                for v in in_nei[u] & seeds_set:
                    un_act_prob *= (1 - act_prob[(v, u)])

                for v in out_nei[u].difference(seeds_set):
                    nei_spread += act_prob[(u, v)] * f_temp[v]

                f[u] += nei_spread

                rank_new = un_act_prob * f[u]
                if abs(rank_new - rank[u]) > diff:
                    diff = abs(rank_new - rank[u])
                rank[u] = rank_new
            if diff < 0.0001:
                break
            count -= 1

        seed, _ = max(rank.items(), key=lambda x: x[1])
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
        seeds[seed] = 1

    print("irie_rank done!")
    return seeds


def MREF(graph, act_prob, values, cost, bound_cost):
    nodes = set(graph.nodes())
    # 入度集合
    in_nei = {}
    # 出度集合
    out_nei = {}
    nei = {}
    for u in nodes:
        in_nei[u] = set(graph.predecessors(u))
        out_nei[u] = set(graph.successors(u))
        nei[u] = in_nei[u] | out_nei[u]
    seeds = set()
    total_cost = 0

    graph_back = copy.deepcopy(graph)
    f = update(graph_back, nodes, out_nei, act_prob, values, seeds)

    mark = {x: True for x in range(1, 10)}
    rank = {}.fromkeys(list(nodes), 1.0)
    while True:
        for u in nodes:
            un_act_prob = 1

            for v in in_nei[u] & seeds:
                un_act_prob *= (1 - act_prob[(v, u)])

            rank[u] = un_act_prob * f[u]

        seed, _ = max(rank.items(), key=lambda x: x[1])
        if total_cost + cost[seed] > bound_cost:
            break
        total_cost += (cost[seed])

        index = int(10 * total_cost / bound_cost)
        if index in mark.keys() and mark[index]:
            f = update(graph_back, nodes, out_nei, act_prob, values, seeds)
            mark[index] = False

        seeds.add(seed)
        rank[seed] = -1
        nodes = nodes.difference({seed})

    print("MREF done!")
    return seeds


def update(graph, nodes, out_nei, act_prob, values, seeds):
    g = {}
    f = {}
    seeds_set = seeds
    for u in graph:
        fail_prob = 1
        for v in graph.successors(u):
            if v not in seeds_set:
                fail_prob *= 1 - act_prob[(u, v)]
        g[u] = 1 - fail_prob
        f[u] = values[u] * g[u]

    count = 3
    while count:
        f_temp = copy.deepcopy(f)
        for u in nodes:
            nei_spread = 0

            for v in out_nei[u]:
                if v not in seeds_set:
                    nei_spread += act_prob[(u, v)] * f_temp[v]

            f[u] += nei_spread
        count -= 1
    return f

