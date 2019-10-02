# Author: cym
import copy
import math


def knapsack01(cost, value, capacity):
    n = len(cost)
    if n == 0:
        return 0

    memo = [-1 for _ in range(capacity + 1)]
    nodes_set_list = [set() for _ in range(capacity + 1)]
    for i in range(capacity + 1):
        if i >= cost[0]:
            memo[i] = value[0]
            nodes_set_list[i].add(0)
        else:
            memo[i] = 0

    # print("#", 0, nodes_set_list)
    for i in range(1, n):
        for j in range(capacity, cost[i] - 1, -1):

            if memo[j] < value[i] + memo[j - cost[i]]:
                memo[j] = value[i] + memo[j - cost[i]]
                # tmp = copy.deepcopy(SList[j - w[i]])
                tmp = copy.copy(nodes_set_list[j - cost[i]])
                tmp.add(i)
                nodes_set_list[j] = tmp
        # print("#", i, nodes_set_list)


    print("max_value: ", memo[capacity])
    return nodes_set_list[capacity]

# C = 8
# n = [0, 1, 2, 3]
# w = [2, 3, 4, 5]
# v = [3, 4, 5, 6]
#
# print(knapsack01(w, v, C))


def dynamic(graph, act_prob, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    capacity = 10000
    alpha = capacity / bound_cost
    expand_cost = {u: cost[u] * alpha for u in graph_back}
    nodes_list = sorted(cost.keys())
    # gain = copy.deepcopy(cost)
    gain = dict.fromkeys(nodes_list, 0)
    for u in graph_back:
        for v in graph_back.successors(u):
            gain[u] += act_prob[(u, v)] * expand_cost[v]

    expand_fixed_cost = fixed_cost * alpha
    cost_list = [math.ceil(expand_cost[x] + expand_fixed_cost) for x in nodes_list]
    gain_list = [math.ceil(gain[x]) for x in nodes_list]

    nodes_set = knapsack01(cost_list, gain_list, capacity)

    for v in nodes_set:
        seeds_set.add(nodes_list[v])

    print("dynamic done!")
    return seeds_set


def dynamic2(graph, act_prob, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    capacity = 10000
    alpha = capacity / bound_cost
    expand_cost = {u: cost[u] * alpha for u in graph_back}
    nodes_list = sorted(cost.keys())

    # gain = dict.fromkeys(nodes_list, 0)
    gain = dict.fromkeys(nodes_list, 0)
    for u in graph_back:
        for v in graph_back.successors(u):
            gain[u] += act_prob[(u, v)] * expand_cost[v]

    count = 10
    while count:
        count -= 1
        gain_tmp = copy.copy(gain)
        for u in graph_back:
            for v in graph_back.successors(u):
                gain[u] += act_prob[(u, v)] * 0.7 * gain_tmp[v]

    expand_fixed_cost = fixed_cost * alpha
    cost_list = [math.ceil(expand_cost[x] + expand_fixed_cost) for x in nodes_list]
    gain_list = [math.ceil(gain[x]) for x in nodes_list]

    nodes_set = knapsack01(cost_list, gain_list, capacity)

    for v in nodes_set:
        seeds_set.add(nodes_list[v])

    print("dynamic2 done!")
    return seeds_set


def func(graph, act_prob, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    no_use_nodes = set()

    def in_effi(u):
        in_neigh = set(graph_back.predecessors(u)).difference(no_use_nodes).difference(seeds_set)
        in_cost = 0
        unact_prob = 1
        for v in in_neigh:
            in_cost += cost[v] + fixed_cost
            unact_prob *= 1 - act_prob[(v, u)]
        in_gain = (1 - unact_prob) * cost[u]
        if in_cost == 0:
            return 0, in_neigh
        return in_gain / in_cost, in_neigh

    def out_effi(u):
        out_neigh = set(graph_back.successors(u)).difference(no_use_nodes).difference(seeds_set)
        out_cost = cost[u] + fixed_cost
        out_gain = 0
        for v in out_neigh:
            out_gain += act_prob[(u, v)] * cost[v]
        return out_gain / out_cost

    # influence_rank = {}
    # influ_un_act_prob = {}
    # act_ability = {}
    # for u in graph_back:
    #     influ_un_act_prob[u] = 1
    #     act_ability[u] = 1
    #     for v in graph_back.successors(u):
    #         act_ability[u] += act_prob[(u, v)]
    #     influence_rank[u] = influ_un_act_prob[u] * act_ability[u]
    #
    # gain_rank = {}
    # gain_un_act_prob = {}
    # act_gain = {}
    # for u in graph_back:
    #     gain_un_act_prob[u] = 1
    #     act_gain[u] = 0
    #     for v in graph_back.successors(u):
    #         act_gain[u] += act_prob[(u, v)] * cost[v]
    #     gain_rank[u] = gain_un_act_prob[u] * act_gain[u]

    cost_rank = copy.deepcopy(cost)

    total_cost = 0
    while True:

        max_cost_node = max(cost_rank.items(), key=lambda x: x[1])[0]
        # max_influ_node = max(influence_rank.items(), key=lambda x: x[1])[0]
        # max_gain_node = max(gain_rank.items(), key=lambda x: x[1])[0]


        in_effi_value, in_nei = in_effi(max_cost_node)
        out_effi_value = out_effi(max_cost_node)
        if in_effi_value > out_effi_value:
            for u in in_nei:
                if total_cost + cost[u] + fixed_cost > bound_cost:
                    break
                total_cost += cost[u] + fixed_cost
                seeds_set.add(u)
                cost_rank[u] = -1
            no_use_nodes.add(max_cost_node)
        else:
            if total_cost + cost[max_cost_node] + fixed_cost > bound_cost:
                break
            total_cost += cost[max_cost_node] + fixed_cost
            seeds_set.add(max_cost_node)
            cost_rank[max_cost_node] = -1

    print("mixed done!")
    return seeds_set


# def in_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node):
#     in_neigh = set(graph.predecessors(node)).difference(no_use_nodes).difference(seeds_set)
#     in_cost = 0
#     for v in in_neigh:
#         in_cost += cost[v] + fixed_cost
#         unact_prob *= 1 - act_prob[(v, node)]
#     in_gain = (1 - unact_prob) * values[node]
#     if in_cost == 0:
#         return 0, in_neigh
#     return in_gain / in_cost, in_neigh

def in_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node):
    in_neigh = set(graph.predecessors(node)).difference(no_use_nodes).difference(seeds_set)
    in_cost = 0
    in_gain = 0
    for v in in_neigh:
        in_cost += cost[v] + fixed_cost
        unact_prob *= 1 - act_prob[(v, node)]
        for w in set(graph.successors(v)).difference(seeds_set):
            in_gain += act_prob[(v, w)] * values[w]

    in_gain += (1 - unact_prob) * values[node]
    if in_cost == 0:
        return 0, in_neigh
    return in_gain / in_cost, in_neigh


def out_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node):
    out_neigh = set(graph.successors(node)).difference(no_use_nodes).difference(seeds_set)
    out_cost = cost[node] + fixed_cost
    out_gain = 0
    for v in out_neigh:
        out_gain += act_prob[(node, v)] * values[v]

    return out_gain / out_cost


def better_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node):
    in_effi_value, in_nei = in_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node)
    out_effi_value = out_effi(graph, act_prob, fixed_cost, seeds_set, no_use_nodes, unact_prob, values, cost, node)
    if in_effi_value > out_effi_value:
        return [in_effi_value, "nei", in_nei, node]
    else:
        return [out_effi_value, "self", {node}, node]


def func2(graph, act_prob, values, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    cost_rank = copy.deepcopy(cost)
    values_rank = copy.deepcopy(values)
    seeds_set = set()
    no_use_nodes = set()
    un_act_prob = {}.fromkeys(cost.keys(), 1)

    influence_rank = {}
    act_ability = {}

    gain_rank = {}
    act_gain = {}

    # 初始化influence排名和gain排名
    for u in graph_back:
        act_ability[u] = 1
        # act_gain[u] = 0
        act_gain[u] = values[u]
        for v in graph_back.successors(u):
            act_ability[u] += act_prob[(u, v)]
            act_gain[u] += act_prob[(u, v)] * values[v]
        influence_rank[u] = un_act_prob[u] * act_ability[u]
        gain_rank[u] = un_act_prob[u] * act_gain[u]

    total_cost = 0

    cost_count = 0
    influ_count = 0
    gain_count = 0
    while True:
        max_cost_node = max(cost_rank.items(), key=lambda x: x[1])[0]
        max_influ_node = max(influence_rank.items(), key=lambda x: x[1])[0]
        max_gain_node = max(gain_rank.items(), key=lambda x: x[1])[0]
        max_values_node = max(values_rank.items(), key=lambda x: x[1])[0]

        cost_effi = better_effi(graph_back, act_prob, fixed_cost, seeds_set,
                                no_use_nodes, un_act_prob[max_cost_node], values, cost, max_cost_node)

        influ_effi = better_effi(graph_back, act_prob, fixed_cost, seeds_set,
                                 no_use_nodes, un_act_prob[max_influ_node], values, cost, max_influ_node)

        gain_effi = better_effi(graph_back, act_prob, fixed_cost, seeds_set,
                                no_use_nodes, un_act_prob[max_gain_node], values, cost, max_gain_node)

        values_effi = better_effi(graph_back, act_prob, fixed_cost, seeds_set,
                                no_use_nodes, un_act_prob[max_values_node], values, cost, max_values_node)

        better_node = max([cost_effi, influ_effi, gain_effi, values_effi], key=lambda x: x[0])
        if cost_effi[0] >= influ_effi[0]:
            if cost_effi[0] >= gain_effi[0]:
                cost_count += 1
            else:
                gain_count += 1
        else:
            if influ_effi[0] >= gain_effi[0]:
                influ_count += 1
            else:
                gain_count += 1

        # [in_effi_value, "nei", in_nei, node]
        # [out_effi_value, "self", {node}, node]

        if better_node[1] == "nei":
            for u in better_node[2]:
                if total_cost + cost[u] + fixed_cost > bound_cost:
                    break
                total_cost += cost[u] + fixed_cost
                seeds_set.add(u)
                cost_rank[u] = -1
                values_rank[u] = 0
            no_use_nodes.add(better_node[3])
        else:
            if total_cost + cost[better_node[3]] + fixed_cost > bound_cost:
                break
            total_cost += cost[better_node[3]] + fixed_cost
            seeds_set.add(better_node[3])
            cost_rank[better_node[3]] = -1
            values_rank[better_node[3]] = 0

        update_un_act_prob(graph_back, better_node[2], act_prob, un_act_prob)

        update_influence(graph_back, seeds_set, better_node[2],
                         act_prob, un_act_prob, act_ability, influence_rank)
        update_gain(graph_back, seeds_set, better_node[2],
                    act_prob, un_act_prob, act_gain, gain_rank, values)

    print("mixed2 done!")
    print("cost_count:", cost_count, "influ_count:", influ_count, "gain_count:", gain_count)
    counts = sum([cost_count, influ_count, gain_count])
    print("cost_count ratio: ", round(cost_count/counts, 3),
          "influ_count ratio: ", round(influ_count/counts, 3),
          "gain_count ratio: ", round(gain_count/counts, 3))
    return seeds_set


def update_un_act_prob(graph, new_seeds, act_prob, un_act_prob):
    for node in new_seeds:
        for v in graph.successors(node):
            un_act_prob[v] *= (1 - act_prob[(node, v)])


def update_influence(graph, seeds_set, new_seeds, act_prob, un_act_prob, act_ability, rank):
    change_nodes = set()
    for node in new_seeds:
        for v in graph.successors(node):
            change_nodes.add(v)
        for v in graph.predecessors(node):
            act_ability[v] -= act_prob[(v, node)]
            change_nodes.add(v)
        rank[node] = -1
    for v in change_nodes.difference(seeds_set):
        rank[v] = un_act_prob[v] * act_ability[v]


def update_gain(graph, seeds_set, new_seeds, act_prob, un_act_prob, act_gain, rank, values):
    change_nodes = set()
    for node in new_seeds:
        for v in graph.successors(node):
            change_nodes.add(v)
        for v in graph.predecessors(node):
            act_gain[v] -= act_prob[(v, node)] * values[node]
            change_nodes.add(v)
        rank[node] = -1
    for v in change_nodes.difference(seeds_set):
        rank[v] = un_act_prob[v] * act_gain[v]

