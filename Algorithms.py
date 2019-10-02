# Author: cym
import copy


def degree_dis(graph, act_prob, cost, bound_cost, fixed_cost):
    graph_back = copy.deepcopy(graph)
    seeds_set = set()
    rank = {}
    total_cost = 0
    while True:
        for u in graph_back:
            un_act_prob = 1
            act_ability = 1
            for v in graph_back.predecessors(u):
                if v in seeds_set:
                    un_act_prob *= (1 - act_prob[(v, u)])
            for v in graph_back.successors(u):
                act_ability += act_prob[(u, v)]
            if u not in rank.keys() or rank[u] != -1:
                rank[u] = un_act_prob * act_ability
        seed = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0][0]
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
        seeds_set.add(seed)
        # Gb.remove_node(g)
        rank[seed] = -1

    print("degree_dis done!")
    return seeds_set


def degree_dis_improved(graph, act_prob, values, cost, bound_cost, fixed_cost):
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
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
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

    print("degree_dis done!")
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


def degree_dis_effic(graph, act_prob, values, cost, bound_cost, fixed_cost):
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
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
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

    print("degree_dis_effic done!")
    return seeds_set


def min_cost(graph, act_prob, values, costs, bound_cost, fixed_cost):
    seeds_set = set()
    total_cost = 0

    for u in sorted(costs.items(), key=lambda x: x[1]):
        if total_cost + u[1] + fixed_cost > bound_cost:
            return seeds_set
        seeds_set.add(u[0])
        total_cost += (u[1] + fixed_cost)


def just_max_values(graph, act_prob, values, costs, bound_cost, fixed_cost):
    seeds_set = set()
    total_cost = 0

    for u in sorted(values.items(), key=lambda x: x[1], reverse=True):
        if total_cost + costs[u[0]] + fixed_cost > bound_cost:
            return seeds_set
        seeds_set.add(u[0])
        total_cost += (costs[u[0]] + fixed_cost)


def degree_dis_gain(graph, act_prob, values, cost, bound_cost, fixed_cost):
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
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
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

    print("degree_dis_gain done!")
    return seeds_set


def degree_dis_gain_effic(graph, act_prob, values, cost, bound_cost, fixed_cost):
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
        if total_cost + cost[seed] + fixed_cost > bound_cost:
            break
        total_cost += (cost[seed] + fixed_cost)
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

    print("degree_dis_gain_effic done!")
    return seeds_set


def rank_values(graph, act_prob, values, cost, bound_cost, fixed_cost):
    seeds_set = set()
    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
    rank = {}
    count = int(0.2 * len(sorted_values))
    i = 0
    while i < count:
        node = sorted_values[i][0]
        for u in graph.predecessors(node):
            if u not in rank.keys():
                rank[u] = act_prob[(u, node)] * values[node]
            else:
                rank[u] += act_prob[(u, node)] * values[node]
        i += 1
    sn = set(rank.keys())
    for u in sorted_values[:count]:
        if u[0] in sn:
            rank[u[0]] += values[u[0]]
    final_rank = {k: v / (cost[k] + fixed_cost) for k, v in rank.items()}
    total_cost = 0
    while True:
        node = max(final_rank.items(), key=lambda x: x[1])[0]
        if total_cost + cost[node] + fixed_cost >= bound_cost:
            break
        total_cost += cost[node] + fixed_cost
        seeds_set.add(node)
        final_rank[node] = 0
        if len(seeds_set) == len(sn):
            for u in sorted(cost.items(), key=lambda x: x[1]):
                if u[0] not in seeds_set:
                    if total_cost + cost[u[0]] + fixed_cost >= bound_cost:
                        break
                    total_cost += cost[u[0]] + fixed_cost
                    seeds_set.add(u[0])
    return seeds_set






