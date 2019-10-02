# Author: cym
import random
from copy import deepcopy


def run_directed_IC(graph, seeds_set, act_prob, values):
    random.seed()
    target_set = deepcopy(seeds_set)  # targeted set
    new_seeds = deepcopy(seeds_set)
    while len(new_seeds):  # while we have newly activated nodes
        seeds_tmp = set()
        for u in new_seeds:
            for v in set(graph.successors(u)):  # v is the out neighbor of u
                if random.random() < act_prob[(u, v)] and v not in target_set:
                    seeds_tmp.add(v)
                    target_set.add(v)
        new_seeds = deepcopy(seeds_tmp)

    gain = sum([values[x] for x in target_set])
    return gain, len(target_set)


def avg_IC(graph, seeds_set, act_prob, values, iterations):
    score = 0
    count = 0
    for j in range(iterations):
        gain, n = run_directed_IC(graph, seeds_set, act_prob, values)
        score += gain
        count += n
    return score/iterations, count/iterations








