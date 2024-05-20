"""
Regression.
"""
import numpy as np
from strategies import hybrid
from grid import GridMap
import re
import json
import torch


def construct_board(obstacle, silos, origin, size=[5, 5]):
    gmap = GridMap(*size)
    for i, s in enumerate(silos):
        gmap.set_silo(i + 1, s)
    gmap.set_origin(origin)
    gmap.set_blocks(obstacle)
    return gmap


def parse_json_prompt(prompt):
    i, j = 0, 0
    gmap = GridMap(5, 5)
    blocks = []
    human_flag = True
    if 'human_prompt' not in prompt.keys():
        human_flag = False
        prompt['human_prompt'] = [[], prompt['options']['A'][:30]]
    if len(prompt['human_prompt'][0]) == 0:
        human_flag = False
    for k in prompt['human_prompt'][1]:
        if k == '\n':
            i = 0
            j += 1
            continue
        elif k == 'A':
            gmap.set_origin((i, j))
            print("A", i, j)
        elif k == 'X':
            gmap.set_silo_a((i, j))
        elif k == 'Y':
            gmap.set_silo_b((i, j))
        elif k == 'W':
            blocks += [(i, j)]

        i += 1
    gmap.set_blocks(blocks)
    routes = [None] * 4
    i = -1

    def path_parse(string):
        route = []
        for x in re.finditer("(\([0-9], [0-9]\))", string):
            pos = (int(x.group(0)[1]), int(x.group(0)[4]))
            if len(route) > 0 and pos == route[-1]:
                continue
            route += [pos]
        return route

    if not human_flag:
        routes = [
            path_parse(prompt['options'][t]) for t in ['A', 'B', 'C', 'D']
        ]
        gt = prompt['gt']
        best_trj_type = prompt['best_trj_type']
        return gmap, routes, gt, best_trj_type

    for x in re.finditer("(\([0-9], [0-9]\))|(Route )",
                         prompt['human_prompt'][2]):
        # print(x.group(0))
        if x.group(0) == "Route ":
            i += 1
            continue
        pos = (int(x.group(0)[1]), int(x.group(0)[4]))
        if routes[i] is None:
            routes[i] = [pos]
            continue
        if routes[i][-1] != pos:
            routes[i] += [pos]

    return gmap, routes


def calc_torch(
    gmap,
    routes,
    params,
):
    ea, eb, th, pulse = params
    m = hybrid.StrategyHybrid(gmap)
    m.coloring_saturated()
    t = torch.zeros([5, 5])
    s = torch.zeros([25, 2])


def calculate(gmap, routes, alpha, beta):
    # ea = np.exp(-alpha)
    # eb = np.exp(-beta)
    ea = alpha
    eb = beta
    m = hybrid.StrategyHybrid(gmap)
    m.coloring_saturated()
    t = np.zeros([5, 5])
    s = np.zeros([25, 2])
    dictionary = []
    for i in range(5):
        for j in range(5):
            x = m.color_board[(i, j)]
            t[i, j] = (1 - x[1]) * (1 if x[0] > 0.99 else
                                    (-1 if x[2] > 0.99 else 0))
            if x[0] > 0.99:
                s[len(dictionary), 0] = (1 - x[1])
                dictionary += [(i, j)]
            elif x[2] > 0.99:
                s[len(dictionary), 1] = (1 - x[1])
                dictionary += [(i, j)]

    likelihood = np.zeros([4, 2])
    prior = []
    i = -1
    for route in routes:
        i += 1
        g = 1
        for pos in route:
            likelihood[i, (0 if t[pos] > 0 else 1)] += g
            g *= ea
        prior += [eb**len(route)]

    L1 = likelihood.T / np.sum(likelihood, axis=1)
    L2 = L1.T / np.sum(L1, axis=1)

    post = (L2.T * (prior / np.sum(prior)))[0]
    return post / np.sum(post)
