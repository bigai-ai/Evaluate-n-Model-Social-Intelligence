"""
Generate figures
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import os
import torch
import signal_model_torch as S
from regression import parse_json_prompt, construct_board, calculate


def generating_likelihood_sections_iip(gmap,
                                       routes,
                                       regions_log,
                                       params=[(0.3, 1), (0.99, 1), (0.3, 100),
                                               (0.99, 100)],
                                       resolution=200):

    for thres, pulse in params:
        results = np.zeros([resolution, resolution, 4])
        # thres, pulse = 0.99, 1000

        for i, a in enumerate(
                np.linspace(0, 1, resolution, endpoint=False) +
                0.5 / resolution):
            for j, b in enumerate(
                    np.linspace(0, 1, resolution, endpoint=False) +
                    0.5 / resolution):
                # print(a, b)

                iip = S.IIP(torch.tensor([a, b, thres, pulse]), [1, 2], routes,
                            gmap)
                results[i, j, :] = iip.calculate(2, 1).reshape(1, 1, -1)

        fig, ax = plt.subplots(1, 4)
        for i in range(4):
            ax[i].imshow(results[:, :100, i], vmin=0, vmax=1)

        regions_log[(thres, pulse)] = results.copy()
    return regions_log


def draw_model_regions_iip(
        thres,
        pulse,
        regions_log,
        loc={
            'Shortest': [(30, 90), 90, 'x-large', 2],
            'Reversed': [(100, 100), 90, 'x-large', 0],
            'Hybrid': [(80, 20), 0, 'x-large', 1],
            'Avoidant': [(180, 110), 90, 'x-large', 3]
        },
        prefix="../figures",
        resolution=200):
    cs = [
        np.array([134, 197, 198]),
        np.array([253, 96, 78]),
        np.array([158, 171, 174]),
        np.array([202, 220, 111]),
    ]
    results = regions_log[(thres, pulse)]

    combined = np.zeros([resolution, resolution, 2], dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            temp = results[i, j, :].copy()
            combined[i, j, 0] = np.argmax(temp)
            mtemp = np.max(temp)
            temp[int(combined[i, j, 0])] = 0.
            stemp = np.max(temp)
            combined[i, j, 1] = mtemp - stemp

    fig = plt.figure(figsize=[3.5, 3])
    ax = fig.subplots()
    c = ax.contourf(combined[:, :, 1].T,
                    levels=np.linspace(np.min(combined[:, :, 1]),
                                       np.max(combined[:, :, 1]), 400))

    borders = np.zeros([resolution, resolution, 4], dtype=float)
    for k in range(4):
        a = [0, 1, 2, 3]
        a.remove(k)
        borders[:, :, k] = results[:, :, k] - np.max(results[:, :, a], axis=2)
        ax.contour(borders[:, :, k].T,
                   levels=[0],
                   colors=['white', 'yellow', 'orange', 'red'][0:1])

    for key, val in loc.items():
        ax.text(*val[0],
                key,
                fontsize=val[2],
                rotation=val[1],
                c=cs[int(val[3])] / 255)

    ax.set_xticks([0, 99.5, 199])
    ax.set_xticklabels([0, "$exp(-\\alpha)$", 1])
    ax.set_yticks([0, 99.5, 199])
    ax.set_yticklabels([0, "$exp(-\\beta)$", 1],
                       rotation=90,
                       verticalalignment='center')
    fig.colorbar(c, ticks=np.linspace(0, 1, 10, endpoint=False))
    fig.tight_layout()
    fig.savefig(f"{prefix}/model_regions_{thres}_{pulse}.png", dpi=600)


def generate_iip_strategy_selection_regions(example):
    """
    Data preparation for Figure 6.
    """
    gmap, routes, gt, best_trj_type = parse_json_prompt(example)
    regions_log = {}
    for thres, pulse in [(0.3, 1), (0.99, 1), (0.3, 100), (0.99, 100)]:
        resolution = 200
        results = np.zeros([resolution, resolution, 4])
        # thres, pulse = 0.99, 1000
        print(thres, pulse)
        for i, a in enumerate(
                np.linspace(0, 1, resolution, endpoint=False) +
                0.5 / resolution):
            for j, b in enumerate(
                    np.linspace(0, 1, resolution, endpoint=False) +
                    0.5 / resolution):
                # print(a, b)

                iip = S.IIP(torch.tensor([a, b, thres, pulse]), [1, 2], routes,
                            gmap)
                results[i, j, :] = iip.calculate(2, 1).reshape(1, 1, -1)

        fig, ax = plt.subplots(1, 4)
        for i in range(4):
            ax[i].imshow(results[:, :100, i], vmin=0, vmax=1)

        regions_log[(thres, pulse)] = results.copy()
    return regions_log


# draw_model_regions_iip(0.99, 100)

if __name__ == '__main__':

    EXAMPLE = {
        'options': {
            'A':
            '''W****\nW*W**\nWXW**\nW*W**\nY**A*\n
                    (3, 4) (2, 4) (1, 4), (0, 4), (1, 4).(1, 3), (1, 2), ''',
            'B':
            '''W****\nW*W**\nWXW**\nW*W**\nY**A*\n
                    (3, 4) (2, 4) (1, 4), (1, 3), (1, 2), ''',
            'C':
            '''W****\nW*W**\nWXW**\nW*W**\nY**A*\n
                    (3, 4) (3, 3) (3, 2), (3, 1), (3, 0) (2, 0), (1, 0), (1, 1), (1, 2)''',
            'D':
            '''W****\nW*W**\nWXW**\nW*W**\nY**A*\n
                    (3, 4) (3, 3) (3, 4), (2, 4) (1, 4), (1, 3), (1, 2)''',
        },
        'gt': ['midleading', 'shortest', 'far', 'best'],
        'best_trj_type': 2
    }

    try:
        with open("../data/temp/model_regions.dat", "rb") as fp:
            regions_log = pickle.load(fp)["regions_log"]

    except:
        print("Calculating Example...")
        regions_log = generate_iip_strategy_selection_regions(EXAMPLE)
        with open("../data/temp/model_regions.dat", "wb") as fp:
            pickle.dump({"regions_log": regions_log}, fp)

    draw_model_regions_iip(0.99, 100, regions_log)

    draw_model_regions_iip(
        0.99,
        1,
        regions_log,
        loc={
            'Shortest': [(75, 100), 90, 'x-large', 2],
            'Hybrid': [(110, 60), 0, 'x-large', 1],
            'Avoidant': [(180, 105), 90, 'x-large', 3]
        },
    )
    draw_model_regions_iip(
        0.3,
        1,
        regions_log,
        loc={
            'Shortest': [(100, 100), 90, 'x-large', 2],
            'Hybrid': [(120, 20), 0, 'x-large', 1],
            'Avoidant': [(185, 85), 90, 'x-large', 3]
        },
    )

    draw_model_regions_iip(
        0.3,
        100,
        regions_log,
        loc={
            'Shortest': [(60, 80), 90, 'x-large', 2],
            'Reversed': [(120, 80), 90, 'x-large', 0],
            'Hybrid': [(100, 20), 0, 'x-large', 1],
            'Avoidant': [(170, 80), 90, 'x-large', 3]
        },
    )
