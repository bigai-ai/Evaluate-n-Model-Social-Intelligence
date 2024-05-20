"""
Generate regional figures for multiple examples.
"""
import json
import pickle
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import os
import torch
import signal_model_torch as S
from regression import parse_json_prompt, construct_board, calculate
from figures import *

if __name__ == '__main__':
    DICT = {"shortest": 1, "far": 2, "best": 3, "misleading": 0}
    prefix = "../data/regions/"
    files = list(os.walk(prefix))[0][2]
    assert len(files) == 500, "Please run file `./reference_regions.py` first!"

    with open("../raw_data/IIP_500_2_zero_shot_test_10231552_extract.json",
              "r",
              encoding="utf8") as fp:
        problems = json.load(fp)

    try:
        with open("../data/region_average.pkl", "rb") as fp:
            post_sum = pickle.load(fp)
    except:
        post_sum = np.zeros([200, 200, 4], dtype=float)
        for fname in files:
            with open(prefix + fname, "rb") as fp:
                data = pickle.load(fp)

            gt = problems[fname[:-4]]["gt"]
            # print(gt)
            for i in range(4):
                post_sum[:, :, i] += data[(0.99, 100)][:, :, DICT[gt[i]]]

        post_sum /= 500
        with open("../data/region_average.pkl", "wb") as fp:
            pickle.dump(post_sum / 500, fp)

    draw_model_regions_iip(
        0.5,
        100,
        {(0.5, 100): post_sum},
        loc={
            'Shortest': [(30, 90), 90, 'x-large', 2],
            'Reversed': [(100, 100), 90, 'x-large', 0],
            'Hybrid': [(80, 20), 0, 'x-large', 1],
            'Avoidant': [(180, 110), 90, 'x-large', 3]
        },
    )
