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

with open("../raw_data/IIP_500_2_zero_shot_test_10231552_extract.json",
          "r",
          encoding="utf8") as fp:
    problems = json.load(fp)

if os.path.exists("../data/regions/"):
    os.mkdir("../data/regions/")


def single(problem):
    print("Processing problem:", problem[0])
    regions_log = generate_iip_strategy_selection_regions(problem[1])
    with open("../data/regions/" + problems[0] + ".pkl", "wb") as fp:
        pickle.dump(regions_log, fp)


if __name__ == '__main__':
    with mp.Pool() as pool:
        pool.map(single, problems.items())
