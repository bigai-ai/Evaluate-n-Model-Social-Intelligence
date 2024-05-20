"""
Generating the params figures
"""

import multiprocessing as mp
import json
import re
import numpy as np
import pickle
from src import gen_map
from src import signal_model_torch as smt
from src.regression import *

RESOLUTION = 100
THRES = 0.99
PULSE = 100.
METHOD = 'exp'
OUTNAME = f"./data/IIP_rdata_THRES_{THRES}_PULSE_{PULSE}.dat"

with open("./IIP_500_2_zero_shot_test_10231552_extract.json", "r") as fp:
    PROBLEMS = json.load(fp)


def single(args):
    """
    Single Process


    input: an index and a problem

    output: the grid: numpy array [RESOLUTION, RESOLUTION, 4],
                      the 4 figures.

    """

    resolution = RESOLUTION

    index, problem = args
    print("Start Calculating problem", index)

    gmap, routes, gt, best_trj_type = parse_json_prompt(problem)

    params = torch.tensor([0., 0., THRES, PULSE], dtype=float)
    iip = smt.IIP(params, [1, 2], routes, gmap, method=METHOD)

    data = np.zeros([resolution, resolution, 4])
    for i, a in enumerate(
            np.linspace(0, 1, resolution, endpoint=False) + 0.5 / resolution):
        for j, b in enumerate(
                np.linspace(0, 1, resolution, endpoint=False) +
                0.5 / resolution):

            iip.ea = a
            iip.eb = b
            iip.gen_basics()
            data[i, j, :] = iip.calculate(2, 1).numpy().copy()

            # data[i, j, :] = calculate(gmap, routes, a, b).reshape(1, 1, -1)

    # fig, ax = plt.subplots(1, 4)
    # for i in range(4):
    #     ax[i].imshow(data[:, :, i], vmin=0, vmax=1)
    print("Done Calculating problem", index)
    return data


def main(test=True):
    if test:
        ids = list(PROBLEMS.keys())[:2]  ################## TEST
    else:
        ids = list(PROBLEMS.keys())
    gts = [PROBLEMS[index]['gt'] for index in ids]
    with mp.Pool() as pool:
        data = pool.map(single, [(index, PROBLEMS[index]) for index in ids])

    with open(OUTNAME, "wb") as fp:
        pickle.dump({"problem_ids": ids, "gts": gts, "data": data}, fp)
    return 0


if __name__ == '__main__':
    main(test=False)
