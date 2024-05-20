"""
"""
import json
import pickle
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import os
import torch
import signal_model_torch as S
from regression import parse_json_prompt, construct_board, calculate
from figures import *


def visualize(key_index, rdata):
    name = list(data.keys())[key_index]
    return visualize_(data[name], rdata, name)[0], name, data[name]


def aggregation(human_list, problems):
    ret = []
    for d in human_list:
        probid = d['image_id']
        problem = problems[probid]
        ret += [[d, problem]]
    return ret


def visualize_(ds, rdata, name="Null", resolution=100):
    agg_data = aggregation(ds, problems)
    result = np.zeros([resolution, resolution], dtype=float)
    for d in agg_data[:]:
        human, problem = d
        uo = human['user_option']
        po = human['image_id']
        index = problem['gt'].index(uo)
        p_index = rdata['problem_ids'].index(po)
        fig = np.log(rdata['data'][p_index][:, :, index])
        result += fig

    result = result / len(ds)
    return result, name, ds


def draw_likelihood(total,
                    rdata,
                    results,
                    filename="allhuman",
                    prefix="../figures/"):
    """
    Draw figrue for all human data.
    """
    res, name, d = visualize_(total, rdata)
    fig = plt.figure(figsize=np.array([1, 1]) * 3)
    ax = fig.subplots()
    N = np.linspace(np.min(np.exp(res)), np.max(np.exp(res)), 30)
    N2 = np.linspace(np.min(np.exp(res)), np.max(np.exp(res)), 300)
    ax.contour(np.exp(res).T, N, cmap=cm.Reds, alpha=0.3)
    ax.contourf(np.exp(res).T, N2)

    resolution = 100
    borders = np.zeros([resolution, resolution, 4], dtype=float)
    for k in range(4):
        a = [0, 1, 2, 3]
        a.remove(k)
        borders[:, :, k] = results[:, :, k] - np.max(results[:, :, a], axis=2)
        ax.contour(borders[:, :, k].T,
                   levels=[0],
                   colors=['white', 'yellow', 'orange', 'red'][0:1])

    loc = {
        'Shortest': [(20, 90), 90, 'x-large', 2],
        'Reversed': [(120, 115), 90, 'x-large', 0],
        'Hybrid': [(120, 20), 0, 'x-large', 1],
        'Avoidant': [(180, 110), 90, 'x-large', 3]
    }
    csz = [np.array([255, 50, 50])] * 4
    [
        np.array([34, 37, 98]),
        np.array([153, 36, 18]),
        np.array([208, 221, 224]),
        np.array([52, 70, 11]),
    ]
    for key, val in loc.items():
        ax.text(*np.array(val[0]) / 2,
                key,
                fontsize=val[2],
                rotation=val[1],
                c=csz[int(val[3])] / 255)
    # plt.imshow(np.exp(res))
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 99)
    ax.set_xticks([0, 49.5, 99])
    ax.set_xticklabels([0, '$exp(-\\alpha)$', 1])
    ax.set_yticks([0, 49.5, 99])
    ax.set_yticklabels([0, '$exp(-\\beta)$', 1], rotation=90)

    fig.tight_layout()
    fig.savefig(f"{prefix}fixphi_{filename}.png", dpi=600)


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


if __name__ == '__main__':

    try:
        with open("../data/IIP_rdata_THRES_0.99_PULSE_100.dat", 'rb') as fp:
            rdata = pickle.load(fp)
    except:
        print(
            "Please run `./param_data_gen.py` first for generating data file.")

    df = pd.read_excel("../raw_data/human_study_iip.xlsx")

    data = {}
    for i in range(len(df)):
        name = df.loc[i]['name']
        if name not in data.keys():
            data[name] = []

        tmp = {}
        tmp['image_id'] = df.loc[i]['image_id']
        tmp['text_or_image'] = df.loc[i]['text_or_image']
        tmp['one_shot'] = df.loc[i]['one_shot']
        tmp['user_option'] = df.loc[i]['user_option']
        tmp['question_id'] = df.loc[i]['question_id']
        tmp['user_steps'] = df.loc[i]['user_steps']
        tmp['gt_steps'] = df.loc[i]['gt_steps']
        data[name] += [tmp]

    total = []
    for v in data.values():
        total += v

    DICT = {"shortest": 1, "far": 2, "best": 3, "misleading": 0}
    prefix = "../data/regions/"
    files = list(os.walk(prefix))[0][2]
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

    draw_likelihood(total, rdata, post_sum[::2, ::2, :])

    ###########################################################
    df = pd.read_excel("../raw_data/gpt_2023112016_iip.xlsx")
    keys = []
    for row in df:
        keys += [row]

    models = keys[3::2]
    model = 'gpt-4-1106-preview_output_res_option'
    model_data = {}
    for m in models:
        model_data[m] = []
    for row in df.to_numpy():
        for i, m in enumerate(models):
            # print(row)
            if row[2 * i + 3] not in ['best', 'shortest', 'misleading', 'far']:
                continue
            model_data[m] += [{
                "image_id": row[0],
                "user_option": row[2 * i + 3]
            }]

    draw_likelihood(model_data[model], rdata, post_sum[::2, ::2, :], model)
