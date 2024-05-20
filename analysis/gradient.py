"""

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import json
import torch
import torch.nn.functional as F
import signal_model_torch as S
import pickle
from regression import *
import pandas
from torch import optim

REGRESSION_HISTORY = {}

LOG_FP = open("../logs/temp_.log", "w")


def get_iip_problems():
    with open("../raw_data/IIP_500_2_zero_shot_test_10231552_extract.json",
              "r") as fp:
        problems = json.load(fp)

    return problems


def get_human_data():
    df = pandas.read_excel("../raw_data/human_study_iip.xlsx")
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

    return data


def get_llm_data():
    df = pandas.read_excel("../raw_data/gpt_2023112016_iip.xlsx")
    keys = []
    for row in df:
        keys += [row]
    actions = ['best', 'shortest', 'misleading', 'far']
    models = keys[3::2]
    model_data = {}
    for m in models:
        model_data[m] = []
    for row in df.to_numpy():
        for i, m in enumerate(models):
            # print(row)
            if row[2 * i + 3] not in actions:
                continue
            model_data[m] += [{
                "image_id": row[0],
                "user_option": row[2 * i + 3]
            }]

    return models, model_data


HUMAN_DATA = get_human_data()
PROBLEMS = get_iip_problems()
MODELS, MODEL_DATA = get_llm_data()


def aggregation(human_list, problems):
    ret = []

    for d in human_list:

        probid = d['image_id']
        problem = problems[probid]
        ret += [[d, problem]]
    return ret


class Regression:
    def __init__(self,
                 agg_data,
                 init=[0.5, 0.5, 0.5, 0.5],
                 mask=(None, None, None, None),
                 **args):
        problems = PROBLEMS
        self.args = args
        self.mask = mask
        self.agg_data = agg_data
        self.params = torch.tensor(init, dtype=float, requires_grad=True)
        self.aug_params = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.],
                                       dtype=float,
                                       requires_grad=True)
        # self.optimizer = optim.SGD([self.params, self.aug_params], lr=0.001)

        self.optimizer = args.get(
            "optimizer", optim.SGD([self.params, self.aug_params], lr=0.001))
        self.params_ = torch.tensor([
            self.params[0], self.params[1], self.params[2],
            -torch.log(self.params[3])
        ])
        self.models = []
        self.selections = []
        self.logs = []
        self.history = []
        self.gmaps = []
        self.routes = []
        for data in agg_data:
            human, problem = data
            gmap, routes, gt, best_trj_type = parse_json_prompt(problem)
            inv = dict((v, i) for i, v in enumerate(gt))
            self.gmaps += [gmap]
            self.routes += [routes]
            self.models += [
                S.IIP(self.params_, [1, 2], routes, gmap, method='exp')
            ]
            self.logs += [self.models[-1].log]
            self.selections += [inv[human['user_option']]]
        self.vol = len(self.models)

    def single_step(self):

        self.loss = self.step()
        self.loss.backward()
        self.optimizer.step()
        #         for i in range(4):
        #             if self.params[i] < 0:
        #                 self.params[i] = 0

        print("Loss:", self.loss)
        print("Gradient: ", self.params.grad, torch.sum(self.params.grad**2))
        print("Params: ", self.params)

        LOG_FP.write("Loss: " + str(self.loss))
        LOG_FP.write("Gradient: " + str(self.params.grad) +
                     str(torch.sum(self.params.grad**2)))
        LOG_FP.write("Params: " + str(self.params))

        if torch.any(self.params < 1e-7) or torch.any(self.params > 1 - 1e-7):
            self.params = torch.tensor(F.relu(self.params - 1e-7) + 1e-7,
                                       dtype=float,
                                       requires_grad=True)
            self.params = torch.tensor(1 - F.relu(1 - self.params - 1e-7) -
                                       1e-7,
                                       dtype=float,
                                       requires_grad=True)
            forced_same = True
            self.optimizer = self.args.get(
                "optimizer", optim.SGD)([self.params, self.aug_params],
                                        lr=0.001)
            print("BOUNDED")
        else:
            self.history += [[
                self.params.clone().detach(),
                self.params.grad.clone().detach(),
                self.loss.clone().detach()
            ]]
            forced_same = False
        print("=" * 80)
        return forced_same

    def step(self):
        self.params_ = torch.concat([
            self.params[:1]
            if self.mask[0] is None else torch.tensor([self.mask[0]]),
            self.params[1:2]
            if self.mask[1] is None else torch.tensor([self.mask[1]]),
            self.params[2:3]
            if self.mask[2] is None else torch.tensor([self.mask[2]]),
            -torch.log(self.params[3:])
            if self.mask[3] is None else torch.tensor([self.mask[3]]),
        ])

        ret = torch.tensor([0.], dtype=float)
        for i in range(len(self.models)):
            self.models[i] = S.IIP(self.params_, [1, 2],
                                   self.routes[i],
                                   self.gmaps[i],
                                   method='exp',
                                   log=self.logs[i])
        for selection, model in zip(self.selections, self.models):
            ret += torch.log(model.calculate(
                2, 1)[selection])  # Fix level 2 and target 1
        aug = torch.tensor([0.], dtype=float)
        for i in range(4):
            if self.mask[i] is not None:
                aug += self.aug_params[i:i + 1] * self.params[i:i + 1]
                aug += self.aug_params[i + 4:i + 5] * (1 -
                                                       self.params[i:i + 1])

        return -(ret / self.vol)

    def train(self, max_step=1000, epsilon=1e-5):
        last_loss = -1
        nflag = 0
        for _ in range(max_step):
            flag = self.single_step()
            # if flag or
            if torch.isnan(self.loss):
                nflag += 1
                print("NFLAG =", nflag)
            if abs(last_loss - self.loss) < epsilon or nflag > 20:
                break
            last_loss = min(1e7 if torch.isnan(self.loss) else self.loss,
                            last_loss)
        return self.params


def visualize(key_index, rdata, data):
    name = list(data.keys())[key_index]
    return visualize_(data[name], rdata, name)[0], name, data[name]


def visualize_(ds, rdata, name="Null"):
    problems = PROBLEMS
    agg_data = aggregation(ds, problems)
    result = np.zeros([100, 100], dtype=float)
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


def human(index):
    if index < 0:
        total = []
        for v in HUMAN_DATA.values():
            total += v
        return total
    return HUMAN_DATA[list(HUMAN_DATA.keys())[index]]


def train_process(test_data,
                  name="total",
                  init=[0.5, 0.5, 0.5, 0.5],
                  mask=[None, None, None, None],
                  max_step=2000,
                  epsilon=1e-6,
                  **args):
    global REGRESSION_HISTORY
    agg_data = aggregation(test_data, PROBLEMS)
    reg = Regression(agg_data, init, mask, **args)
    reg.train(max_step, epsilon)
    REGRESSION_HISTORY[name] = reg.history
    return name, reg.params, reg


def post_process(raw, rdata, name='unknown', reg=None):
    res, name, d = visualize_(raw, rdata, name)
    # print(name, [x['user_option'] for x in d])
    fig = plt.figure(figsize=[5, 5])
    ax = fig.subplots()
    ax.imshow(np.exp(res))
    # plt.imshow(res)
    xx = []
    yy = []
    if reg is None:
        return
    for i, x in enumerate(reg.history):
        xx += [x.detach().numpy()[0] * 101 - 0.5]
        yy += [x.detach().numpy()[1] * 101 - 0.5]
        ax.scatter(yy[-1], xx[-1], c=cm.Reds(i / len(reg.history)))
    ax.plot(xx, yy)
    return fig


def train_all_human(init=[0.5] * 4,
                    mask=[None] * 4,
                    filename="human_4dim_log",
                    start=0):
    data = {}
    for i in range(start, len(HUMAN_DATA)):
        name = "total" if i < 0 else list(HUMAN_DATA.keys())[i]
        test_data = human(i)
        print(i, name)
        name, result, reg = train_process(test_data, name=name)
        data[name] = [result, reg.history]
        with open(f"../logs/{filename}_{i}.dat", "wb") as fp:
            pickle.dump(data[name], fp)


def train_all_llm(init=[0.5] * 4, mask=[None] * 4, filename="4dim_log"):
    data = {}

    for name in MODELS:
        test_data = MODEL_DATA[name]
        print(name)
        name, result, reg = train_process(test_data, name=name)
        data[name] = [result, reg.history]
        with open(f"../logs/llm_{filename}_{name}.dat", "wb") as fp:
            pickle.dump(data[name], fp)

    for i in range(-1, 0):
        name = "total" if i < 0 else list(HUMAN_DATA.keys())[i]
        test_data = human(i)
        print(i, name)
        name, result, reg = train_process(test_data, name=name)
        data[name] = [result, reg.history]
        with open(f"../logs/human_{filename}_{i}.dat", "wb") as fp:
            pickle.dump(data[name], fp)


if __name__ == '__main__':
    # fig = post_process()
    # fig.savefig(f"../figs/human_{i:02}_{name}")
    # train_all_human()
    train_all_human(mask=[None, None, None, None], filename="4dim-human")
    train_all_llm(mask=[None, None, None, None], filename="4dim--llms")
