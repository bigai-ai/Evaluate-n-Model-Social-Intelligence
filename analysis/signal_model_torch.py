"""
Model of Generating Likelihood and Prior
"""
from grid import GridMap
from strategies import hybrid
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
import torch.nn.functional as F


class RecurBayes:
    """
    Calculate: priors, and likelihood.
    """
    def __init__(self, params: torch.Tensor, phi: callable, hypos: list,
                 paths: list, gmap: GridMap):
        """
        Initialize
        """
        self.ea = params[0]  # ea\in [0,1]
        self.eb = params[1]
        self.phi = phi
        self.hypos = hypos
        self.paths = [np.array(path, dtype=int) for path in paths]
        self.gmap = gmap

    def gen_basics(self):
        "Generate priors"
        self.prior_h = torch.ones(len(self.hypos)) / len(self.hypos)
        self.prior_p = torch.pow(
            self.ea, torch.tensor([len(path) for path in self.paths]))
        self.prior_p = self.prior_p / torch.sum(self.prior_p)
        self.mat = torch.zeros((self.prior_p.shape[0], self.prior_h.shape[0]))
        for ihy, hypo in enumerate(self.hypos):
            for ipa, path in enumerate(self.paths):
                for k in range(1, len(path)):
                    self.mat[ipa, ihy] += self.eb**(k) * self.phi(
                        path[:k + 1], hypo)[0]

    def calculate(self, level: int, ref: list):
        """
        Calculate using recursive Bayesian Inference
        """
        if level == 0:
            return self.prior_p
        ref = np.array(ref, dtype=int)
        mat = self.mat / torch.sum(self.mat, dim=0)

        while level // 2 > 0:
            level -= 2
            mat = mat * self.prior_h
            mat = mat / torch.sum(mat, dim=1).reshape(-1, 1)
            mat = mat * self.prior_p.reshape(-1, 1)
            mat = mat / torch.sum(mat, dim=0)

        if level == 1:  # odd level, ref IS path, last row normalization
            mat = mat * self.prior_h
            mat = mat / torch.sum(mat, dim=1).reshape(-1, 1)
            ref_index = np.argwhere(
                np.array([
                    False if ref.shape != p.shape else np.sum(np.abs(ref -
                                                                     p)) == 0
                    for p in self.paths
                ]))[0][0]
            return mat[ref_index, :]

        ref_index = np.argwhere(
            np.array([np.sum(np.abs(ref - p)) for p in self.hypos]) == 0)[0][0]
        return mat[:, ref_index]


class IIP(RecurBayes):
    """
    IIP
    """
    def __init__(self, param4, hypos, paths, gmap, method="exp", log=None):
        """
        Initialize

        Parameters: alpha, beta, thres, pulse, hypos, paths, gmap
        param4: torch.tensor([1,1,1,1.]) = [ea, eb, thres, pulse]
        """
        self.params = param4
        self.pulse = param4[3]
        self.thres = param4[2]
        self.method = method
        super().__init__(param4[:2], self.phi_iip, hypos, paths, gmap)
        if log is None:
            self.hybrid = hybrid.StrategyHybrid(self.gmap)
            self.hybrid.coloring_saturated()
            self.log = self.hybrid.log
        else:
            self.log = log
        self.colors = {}
        for hypo in hypos:
            self.colors[hypo] = torch.zeros(gmap.board.shape, dtype=float)

        self.gen_basics()

    def gen_basics(self):
        """
        Override the gen_basics in parent class.
        """
        for k, lvl in enumerate(self.log):
            for hypo, val in lvl.items():
                if self.method == "exp":
                    factor = self.thres**k
                elif self.method == "relu":
                    factor = F.sigmoid(1. / k / (1 - self.thres))
                # print(hypo, val)
                for v in val:
                    self.colors[hypo][tuple(v)] = factor

        super().gen_basics()

    def phi_iip(self, path, hypo):
        """
        Phi function: generating signal amplitude of path on hypo
        """
        assert hypo in self.hypos
        ret = torch.tensor([0.], dtype=float)
        if len(path) > 1:
            if (tuple(path[-2]) in self.gmap.silos.values()
                    and tuple(path[-2]) != self.gmap.silos[hypo]):
                # print(ret)
                ret += self.pulse

        ret += self.colors[hypo][tuple(path[-1])]
        # print("PHI_IIP", ret, self.colors, hypo, path[-1])
        return ret
