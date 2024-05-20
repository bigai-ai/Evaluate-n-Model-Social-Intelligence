"""
Strategy Hybrid @ ToM GW

J. Wang
"""

import numpy as np
from typing import List
from grid import GridMap
from .base import BaseStrategy


def softmax(v: np.ndarray):
    expv = np.exp(v)
    return expv / np.sum(expv)


class StrategyHybrid(BaseStrategy):
    """Strategy 2: Hybrid."""
    def __init__(self, grid_map: GridMap = None):
        "Initialize."
        super().__init__(grid_map)
        self.dist = {}
        for key, val in self.grid_map.silos.items():
            self.dist[key] = self.grid_map.dist_from_coord(val)

        # self.dist_a = self.grid_map.dist_from_coord(self.grid_map.silo_a)
        # self.dist_b = self.grid_map.dist_from_coord(self.grid_map.silo_b)
        self.light_color = self.grid_map.board.copy()
        self.gen_dist_delta()
        self.color_board = np.ones([self.xlim, self.ylim] + [4], dtype=float)

    def reset_colorboard(self):
        self.color_board = np.ones([self.xlim, self.ylim] + [4], dtype=float)

    def gen_dist_delta(self):
        """Generate distance delta."""
        dist_inter = {}
        self.dist_delta = {}
        for key, val in self.dist.items():
            dist_inter[key] = self.dist_p + val
            dist_inter[key][
                dist_inter[key] >= self.grid_map.INF_THRES] = self.grid_map.INF

            self.dist_delta[key] = dist_inter[key] - val[tuple(
                self.grid_map.get_origin())]

    def gen_regional_dist_delta(self, regions: dict = None) -> dict:
        """
        Parameters
        ----------
        regions: dict : , optional

        Returns
        -------
        out : distance delta to regions
        """
        if regions is None:
            return {}
        dist_inter = {}
        dist = {}
        dist_delta = {}
        origin = self.grid_map.get_origin()
        for key, val in regions.items():
            dist[key] = self.grid_map.dist_from_region(val)
            dist_inter[key] = self.dist_p + dist[key]
            dist_inter[key][
                dist_inter[key] >= self.grid_map.INF_THRES] = self.grid_map.INF

            dist_delta[key] = dist_inter[key] - dist[key][tuple(origin)]

        return dist_delta, dist

    def strict_coloring(self):
        """
        Color the map in odd/even styles.

        Deprecated.
        """
        self.colors = {}
        for this_key in self.dist_delta:
            self.colors[this_key] = np.ones_like(self.dist_p, )
            for key, val in self.dist_delta.items():
                if key == this_key:
                    self.colors[this_key] = np.bitwise_and(
                        self.colors[this_key], np.isclose(val, 0))
                else:
                    self.colors[this_key] = np.bitwise_and(
                        self.colors[this_key],
                        np.bitwise_not(np.isclose(val, 0)))

            # print(this_key, np.argwhere(self.colors[this_key] == 1))
            self.light_color[self.colors[this_key] == 1] = (
                -this_key - self.grid_map.silo_num)

        print(self.light_color)

    def coloring(self, beta: float = 2.):
        """
        Color the map.

        Deprecated.
        """
        self.gen_dist_delta()
        light_region = {x: [] for x in self.grid_map.silos}
        self.silo_keys = sorted(self.grid_map.silos.keys())
        cm = np.array([self.grid_map.COLORMAP[-i] for i in self.silo_keys])
        self.delta_array = np.array([
            self.dist_delta[key] for key in self.silo_keys
        ]).transpose(1, 2, 0)
        for i in range(self.grid_map.xlim):
            for j in range(self.grid_map.ylim):
                if self.grid_map.board[i, j] > self.grid_map.INF_THRES:
                    continue
                spec = softmax(-beta * self.delta_array[i, j, :])
                if np.isclose(np.max(spec), np.min(spec)):
                    continue
                light_region[np.argmax(spec) + 1] += [(i, j)]
                self.color_board[i, j, :] = np.einsum("i,ij->j", spec, cm)
                self.color_board[i,
                                 j, :3] = (self.color_board[i, j, :3] + 1) / 2

        self.grid_map.set_base_colors()
        return light_region

    def weak_coloring(self,
                      regions: dict = None,
                      zeta: float = 0.5,
                      beta: float = 4):
        """
        Color the map in odd/even styles.
        """
        if regions is None:
            regions = {k: [v] for k, v in self.grid_map.silos.items()}
        dist_delta, dist = self.gen_regional_dist_delta(regions)
        # print(dist, dist_delta)
        light_region = {x: [] for x in self.grid_map.silos}

        self.silo_keys = sorted(self.grid_map.silos.keys())
        cm = np.array([self.grid_map.COLORMAP[-i] for i in self.silo_keys])
        cm = cm * (zeta) + (1 - zeta) * np.ones_like(cm)
        delta_array = np.array([dist_delta[key]
                                for key in self.silo_keys]).transpose(1, 2, 0)
        for i in range(self.grid_map.xlim):
            for j in range(self.grid_map.ylim):
                if self.dist_p[i, j] > self.grid_map.INF_THRES:
                    continue
                if any(d_mat[i, j] == 0 for d_mat in dist.values()):
                    continue
                spec = softmax(-beta * delta_array[i, j, :])

                if np.isclose(np.max(spec), np.min(spec)):
                    continue

                light_region[np.argmax(spec) + 1] += [(i, j)]
                self.color_board[i, j, :] = np.einsum("i,ij->j", spec, cm)

        self.grid_map.set_base_colors(board=self.color_board)
        return light_region

    def coloring_saturated(self,
                           regions: dict = None,
                           gamma: float = 0.6,
                           beta: float = 4) -> dict:
        """
        Color the map until it is saturated.

        Parameters
        ----------
        regions: original color regions (i.e. the silos.)

        Return
        ------
        regions: {silo_id: [(colored-tile-coordinates)]}
        """
        zeta = 1.
        self.log = []
        self.spec_color_board = np.ones_like(self.grid_map.board, dtype=float)
        # We need spec_color_beard to be of values [genre, level(alpha)]
        if regions is None:
            original_delta = {x: [v] for x, v in self.grid_map.silos.items()}
        else:
            original_delta = regions
        regions = {x: [] for x in self.grid_map.silos}

        self.grid_map.reset_colorboard()
        self.grid_map.set_base_colors()
        while sum(len(new) for new in original_delta.values()) > 0:
            self.log += [original_delta.copy()]
            for key, val in regions.items():
                val += original_delta[key]
            zeta *= gamma
            original_delta = self.weak_coloring(regions, zeta=zeta, beta=beta)

        return regions

    def level_1_std(self, dest_silo: int, info: dict) -> tuple:
        """
        Select a position as ritual position.

        dest_silo: tuple, true candidate_pos
        info: dict[candidate_pos] = [dist_to_origin,
                                     color_darkness,
                                     dest_silo,
                                     dist_to_silo,]

        Returns
        -------
        (is ritual finished, ritual target point)
        """
        comparison = {
            key: val
            for key, val in info.items() if val[2] == dest_silo
        }
        if len(comparison) == 0:
            # return False, list(info.keys())[0]
            return False, None

        if len(comparison) == 1:
            return True, list(comparison.keys())[0]

        # print("Level_1_std:", comparison.items())
        return True, sorted(comparison.items(),
                            key=lambda x: x[1][3] * 1000 + x[1][1])[0][0]

    def level_2_std(self, info):
        """
        Select a position as ritual position.

        info: dict
        """

    def _path_next_random_sampler(self, x, **kwargs):
        return x[np.random.choice(len(x))]

    def _path_next_greedy(self, x, **kwargs):
        target = kwargs["dest_silo"]
        keygen = lambda x: self.layered_color_board[x[0], x[1], 0] + abs(
            target - self.layered_color_board[x[0], x[1], 1]) * 10000
        print(x, keygen(x))
        return sorted(x, key=keygen)[0]

    def _path_next_greedily_safe(self, x, **kwargs):
        """
        greedily save adjustor: first greedy, then safe (far distance to other colors)
        """
        if len(x) == 1:
            return x[0]
        target = kwargs["dest_silo"]
        tmp = np.zeros_like(self.grid_map.board)
        for silo in self.grid_map.silos:
            if silo == target:
                continue
            tmp += self.grid_map.dist_from_region(self.color_regions[silo])

        def key1(coord):
            next_pos = self._get_next_pos(coord, kwargs['other'])
            return np.max([tmp[r] for r in next_pos])

        key2 = lambda x: self.layered_color_board[x[0], x[1], 0] + abs(
            target - self.layered_color_board[x[0], x[1], 1]) * 10000

        keygen = lambda coord: (-tmp[coord], -key1(coord), key2(coord))
        # print("KEYGEN", x, [keygen(t) for t in x])
        return sorted(x, key=keygen)[0]

    def _get_next_pos(self, cur: tuple, other: tuple):
        distances = self.grid_map.dist_from_coord(other)
        next_pos = self.grid_map.adjacents(cur)
        return [x for x in next_pos if distances[x] == distances[cur] - 1]

    def shortest_path_gen(self,
                          this: tuple,
                          other: tuple,
                          adjustor: callable = None,
                          **kwargs) -> List[tuple]:
        """Generate shortest paths."""

        if adjustor is None:
            adjustor = self._path_next_random_sampler
            if "dest_silo" in kwargs:
                adjustor = self._path_next_greedily_safe

        distances = self.grid_map.dist_from_coord(other)
        ret = [
            this,
        ]
        for i in range(distances[this] - 1, -1, -1):
            cur = ret[-1]
            next_pos = self.grid_map.adjacents(cur)
            next_pos = [x for x in next_pos if distances[x] == i]
            ret += [adjustor(next_pos, other=other, **kwargs)]

        return ret

    def path_generation(self,
                        dest_silo: int,
                        regions: dict = None,
                        gamma: float = 0.6,
                        beta: float = 4,
                        level_1_selection: callable = None,
                        level_2_selection: callable = None) -> tuple:
        """
        Generate Path.
        """
        if dest_silo not in self.grid_map.silos:
            raise Exception("No such destinations!")

        if level_1_selection is None:
            level_1_selection = self.level_1_std
        if level_2_selection is None:
            level_2_selection = self.level_2_std

        self.color_regions = self.coloring_saturated(regions, gamma, beta)
        self.layered_color_board = np.zeros([self.xlim, self.ylim, 2],
                                            dtype=float)
        # [[(layer, target)]*ylim]*xlim

        for k, layer in enumerate(self.log):
            for key, val in layer.items():
                for coord in val:
                    self.layered_color_board[coord[0], coord[1], 0] = k
                    self.layered_color_board[coord[0], coord[1], 1] = key

        # print(self.layered_color_board)
        constructed_path = []
        ritual_finished = False
        dist_away = 0
        # find the signal part of the path.
        while not ritual_finished:
            dist_away += 1
            focus = np.argwhere(self.dist_p == dist_away)
            # print("FOCUS:", focus)
            if focus.shape[0] == 0:
                raise Exception("Player is bouned in a nutshell!")
            if focus.shape[0] == 1:
                continue
            temp = {}
            for x in focus:
                _info = self.layered_color_board[tuple(x)]
                key = int(_info[1])
                # print("INFO", _info)
                if key not in self.grid_map.silos:
                    continue
                temp[tuple(x)] = [
                    dist_away, _info[0], key,
                    self.grid_map.dist_from_coord(
                        self.grid_map.silos[key])[tuple(x)]
                ]

            # print("TEMP:", temp)
            ritual_finished, ritual_pt = level_1_selection(dest_silo, temp)
        # print("KEYS", ritual_pt, self.grid_map.get_origin(),
        #       self.grid_map.silos[dest_silo])
        constructed_path += self.shortest_path_gen(self.grid_map.get_origin(),
                                                   ritual_pt,
                                                   dest_silo=dest_silo)
        ritual_steps = len(constructed_path) - 1
        constructed_path += self.shortest_path_gen(
            ritual_pt, self.grid_map.silos[dest_silo], dest_silo=dest_silo)[1:]

        return ritual_steps, constructed_path
