"""
Avoidant Path Generation @ ToM GW

J. Wang
"""

import numpy as np
from typing import List
from pathfinding.core.grid import Grid
from pathfinding.finder.bi_a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


from .base import BaseStrategy
from .hybrid import StrategyHybrid
from grid import GridMap


class StrategyAvoidant(StrategyHybrid):
    """Strategy 3: Avoidant"""
    path_finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

    def __init__(self, grid_map: GridMap, **kwargs):
        "Initialize"
        super().__init__(grid_map, **kwargs)
        self.avoidance = self.grid_map.board.copy()
        self.avoidance = 1 - (self.avoidance == GridMap.TILE_ENC['block'])
        # self.avoidance = Grid(matrix=self.avoidance)
        self.frontier = []

    def _stop_cond(self, **kwargs):
        "The stop condition of BFS. Now it means if flood can go no more."
        return len(kwargs.get("frontier", [])) == 0

    @staticmethod
    def _find_path(board: np.ndarray, start: np.ndarray, end: np.ndarray):
        grid = Grid(matrix=board.T)
        start = grid.node(*start)
        end = grid.node(*end)
        path, _ = StrategyAvoidant.path_finder.find_path(start, end, grid)
        # print(board.T)
        return path

    @staticmethod
    def _is_connected(board: np.ndarray, start: np.ndarray, end: np.ndarray):
        path = StrategyAvoidant._find_path(board, start, end)
        return all(x.walkable for x in path) and len(path) > 0
        # return len(StrategyAvoidant._find_path(board, start, end)) > 0

    def generate_max_block(self,
                           dest_silo: int,
                           counter_region: List[tuple],
                           stop_cond: callable = None):
        """
        Generate Max Blocks, which leaves only one way out to destination.
        """
        self.avoidance = self.grid_map.board.copy()
        self.avoidance = 1 - (self.avoidance == GridMap.TILE_ENC['block'])
        dist_d = self.grid_map.dist_from_coord(self.grid_map.silos[dest_silo])
        dist_o = self.grid_map.dist_from_coord(self.grid_map.get_origin())
        dist_other = self.grid_map.dist_from_region(counter_region)
        if stop_cond is None:
            stop_cond = self._stop_cond

        start = self.grid_map.origin
        end = tuple(self.grid_map.silos[dest_silo])
        self.frontier = counter_region
        # another choice is: `self.frontier` = all tiles of other colors

        # BFS on flooding
        while not stop_cond(frontier=self.frontier):
            current_frontier = []
            for v in self.frontier:
                self.avoidance[tuple(v)] = 0
                if not self._is_connected(self.avoidance, start, end):
                    self.avoidance[tuple(v)] = 1
                    continue
                current_frontier += [v]

            new_frontier = []
            for v in current_frontier:
                adj = self.grid_map.adjacents(list(v))

                for a in adj:
                    if self.avoidance[a] <= 0:
                        continue
                    new_frontier += [a]

            self.frontier = sorted(
                new_frontier,
                key=lambda x: (dist_other[x], dist_d[x], -dist_o[x]),
                # The earlier to check, the more likely to
                #     be added to virtual blocks.
                # First check those far to others, leave those
                #     already checked last round.
                # Then those far from destination
                # Finally those close to origin.
                reverse=True)

        return self.avoidance

    def path_generation(self, dest_silo: int, **kwargs):
        """
        Generate the Avoidant path.
        """
        if kwargs.get("from_colored", False):
            regions = self.coloring_saturated()
            counter_region = []
            for key, val in regions.items():
                if key != dest_silo:
                    counter_region += val
        else:
            counter_region = [
                v for k, v in self.grid_map.silos.items() if k != dest_silo
            ]

        self.generate_max_block(dest_silo, counter_region,
                                kwargs.get("stop_cond", None))
        path = self._find_path(self.avoidance, self.grid_map.origin,
                               self.grid_map.silos[dest_silo])

        return None, [(v.x, v.y) for v in path]


def construct_board(obstacle, silos, origin, size=[5, 5]):
    gmap = GridMap(*size)
    for i, s in enumerate(silos):
        gmap.set_silo(i + 1, s)
    gmap.set_origin(origin)
    gmap.set_blocks(obstacle)
    return gmap


def main(**kwargs):
    """
    The main function
    """
    if "example" in kwargs:
        gmap = GridMap(7, 5, silo_a=(4, 0), silo_b=(6, 0))
        gmap.set_tiles("block", [(0, 2), (1, 2), (2, 2), (3, 2)])
        gmap.set_origin((2, 4))
    else:
        size = kwargs.get("size", [5, 5])
        obstacles = kwargs.get("obstacles", tuple())
        objects = kwargs.get("objects", kwargs.get("silos", tuple()))
        target = kwargs.get("target", 1)
        agent = kwargs.get("agent", kwargs.get("origin", tuple()))
        gmap = construct_board(obstacles, objects, agent, size)

    # print(gmap.get_origin())
    model = StrategyAvoidant(gmap)
    path = model.path_generation(1, from_colored=True)
    print(model.avoidance.T)
    print("Path_1:", path)

    path = model.path_generation(2, from_colored=True)
    print(model.avoidance.T)
    print("Path_2:", path)


if __name__ == '__main__':
    main(example=True)
