"""
Map Generation @ ToM GW

J. Wang
"""

import numpy as np
from itertools import product
from ..generation.grid import GridMap


class GenMap:
    """
    Generate a map.
    """
    def __init__(self,
                 xlim: int,
                 ylim: int,
                 silo_num: int = 2,
                 block_ratio: float = 0.3):
        "Initialize."

        self.xlim = xlim
        self.ylim = ylim
        self.silo_num = silo_num
        self.block_ratio = block_ratio

    def generate(self):
        """Generate radom map."""

        num_block = int(np.floor(self.xlim * self.ylim * self.block_ratio))

        self.grid_map = GridMap(self.xlim, self.ylim, silo_num=self.silo_num)

        positions = list(product(np.arange(self.xlim), np.arange(self.ylim)))

        indices = self.xlim * self.ylim
        index = np.random.randint(indices)
        self.grid_map.set_origin(positions.pop(index))
        indices -= 1
        for ii in range(1, self.silo_num + 1):
            self.grid_map.set_silo(ii,
                                   positions.pop(np.random.randint(indices)))
            indices -= 1

        for _ in range(num_block):
            self.grid_map.set_blocks(
                [positions.pop(np.random.randint(indices))])
            indices -= 1

        if not self.check_availability():
            print("generation failed.")
            return "Generation Failed"

        print("generation succeeded.")
        return self.grid_map

    def check_availability(self):
        """Check availability."""
        dist = self.grid_map.dist_from_coord(self.grid_map.get_origin())
        distances = np.array(
            [dist[pos] for pos in self.grid_map.silos.values()])
        return np.all(distances < self.grid_map.INF - 10)
