"""
Base strategy class @ ToM GW

J. Wang
"""

import numpy as np
from ..generation.grid import GridMap

EXAMPLE_BOARD = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [2, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, -2],
],
                         dtype=np.int8)


class BaseStrategy:
    """Base strategy."""
    def __init__(self, grid_map):
        """Initialization"""
        if grid_map is None:
            self.grid_map = GridMap(*EXAMPLE_BOARD.shape,
                                    pattern=EXAMPLE_BOARD)
        else:
            self.grid_map = grid_map

        self.dist_p = self.grid_map.dist_from_coord(self.grid_map.get_origin())
        self.xlim, self.ylim = grid_map.board.shape

    def path_generation(self, *args, **kwargs):
        """Generate Path. Abstract method."""
        raise Exception("This method should be overridden.")
