import numpy as np


class BaseStrategy:
    """Base strategy."""
    def __init__(self, grid_map):
        """Initialization"""

        self.grid_map = grid_map

        self.dist_p = self.grid_map.dist_from_coord(self.grid_map.get_origin())
        self.xlim, self.ylim = grid_map.board.shape

    def path_generation(self, *args, **kwargs):
        """Generate Path. Abstract method."""
        raise Exception("This method should be overridden.")