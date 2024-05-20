"""
Utilities @ ToM GW

J. Wang
"""
import numpy as np
from typing import List, Union, Tuple
from matplotlib import pyplot as plt


class GridMap:
    """
    Grid world map.

    TILE_ENC should be fewer than 256 status
    """

    INF = 500000000
    INF_THRES = 400000000
    TILE_ENC = {
        "blank": 0,
        "silo_1": -1,
        "silo_2": -2,
        "block": 1,
        "origin": 2,
    }
    TILE_DEC = {}
    BLOCK_TILE = 1

    def __init__(self,
                 xlim: int,
                 ylim: int,
                 trg_silo: int = None,
                 pattern: np.ndarray = None,
                 silo_num: int = 2,
                 silo_a: List[int] = None,
                 silo_b: List[int] = None):
        """Initialize."""
        self.xlim = xlim
        self.ylim = ylim
        self.silo_num = silo_num
        self.board = np.zeros([xlim, ylim], dtype=np.int8)
        self.silos = {}
        self.origin = None
        self.trg_silo = trg_silo
        self.dist_action = {
            self.TILE_ENC["block"]: (lambda x: (False, self.INF)),
            "OTHER": lambda x: (True, 1)
        }
        if pattern is not None:
            self.board = np.asarray(pattern, dtype=np.int8)
            self.parse_pattern()

        self.dist_bank = {}
        if silo_a is not None:
            self.set_silo_a(silo_a)

        if silo_b is not None:
            self.set_silo_b(silo_b)

        # RGBA setup of the board.
        self.reset_colorboard()

    @property
    def COLORMAP(self):
        colormap = {
            1: np.array([0.5, 0.5, 0.5, 1.]),
            2: np.array([0., 1., 0., 1.]),
            # RED
            -1: np.array([1., 0., 0., 1.]),
            # BLUE
            -2: np.array([0., 0., 1., 1.])
        }
        if self.trg_silo == 1:
            colormap[-1], colormap[-2] = colormap[-2], colormap[-1]
        return colormap

    def reset_colorboard(self):
        self.color_board = np.ones(list(self.board.shape) + [4], dtype=float)

    def parse_pattern(self):
        """Parse Pattern."""
        self.origin = tuple(np.argwhere(self.board == 2)[0])
        for i in range(1, self.silo_num + 1):
            silo_i = tuple(np.argwhere(self.board == -i)[0])
            self.set_silo(i, silo_i)

    def set_silo_a(self, silo_a: List[int]):
        """Set target A"""
        self.set_silo(1, silo_a)

    def set_silo_b(self, silo_b: List[int]):
        """Set target A"""
        self.set_silo(2, silo_b)

    def set_silo(self, i: int, pos: List[int]):
        """Set general silo."""
        assert 0 <= pos[0] < self.xlim
        assert 0 <= pos[1] < self.ylim
        if f"silo_{i}" not in self.TILE_ENC:
            self.TILE_ENC[f"silo_{i}"] = -i
            self.TILE_DEC[-i] = f"silo_{i}"
        self.board[tuple(pos)] = self.TILE_ENC[f"silo_{i}"]
        self.silos[i] = tuple(pos)

    def set_tiles(self, tile_type: Union[str, int], coords: np.ndarray):
        """Set tiles of the map."""
        if isinstance(tile_type, str):
            tile_type = self.TILE_ENC[tile_type]
        for pt in coords:
            self.board[tuple(pt)] = tile_type

    def set_blocks(self, coords: np.ndarray):
        """Set blocks."""
        self.set_tiles("block", coords)

    def set_origin(self, coord: np.ndarray):
        """Set blocks."""
        self.set_tiles("origin", [coord])
        self.origin = coord

    def get_origin(self):
        """Get Origin."""
        return self.origin

    def set_base_colors(self, colormap: dict = None, board: np.ndarray = None):
        """Color the map: `self.color_board`."""
        if board is None:
            board = self.color_board
        cmap = self.COLORMAP.copy()
        cmap.update({} if colormap is None else colormap)

        for i in [-self.TILE_ENC['block'], -self.TILE_ENC['origin']] + list(
                self.silos.keys()):
            board[self.board == -i, :] = cmap[-i]

    def adjacents(self, coord: List[int]) -> List[tuple]:
        """Find all adjacents."""
        adj = []
        if (x := coord[0]) > 0:
            adj += [(x - 1, (y := coord[1]))]
        if (y := coord[1]) > 0:
            adj += [(x, y - 1)]
        if x < self.xlim - 1:
            adj += [(x + 1, y)]
        if y < self.ylim - 1:
            adj += [(x, y + 1)]
        return adj

    def dist_from_coord(self, coord: List[int], cached: bool = True):
        """
        Parameters
        ----------
        coord: List[int] :

        cached: bool : , optional


        Returns
        -------
        out :

        """
        coord = tuple(coord)
        if tuple(coord) in self.dist_bank:
            return self.dist_bank[tuple(coord)]

        dist = np.ones_like(self.board, dtype=np.int64) * self.INF
        if self.board[coord] == self.BLOCK_TILE:
            return dist

        process_list = [coord]
        dist[coord] = 0
        original_coord = coord

        while len(process_list) > 0:
            coord = process_list.pop(0)
            for p in (a := self.adjacents(coord)):

                if_attach, dist_increment = self.dist_action.get(
                    self.board[p], self.dist_action["OTHER"])((p, self.board))

                if if_attach and dist[coord] + dist_increment < dist[p]:
                    process_list += [p]
                    dist[p] = dist_increment + dist[coord]

        if cached:
            self.dist_bank[tuple(original_coord)] = dist.copy()

        return dist.copy()

    def dist_from_region(self, region: Tuple[Tuple[int]], cached: bool = True):
        """
        Distance from a region.

        Dynamic Programming, or BFS.
        """
        region = tuple(tuple(x) for x in region)
        if region in self.dist_bank:
            return self.dist_bank[region]

        dist = np.ones_like(self.board, dtype=np.int64) * self.INF
        process_list = [tuple(x) for x in region]
        for x in process_list:
            dist[x] = 0

        while len(process_list) > 0:
            coord = process_list.pop(0)
            for p in (a := self.adjacents(coord)):
                if_attach, dist_increment = self.dist_action.get(
                    self.board[p], self.dist_action["OTHER"])((p, self.board))

                if if_attach and dist[coord] + dist_increment < dist[p]:
                    process_list += [p]
                    dist[p] = dist_increment + dist[coord]
        if cached:
            self.dist_bank[region] = dist.copy()
        return dist.copy()

    @staticmethod
    def construct_board(obstacle, silos, trg_silo, origin, size=None):
        if size is None:
            size = [5, 5]
        gmap = GridMap(*size, trg_silo=trg_silo)
        for i, s in enumerate(silos):
            gmap.set_silo(i + 1, s)
        gmap.set_origin(origin)
        gmap.set_blocks(obstacle)
        return gmap


def grid_map_dec_constructor():
    """
    Construct GridMap.TILE_DEC
    """
    for k, v in GridMap.TILE_ENC.items():
        GridMap.TILE_DEC[v] = k


grid_map_dec_constructor()
