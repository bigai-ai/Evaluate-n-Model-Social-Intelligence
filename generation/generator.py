import numpy as np
from collections import defaultdict
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder
import pygame
import matplotlib.pyplot as plt
from utils.bfs import bfs_find_paths
from utils.consts import I2C_MAP
from render import draw_grid, draw_arrow, draw_path, WINDOW_HEIGHT


class Generator:
    '''
    Generator for iip & food truck
    '''
    def __init__(self, world_size=5, num_obstacles=2, num_objects=2):
        self.world_size = world_size
        self.num_obstacles = num_obstacles
        self.num_objects = num_objects
        # [r, c] == [y, x]
        # [x, y] == [c, r]
        self.grid_array = np.ones((world_size, world_size), dtype=np.uint8)
        self.agent_pos = []
        self.object_pos = []
        self.obstacle_pos = []
        self.target_pos = []
        self.data_type = ''

    def sample_obstacles(self):
        self.obstacle_pos = []
        for _ in range(self.num_obstacles):
            # length
            if self.data_type == 'IIP':
                l = np.random.randint(1, self.world_size)
            else:
                l = np.random.randint(2, self.world_size)

            # direction
            d = np.random.randint(0, 2)

            # horizontal
            if d == 0:
                row = np.random.randint(0, self.world_size)
                col = np.random.randint(0, self.world_size-l)
                self.grid_array[row, col:col + l] = 0
                # self.obstacle_pos.extend([(c, row) for c in range(col, col+l)])
                for c in range(col, col+l):
                    if (c, row) not in self.obstacle_pos:
                        self.obstacle_pos.append((c, row))
            # vertical
            else:
                col = np.random.randint(0, self.world_size)
                row = np.random.randint(0, self.world_size-l)
                self.grid_array[row:row + l, col] = 0
                for r in range(row, row+l):
                    if (col, r) not in self.obstacle_pos:
                        self.obstacle_pos.append((col, r))
                # self.obstacle_pos.extend([(col, r) for r in range(row, row+l)])

    def sample_objects(self):
        while True:
            self.object_pos = []
            for _ in range(self.num_objects):
                x, y = np.random.randint(0, self.world_size), np.random.randint(0, self.world_size)
                while True:
                    if (x, y) in self.obstacle_pos:
                        x, y = np.random.randint(0, self.world_size), np.random.randint(0, self.world_size)
                    else:
                        break
                self.object_pos.append((x, y))

            if self.data_type == 'IIP':
                pos1, pos2 = self.object_pos
                path = self.dijkstras_path(pos1, pos2)

                if path and len(path) >= 3:
                    break
            else:
                if len(set(self.object_pos)) == self.num_objects:
                    break

    def sample_agent(self):
        self.agent_pos = []
        x, y = np.random.randint(0, self.world_size), np.random.randint(0, self.world_size)
        while True:
            if (x, y) in self.object_pos or (x, y) in self.obstacle_pos:
                x, y = np.random.randint(0, self.world_size), np.random.randint(0, self.world_size)
            else:
                break
        self.agent_pos = (x, y)

    # loc to loc
    def dijkstras_path(self, start, end, grid_array=None):
        finder = DijkstraFinder(diagonal_movement=DiagonalMovement.never)
        grid = Grid(matrix=grid_array if grid_array is not None else self.grid_array)
        start = grid.node(*start)
        end = grid.node(*end)
        path, runs = finder.find_path(start, end, grid)
        return [(p.x, p.y) for p in path]

    def find_all_shortest_paths(self, start, end, grid_array=None):
        grid = None
        if grid_array is None:
            grid = self.grid_array
        else:
            grid = grid_array

        rows, cols = len(grid), len(grid)
        # x: row, y: col
        c, r = start
        if not (0 <= r < rows and 0 <= c < cols and grid[r][c] != 0):
            return []

        shortest_paths = []
        min_path_length = float('inf')

        def backtrack(c, r, path, path_length):
            nonlocal min_path_length

            if (c, r) == end:
                if path_length < min_path_length:
                    shortest_paths.clear()
                    min_path_length = path_length
                if path_length == min_path_length:
                    shortest_paths.append(path + [(c, r)])
                return

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0 and (nc, nr) not in path:
                    backtrack(nc, nr, path + [(c, r)], path_length + 1)

        backtrack(c, r, [], 0)
        return shortest_paths

    # loc to locs (region)
    def dijkstras_path_to_region(self, start, regions):
        min_path_len = 10000
        min_path = []
        for end in regions:
            path = self.dijkstras_path(start, end)
            if len(path) < min_path_len:
                min_path = path
                min_path_len = len(path)
        return min_path

    def more_path_diff(self, start, mid, end):
        return len(self.dijkstras_path(start, mid)) + len(self.dijkstras_path(mid, end)[1:]) - len(self.dijkstras_path(start, end))

    def more_path_diff_with_regions(self, start, mid, ends_region):
        min_diff = 10000
        ends_region = sum(ends_region, [])
        for end in ends_region:
            diff = self.more_path_diff(start, mid, end)
            if diff < min_diff:
                min_diff = diff
        return min_diff

    def find_all_paths(self, grid, start, end, base=10000):
        # find all path by bfs
        start = (start[0], start[1])
        end = (end[0], end[1])
        return bfs_find_paths(grid, start, end, base)

    def render(self, image_name, regions=None, save=True, color_board=None):
        screen = draw_grid(self.data_type, grid_size=self.world_size,
                           agent_pos=self.agent_pos,
                           obstacle_pos=self.obstacle_pos,
                           object_pos=self.object_pos,
                           target_pos=self.target_pos,
                           color_board=color_board,
                           regions=regions)
        if save:
            pygame.image.save(screen, image_name)
        return screen

    def region_coloring(self, rgba, image_name, paths=None):
        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(rgba)
        ax.set_xticks([])
        ax.set_yticks([])
        if paths:
            for path in paths:
                draw_path(path, ax, ec="k", fc="k")
        fig.savefig(image_name, dpi=200)
        plt.close('all')

    # def region_coloring2(self, image_name, color_board):
    #     self.render(image_name, color_board=color_board, save=True)

    def turn_or_not(self, trj):
        '''
        :param trj:
        :return: true, means
        '''
        if all([p[0] == trj[0][0] for p in trj]):
            return False
        if all([p[1] == trj[0][1] for p in trj]):
            return False
        return True

    def turn_or_not_strictly(self, trj):
        if all([p1[0]-p2[0] == 1 for p1, p2 in zip(trj[1:], trj[:-1])]):
            return False
        if all([p1[0]-p2[0] == -1 for p1, p2 in zip(trj[1:], trj[:-1])]):
            return False
        if all([p1[1]-p2[1] == 1 for p1, p2 in zip(trj[1:], trj[:-1])]):
            return False
        if all([p1[1]-p2[1] == -1 for p1, p2 in zip(trj[1:], trj[:-1])]):
            return False
        return True

    def turn_count(self, trj):
        if len(trj) <= 2:
            return 0
        if all([p[0] == trj[0][0] for p in trj]):
            return 0
        if all([p[1] == trj[0][1] for p in trj]):
            return 0

        turn_index = 0
        for i in range(1, len(trj)+1):
            if self.turn_or_not(trj[:i]):
                turn_index = i-1
                break
        return 1 + self.turn_count(trj[turn_index:])


    def trj_direction_desc(self, trj, desc_list):
        # desc_list = []
        if not self.turn_or_not(trj):
            # up or down;
            if all([p[0] == trj[0][0] for p in trj]):
                if trj[1][1] == trj[0][1] + 1:
                    desc_list.append(f'goes down to {trj[-1]}')
                    return
                if trj[1][1] == trj[0][1] - 1:
                    desc_list.append(f'goes up to {trj[-1]}')
                    return
            # left or right;
            if all([p[1] == trj[0][1] for p in trj]):
                if trj[1][0] == trj[0][0] + 1:
                    desc_list.append(f'goes right to {trj[-1]}')
                    return
                if trj[1][0] == trj[0][0] - 1:
                    desc_list.append(f'goes left to {trj[-1]}')
                    return

        turn_index = -1
        for i in range(2, len(trj)+1):
            if self.turn_or_not(trj[:i]):
                turn_index = i-1
                break
        if turn_index != -1:
            self.trj_direction_desc(trj[:turn_index], desc_list)
            self.trj_direction_desc(trj[turn_index-1:], desc_list)
        return

    def trj_direction_details_desc(self, trj, desc_list):
        # desc_list = []
        if not self.turn_or_not_strictly(trj):
            # up or down;
            if all([p[0] == trj[0][0] for p in trj]):
                if trj[1][1] == trj[0][1] + 1:
                    for src, tgt in zip(trj[0:-1], trj[1:]):
                        desc_list.append(f'Move down from {src} to {tgt}')
                    return
                if trj[1][1] == trj[0][1] - 1:
                    for src, tgt in zip(trj[0:-1], trj[1:]):
                        desc_list.append(f'Move up from {src} to {tgt}')
                    return
            # left or right;
            if all([p[1] == trj[0][1] for p in trj]):
                if trj[1][0] == trj[0][0] + 1:
                    for src, tgt in zip(trj[0:-1], trj[1:]):
                        desc_list.append(f'Move right from {src} to {tgt}')
                    return
                if trj[1][0] == trj[0][0] - 1:
                    for src, tgt in zip(trj[0:-1], trj[1:]):
                        desc_list.append(f'Move left from {src} to {tgt}')
                    return

        turn_index = -1
        for i in range(2, len(trj)+1):
            if self.turn_or_not_strictly(trj[:i]):
                turn_index = i-1
                break
        if turn_index != -1:
            self.trj_direction_details_desc(trj[:turn_index], desc_list)
            self.trj_direction_details_desc(trj[turn_index-1:], desc_list)
        return

    def draw_trj(self, screen, trj, CELL_SIZE=80, ratio=0.2):
        path = np.array(trj) * CELL_SIZE + CELL_SIZE // 2
        path_coord = np.zeros_like(path, dtype=float)

        path_coord[0] = path[0]

        delta_directions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        directions = delta_directions[np.arange(1, 5) % 4, :]
        cur = np.zeros(4, dtype=int)
        last = np.zeros(4, dtype=int)
        # ratios = np.linspace(0.1, 0.3, len(path))
        for i in range(len(path) - 1):
            spec = np.einsum("ij,j->i", directions, (path[i + 1] - path[i]))
            spec[spec < 0] = 0
            spec = np.array(spec, dtype=int)
            last = np.bitwise_or(spec, last)
            cur = spec
            path_coord[i] = path[i] - ratio * np.einsum("ij,i->j", delta_directions, last)
            last = cur

        path_coord[-1] = path[-1] - ratio * np.einsum("ij,i->j", delta_directions, last)

        for i in range(len(path_coord) - 1):
            draw_arrow(screen, path_coord[i], path_coord[i + 1], i, len(path) - 1)
        return screen

    def draw_ft_trj(self, image_name, trj, ratio=0.2):
        screen = self.render(image_name)
        screen = self.draw_trj(screen, trj, ratio=ratio)
        # preference = '-'.join([I2C_MAP[p] for p in preference])
        # if image_name.count('.') == 2:
        #     image_name = image_name[:image_name.rindex('.')]
        pygame.image.save(screen, image_name)
        return screen

    def render_iip_trj(self, image_name, trj, screen=None):
        if screen is None:
            screen = self.render(image_name, save=False)
        CELL_SIZE = WINDOW_HEIGHT // self.world_size
        screen = self.draw_trj(screen, trj, CELL_SIZE=CELL_SIZE)
        pygame.image.save(screen, image_name)


    def grid_layout_rep(self):
        text_grid = [['*']*self.world_size for _ in range(self.world_size)]
        for c in range(self.world_size):
            for r in range(self.world_size):
                if self.grid_array[r, c] == 0:
                    text_grid[r][c] = 'W'
        c, r = self.agent_pos
        text_grid[r][c] = 'A'

        for i, (c, r) in enumerate(self.object_pos):
            if self.data_type == 'IIP':
                text_grid[r][c] = 'X' if self.target_pos == (c, r) else 'Y'
            else:
                text_grid[r][c] = I2C_MAP[i]
        return '\n'.join([''.join(r) for r in text_grid])

    def sample(self, N=5):
        raise NotImplementedError


if __name__ == '__main__':

    desc_list = []
    trj = [(0, 4), (1, 4), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)]

    gen = Generator()
    # gen.trj_direction_desc(trj, desc_list)
    # gen.turn_count(trj)
    trj = [(0, 4), (1, 4), (1, 3), (2, 3)
        , (2, 2), (2, 1),
           # (2, 0), (1, 0)
           ]
    print(gen.turn_count(trj))
