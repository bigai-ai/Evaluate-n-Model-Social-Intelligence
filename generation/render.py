
import pygame
import math
import numpy as np
from utils.consts import I2C_MAP

pygame.font.init()
my_font = pygame.font.SysFont('Times New Roman', 40)

WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400
BLACK = (0, 0, 0)
LINE_BLACK = (226, 226, 226)
WHITE = (255, 255, 255)

GREY = (183, 183, 183)
SILVER_GREY = (192, 192, 192)

RED = (255, 0, 0)
# LIGHT_RED = (255, 192, 203)
# 75%, FB8488
LIGHT_RED = (251, 132, 136)
BLUE = (0, 0, 255)
# LIGHT_BLUE = (0, 255, 255)
# LIGHT_BLUE = (125, 187, 232)
# 75%, 93C7EC
LIGHT_BLUE = (147, 199, 236)
GREEN = (0, 255, 0)
LIGHT_GREEN = (166, 227, 95)
YELLOW = (255, 215, 0)

RED_MAP = {
    1: (255, 51, 51),
    2: (255, 102, 102),
    3: (255, 153, 153),
    4: (255, 204, 204)
}

BLUE_MAP = {
    1: (51, 51, 255),
    2: (102, 102, 255),
    3: (153, 153, 255),
    4: (204, 204, 255)
}


def pos_to_level(locs_levels):
    assert len(locs_levels) >= 1
    level_map = {}
    for level, locs in enumerate(locs_levels[1:]):
        for loc in locs:
            level_map[loc] = level+1
    return level_map


def draw_grid(data_type, grid_size, agent_pos, obstacle_pos, object_pos, target_pos, regions=None, color_board=None):
    cell_size = WINDOW_HEIGHT // grid_size
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags=pygame.HIDDEN)
    screen.fill(WHITE)
    for r, y in enumerate(range(0, WINDOW_HEIGHT, cell_size)):
        for c, x in enumerate(range(0, WINDOW_WIDTH, cell_size)):
            # r = grid_size-1-r
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if (c, r) == agent_pos:
                pygame.draw.rect(screen, LIGHT_GREEN, rect)
                text_surface = my_font.render(f'A', False, BLACK)
                screen.blit(text_surface, (x + 20, y + 14))
            elif (c, r) in object_pos:
                if data_type == 'IIP':
                    if (c, r) == target_pos:
                        pygame.draw.rect(screen, LIGHT_RED, rect)
                        text_surface = my_font.render(f'X', False, BLACK)
                        screen.blit(text_surface, (x + 20, y + 14))
                    else:
                        pygame.draw.rect(screen, LIGHT_BLUE, rect)
                        text_surface = my_font.render(f'Y', False, BLACK)
                        screen.blit(text_surface, (x + 20, y + 14))
                else:
                    pygame.draw.rect(screen, LIGHT_RED, rect)
                    text_surface = my_font.render(f'{I2C_MAP[object_pos.index((c, r))]}', False, (0, 0, 0))
                    screen.blit(text_surface, (x+20, y+14))
            elif (c, r) in obstacle_pos:
                pygame.draw.rect(screen, GREY, rect)

            elif regions and data_type == 'IIP':
                trg_1_locs, trg_2_locs, eq_locs = regions
                trg_1_map = pos_to_level(trg_1_locs)
                trg_2_map = pos_to_level(trg_2_locs)

                trg_idx = object_pos.index(target_pos)
                if (c, r) in sum(trg_1_locs, []):
                    if trg_idx == 0:
                        pygame.draw.rect(screen, RED_MAP[trg_1_map[(c, r)]], rect)
                    else:
                        pygame.draw.rect(screen, BLUE_MAP[trg_1_map[(c, r)]], rect)
                elif (c, r) in sum(trg_2_locs, []):
                    if trg_idx == 0:
                        pygame.draw.rect(screen, BLUE_MAP[trg_2_map[(c, r)]], rect)
                    else:
                        pygame.draw.rect(screen, RED_MAP[trg_2_map[(c, r)]], rect)
                # elif (c, r) in eq_locs:
                #     pygame.draw.rect(screen, YELLOW, rect)
            else:
                if color_board is None:
                    pygame.draw.rect(screen, LINE_BLACK, rect, 1)
                else:
                    from collections import Counter
                    list_of_tuples = [tuple(ele) for ele in color_board.round(2).reshape(-1, 4).tolist()]
                    counter = Counter(list_of_tuples)
                    pygame.draw.rect(screen, color_level_map(color_board.round(2)[r][c]), rect)
    pygame.draw.rect(screen, BLACK, (0, 0, 400, 400), 3)
    return screen


def convert_rgba_to_rgb(rgba):
    return (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))


def color_level_map(rgba):
    # red
    if np.sum(np.abs(np.asarray(rgba)-(1.0, 0.4, 0.4, 1.0))) < 0.1:
        # 80% (FB8488 75%)
        return (252, 156, 159, 255)
    if np.sum(np.abs(np.asarray(rgba)-(1.0, 0.64, 0.64, 1.0))) < 0.1:
        # 85% (FB8488 75%)
        return (253, 181, 183, 255)
    if np.sum(np.abs(np.asarray(rgba)-(1.0, 0.78, 0.78, 1.0))) < 0.1:
        # 90% (FB8488 75%)
        return (254, 205, 207, 255)

    # blue
    if np.sum(np.abs(np.asarray(rgba)-(0.4, 0.4, 1.0, 1.0))) < 0.1:
        # 80%
        return (169, 208, 239, 255)
    if np.sum(np.abs(np.asarray(rgba)-(0.64, 0.64, 1.0, 1.0))) < 0.1:
        # 85%
        return (190, 219, 243, 255)
    if np.sum(np.abs(np.asarray(rgba)-(0.78, 0.78, 1.0, 1.0))) < 0.1:
        # 90% (93C7EC)
        return (212, 231, 247, 255)

    return convert_rgba_to_rgb(rgba)


def draw_arrow(screen, start, end, i, n):
    pygame.draw.line(screen, BLACK, start, end, 3)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    radius = 10
    # red: (255, 192-192*((i+1)/n), 192-192*((i+1)/n))
    # yellow: (255, 255, 192-192*((i+1)/n))
    # black:(192-192*((i+1)/n), 192-192*((i+1)/n), 192-192*((i+1)/n))
    pygame.draw.polygon(screen,
                        (255, 192-192*((i+1)/n), 192-192*((i+1)/n)),
                        # (192 - 192 * ((i + 1) / n), 255, 192-192*((i+1)/n)),
                        ((end[0]+radius*math.sin(math.radians(rotation)), end[1]+radius*math.cos(math.radians(rotation))),
                         (end[0]+radius*math.sin(math.radians(rotation-120)), end[1]+radius*math.cos(math.radians(rotation-120))),
                         (end[0]+radius*math.sin(math.radians(rotation+120)), end[1]+radius*math.cos(math.radians(rotation+120)))
                         )
                        )


def draw_path(path, ax, ratio=0.2, **kwargs):
    path = np.array(path)
    path_coord = np.zeros_like(path, dtype=float)

    path_coord[0] = path[0]

    delta_directions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    directions = delta_directions[np.arange(1, 5) % 4, :]
    cur = np.zeros(4, dtype=int)
    last = np.zeros(4, dtype=int)
    for i in range(len(path) - 1):
        spec = np.einsum("ij,j->i", directions, (path[i + 1] - path[i]))
        spec[spec < 0] = 0
        spec = np.array(spec, dtype=int)

        last = np.bitwise_or(spec, last)
        cur = spec
        path_coord[i] = path[i] - ratio * np.einsum("ij,i->j",
                                                    delta_directions, last)
        last = cur
    path_coord[-1] = path[-1] - ratio * np.einsum("ij,i->j", delta_directions,
                                                  last)

    for i in range(len(path) - 1):
        ax.arrow(path_coord[i, 0],
                 path_coord[i, 1],
                 path_coord[i + 1, 0] - path_coord[i, 0],
                 path_coord[i + 1, 1] - path_coord[i, 1],
                 head_width=0.1,
                 length_includes_head=True,
                 **kwargs)


def draw_trajectory():
    pass
