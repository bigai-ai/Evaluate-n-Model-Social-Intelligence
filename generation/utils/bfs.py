from collections import deque
import numpy as np


def neighbors(grid, pos):
    x, y = pos
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        # todo, grid[nx][ny] => grid[ny][nx]
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[ny][nx] == 1:
            yield nx, ny


def get_observable_neighbors(grid, pos, radius=1):
    x, y = pos
    candidates = [(dx, dy) for dx in range(-radius, radius+1) for dy in range(-radius, radius+1)]
    neighbors = []
    for dx, dy in candidates:
        nx, ny = x+dx, y+dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[ny][nx] == 1:
            neighbors.append((nx, ny))
    return neighbors


def get_observable_targets(grid, cur_pos, targets, radius=1):
    observable_targets = []
    for i, target in enumerate(targets):
        if cur_pos in get_observable_neighbors(grid, target, radius):
            observable_targets.append(i)
    return observable_targets


def common_pre_trj_check(dedup_set, path):
    if path in dedup_set:
        return True
    for visited in dedup_set:
        if len(path) > len(visited):
            continue
        if visited[:len(path)] == path:
            return True
    return False


def bfs_find_paths(grid, start, end, base=10000):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    all_paths = set()
    # dedup_set = set(tuple([start], ))
    while queue:
        (c, r), path = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            new_point = (nc, nr)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0 and new_point not in path:
                if new_point == end:
                    all_paths.add(tuple(path + [new_point]))
                else:
                    l = path + [new_point]
                    if len(l) >= 2*base and len(l) >= 10:
                        continue
                    queue.append((new_point, l))
        # print(len(queue))
    return all_paths


def find_all_shortest_paths(grid, start, end):
    rows, cols = len(grid), len(grid[0])
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


if __name__ == '__main__':
    grid = np.asarray([
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])

    start = (0, 0)
    end = (3, 2)

    # paths = bfs_find_paths(grid, start, end)
    paths = find_all_shortest_paths(grid, start, end)
    for path in paths:
        print(path)
