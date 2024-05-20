def is_valid_cell(grid, x, y):
    # Check if the cell is within the grid bounds and is not blocked
    rows, cols = len(grid), len(grid[0])
    return 0 <= x < rows and 0 <= y < cols and grid[x][y] != '#'


def get_neighbors(x, y):
    # Return the possible neighbor cells (up, down, left, right)
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]


def find_all_shortest_paths(grid, start_x, start_y, dest_x, dest_y):
    if not is_valid_cell(grid, start_x, start_y) or not is_valid_cell(grid, dest_x, dest_y):
        return []

    rows, cols = len(grid), len(grid[0])
    shortest_paths = []
    min_path_length = float('inf')

    def backtrack(x, y, path, path_length):
        nonlocal min_path_length

        if x == dest_x and y == dest_y:
            if path_length < min_path_length:
                shortest_paths.clear()
                min_path_length = path_length
            if path_length == min_path_length:
                shortest_paths.append(path + [(x, y)])
            return

        for nx, ny in get_neighbors(x, y):
            if is_valid_cell(grid, nx, ny) and (nx, ny) not in path:
                backtrack(nx, ny, path + [(x, y)], path_length + 1)

    backtrack(start_x, start_y, [], 0)
    return shortest_paths

# Example usage:
grid = [
    ['#', '.', '.', '#', '#'],
    ['.', '.', '.', '.', '#'],
    ['#', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '#'],
    ['#', '#', '.', '.', '.'],
]

# start_x, start_y = 0, 1
# start_x, start_y = 1, 1
start_x, start_y = 1, 1
dest_x, dest_y = 4, 4

shortest_paths = find_all_shortest_paths(grid, start_x, start_y, dest_x, dest_y)
for path in shortest_paths:
    print(len(path), path)
