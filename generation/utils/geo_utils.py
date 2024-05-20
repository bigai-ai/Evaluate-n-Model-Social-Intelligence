import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return np.asarray(vector) / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def direction_target_angle(src_pos, trg_pos, move_direction):
    trg_direction = np.asarray(trg_pos) - np.asarray(src_pos)
    angle = angle_between(trg_direction, move_direction)
    # angle = angle * 180 / np.pi
    return angle


def turn_angle(src_pos, next_pos, trg_pos):
    next_direction = np.asarray(next_pos) - src_pos
    trg_direction = np.asarray(trg_pos) - next_pos
    return angle_between(next_direction, trg_direction)


def angle_level(agent_pos, obj_pos):
    direction = np.asarray(agent_pos) - np.asarray(obj_pos)
    return angle_between([1, 0], direction)


if __name__ == '__main__':
    # src_pos = (2, 3)
    # trg_pos = (4, 0)
    #
    # # right, left, down, up
    # for move_direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    #     print(move_direction, direction_target_angle(src_pos, trg_pos, move_direction))

    agent_pos = (4, 3)
    obj_pos = (3, 2)
    print(angle_level(agent_pos, obj_pos))
