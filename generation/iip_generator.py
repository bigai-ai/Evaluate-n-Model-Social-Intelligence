import numpy as np
from tqdm import tqdm
from enum import Enum, unique
import os
import shutil
import json
from datetime import datetime
from collections import defaultdict
from generator import Generator
from grid import GridMap
from strategies.hybrid import StrategyHybrid
from strategies.avoidant import StrategyAvoidant


@unique
class TrjType(Enum):
    SEEK_FAR = 'seek_far'
    OVERLAP = 'overlap'

@unique
class BestTrjType(Enum):
    TURN_PASS = 0
    TURN_NOTPASS = 1
    NOTTURN_PASS = 2
    NOTTURN_NOTPASS = 3

@unique
class OptionType(Enum):
    # BEST = 'best'
    # SHORTEST = 'shortest'
    # MISLEADING = 'misleading'
    # FAR = 'far'

    BEST = 'hybrid'
    SHORTEST = 'shortest'
    MISLEADING = 'reversed'
    FAR = 'avoidant'


class IIPGenerator(Generator):
    '''
    Inverse Inverse Planning Generator
    '''
    def __init__(self,
                 world_size=5,
                 num_obstacles=2,
                 num_objects=2,
                 one_shot=False):
        super(IIPGenerator, self).__init__(world_size, num_obstacles,
                                           num_objects)
        self.data_type = 'IIP'
        self.image_dir = './data/IIP'
        self.one_shot = one_shot
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def _shortest_path_overlap(self, shortest_paths1, all_path1,
                               shortest_paths2):
        for path1 in shortest_paths1:
            if not all([path1[1] in path2 for path2 in shortest_paths2]):
                return False
        if self.turn_or_not(shortest_paths2[0]):
            return False
        if len(shortest_paths1) == 1 and len(shortest_paths2) == 1:
            if len(set(shortest_paths1[0][1:])
                   & set(shortest_paths2[0][1:])) * 1. / len(
                       set(shortest_paths1[0][1:])
                       | set(shortest_paths2[0][1:])) < 0.3:
                return False
            for path1 in all_path1:
                for path2 in shortest_paths2:
                    if len(set(path1[1:]) & set(path2[1:])) == 0:
                        return True
        return False

    def _valid(self, path1, all_path1, path2, all_path2):
        shortest_paths1 = [p for p in all_path1 if len(p) == len(path1)]
        shortest_paths2 = [p for p in all_path2 if len(p) == len(path2)]
        if not all([len(p) == len([path1]) for p in all_path1]):
            if all([self.object_pos[1] in p for p in shortest_paths1]):
                if not all([self.object_pos[1] in p for p in all_path1]):
                    return True, 0, TrjType.SEEK_FAR

        if not all([len(p) == len([path2]) for p in all_path2]):
            if all([self.object_pos[0] in p for p in shortest_paths2]):
                if not all([self.object_pos[0] in p for p in all_path2]):
                    return True, 1, TrjType.SEEK_FAR

        if self._shortest_path_overlap(shortest_paths1, all_path1,
                                       shortest_paths2):
            return True, 0, TrjType.OVERLAP

        if self._shortest_path_overlap(shortest_paths2, all_path2,
                                       shortest_paths1):
            return True, 1, TrjType.OVERLAP
        return False, None, None, None

    def load_setting(self, obstacles, objects, target, agent, id):
        self.obstacle_pos = obstacles
        for obs in obstacles:
            r, c = obs[::-1]
            self.grid_array[r][c] = 0
        self.object_pos = objects
        self.target_pos = target
        self.agent_pos = agent

        self.targer_pos_index = self.object_pos.index(target)
        self.other_pos_index = 1 - self.targer_pos_index
        self.other_pos = self.object_pos[self.other_pos_index]

        gmap = GridMap.construct_board(
            obstacle=self.obstacle_pos,
            silos=self.object_pos,
            origin=self.agent_pos,
            size=[self.world_size, self.world_size],
            trg_silo=self.targer_pos_index)
        model = StrategyHybrid(gmap)

        regions = model.coloring_saturated()
        misleading_regions = regions[self.other_pos_index + 1]

        paths = []

        for i in range(model.grid_map.silo_num):
            _, path = model.path_generation(i + 1)
            paths += [path]

        ritual_steps, best_trj = model.path_generation(self.targer_pos_index + 1)
        print(best_trj)

        _, other_trj = model.path_generation(self.other_pos_index + 1)

        shortest_paths = self.find_all_shortest_paths(self.agent_pos, self.target_pos)
        selected_shortest_trj = self._select_distinguished_path(best_trj, shortest_paths)

        far_paths = self._select_far_path(misleading_regions)

        selected_far_trj = self._select_distinguished_path(best_trj, far_paths)


        misleading_trjs = self._select_misleading_path(other_trj, selected_shortest_trj)
        selected_misleading_trj = self._select_distinguished_path(best_trj, misleading_trjs)

        trjs_options = best_trj, selected_shortest_trj, selected_misleading_trj, selected_far_trj
        image_name = f'{self.data_type}_demo_{self.num_objects}'
        self.save_results(image_name, 'demo', model, trjs_options,
                          misleading_regions)

    def _initial_setting_prompt(self):

        setting = "Setting:\n" \
                  "A campus area is represented by a 5*5 grid. There are only two restaurants, X and Y on the campus. " \
                  "Student A attends school daily and is fully aware of the locations of each restaurant. " \
                  "He has a clear pre-established preference between X and Y, that is, **he decides to eat at restaurant X.** " \
                  "Observer B is an observer who monitors A's actions and is smart enough to infer A's preference **once it has been signaled**.\n\n"

        action = "Action:\n" \
                 "Student A can only take one step each time in four directions: up, down, left, and right. He wants to carefully plan his actions to achieve two goals.\n" \
                 "Primary goal: He wants to signal his preference (Restaurant X) to B as early as possible with the least ambiguity.\n" \
                 "Secondary goal: Once he thinks that the preference has been signaled, he will move to Restaurant X as soon as possible because he is hungry.\n\n"

        layout = "Layout:\n" \
                 "Below is one possible layout of the campus area. The letter \"A\" stands for Student A, \"*\" stands for empty areas, and \"W\" stands for obstructed walls that block the student. " \
                 "The top-left grid cell is designated as (0,0), the top-right as (4,0), the bottom-left as (0,4), and the bottom-right as (4,4). The letters \"X\" and \"Y\" stand for two restaurants.\n" \
                 "" + self.grid_layout_rep() + '\n\n'

        task = "Task: \n" \
               "Your task is to help A to choose the optimal action trajectory to achieve the above goals. " \
               "Please think step by step and select the most proper route. Also, calculate the number of steps required to achieve the primary goal. " \
               "Double-check your result to ensure it fulfills two goals.\n\n" \
               "Provide your answer according to the following format:\n" \
               "Best Route:\n" \
               "Steps for Signaling:\n" \
               "Explanation:\n\n"

        return setting + action + layout + task

    def _output_requirements(self):
        return "Please think step by step and select the best route. Also, calculate the number of steps required to achieve the primary goal. Double-check your result to ensure it fulfills two goals.\n\n" \
               "Provide your answer according to the following format:\n" \
               "Best Route:\n" \
               "Steps for Signaling: \n" \
               "Explanation: "

    def _select_distinguished_path(self, best_trj, paths):
        if len(paths) == 1:
            return paths[0]

        diff_index_list = []
        for path in paths:
            diff_index_list.append(self.diff_index(best_trj, path))
        assert len(diff_index_list) == len(paths)

        min_diff_paths = [
            paths[i] for i, v in enumerate(diff_index_list)
            if v == min(diff_index_list)
        ]
        sorted_paths = sorted(min_diff_paths,
                              key=lambda path: self.turn_count(path))
        return sorted_paths[0]

    def diff_index(self, best_trj, path):
        for i, (p1, p2) in enumerate(zip(best_trj, path)):
            if p1 != p2:
                return i
        return len(path)

    def _select_far_path_misleading(self, misleading_regions):
        grid_array = self.grid_array.copy()
        for pos in misleading_regions:
            c, r = pos
            grid_array[r][c] = 0
        far_paths = self.find_all_shortest_paths(self.agent_pos,
                                                 self.target_pos,
                                                 grid_array=grid_array)
        return far_paths

    def _select_far_path_flooding(self, *args, **kwargs):
        """Select far path version 2"""
        gmap = GridMap.construct_board(obstacle=self.obstacle_pos,
                                       silos=self.object_pos,
                                       origin=self.agent_pos,
                                       size=[self.world_size, self.world_size],
                                       trg_silo=self.targer_pos_index)

        model_far = StrategyAvoidant(gmap)
        far_paths = []
        for i in range(model_far.grid_map.silo_num):
            if model_far.grid_map.silos[i+1] != self.target_pos:
                continue
            _, path_far = model_far.path_generation(i + 1)
            far_paths += [path_far]

        return far_paths

    _select_far_path = _select_far_path_flooding

    def _select_misleading_path(self, other_best_trj, selected_shortest_path):
        misleading_trjs = []
        for path in self.find_all_shortest_paths(self.other_pos,
                                                 self.target_pos):
            if other_best_trj + path[1:] == selected_shortest_path:
                continue
            misleading_trjs.append(other_best_trj + path[1:])

        return misleading_trjs

    def _options_generate(self, best_trj, selected_far_trj,
                          selected_shortest_path, selected_misleading_path):
        # best
        trj_descs = []
        self.trj_direction_details_desc(best_trj, trj_descs)
        best_trj_desc = '\n'.join(trj_descs) + '\n'

        # shortest
        trj_descs = []
        self.trj_direction_details_desc(selected_shortest_path, trj_descs)
        selected_shortest_trj_desc = '\n'.join(trj_descs) + '\n'

        # misleading
        trj_descs = []
        self.trj_direction_details_desc(selected_misleading_path, trj_descs)
        misleading_trj_desc = '\n'.join(trj_descs) + '\n'

        all_trjs = [
            best_trj, selected_shortest_path, selected_misleading_path,
            selected_far_trj
        ]
        if len(set(map(lambda trj: tuple(trj), all_trjs))) < 4:
            return False, '', '', '', ''

        trj_descs = []
        self.trj_direction_details_desc(selected_far_trj, trj_descs)
        selected_far_trj_desc = '\n'.join(trj_descs) + '\n'

        return True, best_trj_desc, selected_far_trj_desc, selected_shortest_trj_desc, misleading_trj_desc

    def _prompt_for_gpt(self, options):
        prompt = self._initial_setting_prompt()

        d = {
            0: 'Route A:\n',
            1: 'Route B:\n',
            2: 'Route C:\n',
            3: 'Route D:\n'
        }
        options = [d[i] + option[0] for i, option in enumerate(options)]
        final_options = ''
        for option in options:
            final_options += option + '\n'
        prompt += final_options
        return prompt

    def _prompt_for_human(self, options):
        setting = f"Setting:\n" \
                  f"A campus area is represented by a 5*5 grid. There are only two restaurants, X and Y on the campus. Student A attends school daily and is fully aware of the locations of each restaurant. " \
                  f"He has a clear pre-established preference between X and Y, that is, he decides to eat at restaurant X. " \
                  f"Observer B is an observer who monitors A's actions and is smart enough to infer A's preference once it has been signaled.\n\n"
        action = f"Action:\n" \
                 f"Student A can only take one step each time in four directions: up, down, left, and right. He wants to carefully plan his actions to achieve two goals.\n" \
                 f"Primary goal: He wants to signal his preference (Restaurant X) to B as early as possible with the least ambiguity.\n" \
                 f"Secondary goal: Once he thinks that the preference has been signaled, he will move to Restaurant X as soon as possible because he is hungry.\n\n"

        layout = f"Layout:\n" \
                 f"Below is one possible layout of the campus area. The letter \"A\" stands for Student A, \"*\" stands for empty areas, and \"W\" stands for obstructed walls that block the student. " \
                 f"The top-left grid cell is designated as (0,0), the top-right as (4,0), the bottom-left as (0,4), and the bottom-right as (4,4). The letters \"X\" and \"Y\" stand for two restaurants. "


        task = f"Task:\n" \
               f"Your task is to help A to choose the optimal action trajectory to achieve the above goals. Also, calculate the number of steps required to achieve the primary goal."

        routes = f"Route A\n" \
                 f"{options[0][0]}\n" \
                 f"Route B\n" \
                 f"{options[1][0]}\n" \
                 f"Route C\n" \
                 f"{options[2][0]}\n" \
                 f"Route D\n" \
                 f"{options[3][0]}"
        return setting + action + layout, self.grid_layout_rep(), routes

    def _confusing(self, best_trj, shortest_paths, selected_shortest_path):
        # if best_trj in shortest_paths:
        #     return True
        if best_trj in shortest_paths and len(shortest_paths) == 1:
            return True
        if selected_shortest_path == best_trj:
            return True
        return False

    def best_trj_analysis(self, best_trj):
        if any(best_trj.count(p) > 1 for p in best_trj):
            if self.other_pos in best_trj:
                return BestTrjType.TURN_PASS
            return BestTrjType.TURN_NOTPASS
        else:
            if self.other_pos in best_trj:
                return BestTrjType.NOTTURN_PASS
            return BestTrjType.NOTTURN_NOTPASS

    def sample(self, N=5):

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        cnt = 0
        dataset = defaultdict(dict)

        while cnt < N:
            while True:
                self.grid_array = np.ones((self.world_size, self.world_size),
                                          dtype=np.uint8)
                self.sample_obstacles()
                self.sample_objects()
                self.sample_agent()

                paths = []
                for target_pos in self.object_pos:
                    path = self.dijkstras_path(self.agent_pos, target_pos)
                    if not path:
                        continue
                    paths.append(path)

                if len(paths) < self.num_objects:
                    continue

                if not all([len(path) >= 3 for path in paths]):
                    continue

                path1 = paths[0]
                path2 = paths[1]

                all_path1 = self.find_all_paths(self.grid_array,
                                                self.agent_pos,
                                                self.object_pos[0],
                                                base=len(path1))

                all_path2 = self.find_all_paths(self.grid_array,
                                                self.agent_pos,
                                                self.object_pos[1],
                                                base=len(path2))

                ret = self._valid(path1, all_path1, path2, all_path2)

                if not ret[0]:
                    continue

                if True:
                    self.targer_pos_index = ret[1]
                    self.target_pos = self.object_pos[ret[1]]
                    self.other_pos_index = 1 - ret[1]
                    self.other_pos = self.object_pos[1 - ret[1]]


                    gmap = GridMap.construct_board(
                        obstacle=self.obstacle_pos,
                        silos=self.object_pos,
                        origin=self.agent_pos,
                        size=[self.world_size, self.world_size],
                        trg_silo=self.targer_pos_index)
                    model = StrategyHybrid(gmap)

                    regions = model.coloring_saturated()
                    misleading_regions = regions[self.other_pos_index + 1]

                    paths = []

                    for i in range(model.grid_map.silo_num):
                        _, path = model.path_generation(i + 1)
                        paths += [path]

                    ritual_steps, best_trj = model.path_generation(self.targer_pos_index + 1)
                    best_trj_type = self.best_trj_analysis(best_trj).value

                    if not((best_trj_type == 0 and np.random.random() < 0.5)
                           or (best_trj_type == 1 and np.random.random() < 0.9)
                           or (best_trj_type == 2 and np.random.random() < 0.2)
                           or (best_trj_type == 3 and np.random.random() < 0.25)):
                        continue

                    _, other_trj = model.path_generation(self.other_pos_index + 1)

                    shortest_paths = self.find_all_shortest_paths(self.agent_pos, self.target_pos)
                    selected_shortest_trj = self._select_distinguished_path(best_trj, shortest_paths)



                    far_paths = self._select_far_path(misleading_regions)

                    selected_far_trj = self._select_distinguished_path(best_trj, far_paths)

                    if len(selected_far_trj) <= len(best_trj):
                        continue

                    misleading_trjs = self._select_misleading_path(other_trj, selected_shortest_trj)
                    if len(misleading_trjs) == 0:
                        continue

                    selected_misleading_trj = self._select_distinguished_path(best_trj, misleading_trjs)

                    # if len(selected_misleading_trj) <= len(
                    #         selected_shortest_trj):
                    #     continue

                    #
                    # if self._confusing(best_trj, shortest_paths, selected_shortest_trj):
                    #     continue

                    flag, best_trj_desc, selected_far_trj_desc, selected_shortest_trj_desc, misleading_trj_desc = self._options_generate(
                        best_trj, selected_far_trj, selected_shortest_trj,
                        selected_misleading_trj)

                    if not flag:
                        # todo corner cases；
                        continue

                    options = [(best_trj_desc, OptionType.BEST),
                               (selected_shortest_trj_desc,
                                OptionType.SHORTEST),
                               (misleading_trj_desc, OptionType.MISLEADING),
                               (selected_far_trj_desc, OptionType.FAR)]

                    np.random.shuffle(options)

                    gt = ','.join([option[1].value for option in options])

                    gpt_prompt = self._prompt_for_gpt(options)
                    human_prompt = self._prompt_for_human(options)
                    image_name = f'{self.data_type}_{cnt}_{self.num_objects}'

                    dataset[image_name]['gpt_prompt'] = gpt_prompt
                    dataset[image_name]['human_prompt'] = human_prompt

                    dataset[image_name]['gt'] = gt
                    dataset[image_name]['ritual_steps'] = ritual_steps
                    # dataset[image_name]['scene_type'] = ret[2].value
                    dataset[image_name]['best_trj_type'] = best_trj_type

                    trjs_options = best_trj, selected_shortest_trj, selected_misleading_trj, selected_far_trj
                    self.save_results(image_name, cnt, model, trjs_options,
                                      misleading_regions)

                    cnt += 1
                    now = datetime.now().strftime('%H:%M:%S')
                    print(f'{now}, cnt: {cnt}, type: {best_trj_type}, steps: {len(best_trj), len(selected_far_trj), len(selected_shortest_trj), len(selected_misleading_trj)}')
                    break

        ts = datetime.now().strftime('%m%d%H%M')
        filename = f'./data/{self.data_type}_{len(dataset)}_{self.num_objects}_one_shot_{ts}.json' \
            if self.one_shot else f'./data/{self.data_type}_{len(dataset)}_{self.num_objects}_zero_shot_{ts}.json'
        with open(filename, 'w') as f:
            json.dump(dataset, f)

    def save_results(self, image_name, cnt, model, trj_options,
                     misleading_regions):
        best_trj, selected_shortest_trj, selected_misleading_trj, selected_far_trj = trj_options
        image_name = os.path.join(self.image_dir, image_name + '.png')
        self.render(image_name)

        coloring_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_coloring.png')
        coloring_screen = self.render(coloring_image_name,
                                      color_board=model.color_board.transpose(
                                          1, 0, 2),
                                      save=True)
        coloring_trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_coloring_trj.png')
        self.render_iip_trj(coloring_trj_image_name,
                            best_trj,
                            screen=coloring_screen)

        inv_coloring_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_coloring_inv.png')
        inv_color_board = model.color_board.transpose(1, 0, 2)
        for c, r in misleading_regions:
            inv_color_board[r][c] = [.75, .75, .75, 1]
        inv_coloring_screen = self.render(inv_coloring_image_name,
                                          color_board=inv_color_board,
                                          save=True)
        inv_coloring_trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_coloring_inv_trj.png')
        self.render_iip_trj(inv_coloring_trj_image_name,
                            selected_far_trj,
                            screen=inv_coloring_screen)

        # best option
        trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_{OptionType.BEST.value}.png')
        self.render_iip_trj(trj_image_name, best_trj)

        # shortest option
        trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_{OptionType.SHORTEST.value}.png')
        self.render_iip_trj(trj_image_name, selected_shortest_trj)

        # misleading option
        trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_{OptionType.MISLEADING.value}.png')
        self.render_iip_trj(trj_image_name, selected_misleading_trj)

        # far option
        trj_image_name = os.path.join(
            self.image_dir,
            f'{self.data_type}_{cnt}_{self.num_objects}_{OptionType.FAR.value}.png')
        self.render_iip_trj(trj_image_name, selected_far_trj)


def cases():
    # obstacles = ((1, 1), (2, 1), (3, 1),
    #              (1, 2), (2, 2), (3, 2),
    #              (1, 3), (2, 3), (3, 3))
    # objects = ((0, 0), (1, 0))
    # target = (0, 0)
    # agent = (3, 0)

    # generator.load_setting(obstacles, objects, target, agent, id=0)
    # generator.load_setting(obstacles, objects, target, agent, id=0)
    # obstacles = ((1, 1),
    #              (1, 2))
    # objects = ((1, 4), (3, 4))
    # target = (1, 4)
    # agent = (2, 2)
    # generator.load_setting(obstacles, objects, target, agent, id=1)

    # obstacles = ((0, 1), (0, 2), (0, 3),
    #              (2, 4), (3, 4))
    # objects = ((0, 4), (2, 3))
    # target = (0, 4)
    # agent = (4, 4)
    # generator.load_setting(obstacles, objects, target, agent, id=3)

    # obstacles = ((0, 1), (0, 2), (0, 3),
    #              (2, 4), (3, 4))
    # objects = ((0, 4), (2, 3))
    # target = (0, 4)
    # agent = (4, 4)
    # generator.load_setting(obstacles, objects, target, agent, id=3)

    # obstacles = ((0, 0), (0, 1),
    #              (3, 1), (3, 2))
    # objects = ((1, 2), (0, 4))
    # target = (0, 4)
    # agent = (1, 0)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=0)

    # paper options demo
    # obstacles = ((0, 0), (0, 1), (0, 2), (0, 3),
    #              (2, 1), (2, 2), (2, 3))
    # objects = ((0, 4), (1, 2))
    # target = (1, 2)
    # agent = (3, 4)
    # generator.load_setting(obstacles, objects, target, agent, id=0)

    # oneshot cases for qualtrics
    # obstacles = ((1, 2), )
    # objects = ((3, 0), (4, 0))
    # target = (3, 0)
    # agent = (1, 4)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=-1)

    # supp: iip coloring
    # obstacles = [(0, 0), (0, 1), (0, 2), (0, 3),
    #              (2, 1), (2, 2), (2, 3)]
    # objects = ((0, 4), (1, 2))
    # target = (1, 2)
    # agent = (3, 4)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=-1)

    # obstacles = [(1, 1), (1, 2)]
    # objects = ((2, 1), (3, 0))
    # target = (2, 1)
    # agent = (0, 0)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=-1)

    # obstacles = [(2, 1), (2, 2), (0, 4), (1, 4), (2, 4), (3, 4)]
    # objects = ((1, 1), (0, 0))
    # target = (1, 1)
    # agent = (2, 0)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=-1)

    # supp, zeroshot
    # obstacles = [(0, 0), (0, 1), (3, 1), (3, 2)]
    # objects = ((0, 4), (1, 2))
    # target = (0, 4)
    # agent = (1, 0)
    #
    # generator.load_setting(obstacles, objects, target, agent, id=-1)
    pass


if __name__ == '__main__':
    # test dataset
    # np.random.seed(3456)
    # generator = IIPGenerator(num_obstacles=2)
    # shutil.rmtree(generator.image_dir)
    # generator.sample(500)

    # train dataset, 3500 -> filter to uniform distribution
    # np.random.seed(4567)
    # # generator = IIPGenerator(world_size=7, num_obstacles=3)
    # generator = IIPGenerator(num_obstacles=2)
    # shutil.rmtree(generator.image_dir)
    # generator.sample(3500)

    # val dataset
    np.random.seed(5678)
    # generator = IIPGenerator(world_size=7, num_obstacles=3)
    generator = IIPGenerator(num_obstacles=2)
    shutil.rmtree(generator.image_dir)
    generator.sample(1000)

