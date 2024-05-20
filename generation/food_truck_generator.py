import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from collections import defaultdict, Counter
from enum import unique, Enum
import shutil
from generator import Generator
from utils.consts import I2C_MAP, INT2STR_MAP
from utils.bfs import get_observable_neighbors, get_observable_targets


@unique
class FTSceneType(Enum):
    # 'direct picks up the object without/before visiting all'
    DIRECT = 0
    # 'picks up the last object after visiting all'
    FINAL = 1
    # 'after visiting all turn back to pick up the object that were passed by'
    TURN = 2


class FTGenerator(Generator):
    def __init__(self, world_size=5, num_obstacles=2, num_objects=4, one_shot=False):
        super(FTGenerator, self).__init__(world_size, num_obstacles, num_objects)
        self.data_type = 'FOOD_TRUCK'
        self.image_dir = './data/FOOD_TRUCK'
        self.one_shot = one_shot
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def _valid(self):
        for i in range(self.num_objects):
            for j in range(i+1, self.num_objects):
                path = self.dijkstras_path(self.object_pos[i], self.object_pos[j])
                if len(path) <= 2:
                    return False
        return self._obstacle_check()

    def _obstacle_check(self):

        fake_grid = np.ones((self.world_size, self.world_size), dtype=np.uint8)
        cnt = 0
        for i in range(self.num_objects):
            min_path = self.dijkstras_path(self.agent_pos, self.object_pos[i])
            all_paths = self.find_all_paths(fake_grid, self.agent_pos, self.object_pos[i])
            min_paths = [path for path in all_paths if len(path) == len(min_path)]

            if min_path in min_paths:
                cnt += 1
        return cnt < self.num_objects // 2

    def __next_target(self, cur_pos, visited):
        min_steps = 1000
        target = None
        visit = None
        for i, obj in enumerate(self.object_pos):
            if i in visited:
                continue
            neighbors = get_observable_neighbors(self.grid_array, obj)
            for pos in neighbors:
                steps = len(self.dijkstras_path(cur_pos, pos))
                if steps != 0 and steps < min_steps:
                    min_steps = steps
                    visit = i
                    target = pos
        return target, visit

    def load_setting(self, agent_pos, object_pos, obstacle_pos):
        self.agent_pos = agent_pos
        self.object_pos = object_pos
        self.obstacle_pos = obstacle_pos
        for pos in self.obstacle_pos:
            c, r = pos
            self.grid_array[r][c] = 0

    def gen_atomic_trj_desc_vo(self, trj_tuple):
        trj, pick = trj_tuple

        # view only
        desc = f"Here is the Agent's trajectory. The coordinates reflect the position of the Agent. Each time agent can move one step. " \
               f"View indicates the food within the current view range. Pick indicates the Agent's final choice.\n"

        for pos in trj[:-1]:
            new_observables = get_observable_targets(self.grid_array, pos, self.object_pos, radius=1)
            if len(new_observables) == 0:
                desc += f'{pos}\n'
            else:
                object_desc = ','.join([I2C_MAP[obs] for obs in new_observables])
                desc += f'{pos} - view {object_desc}\n'
        new_observables = get_observable_targets(self.grid_array, trj[-1], self.object_pos, radius=1)
        object_desc = ','.join([I2C_MAP[obs] for obs in new_observables])
        desc += f'{trj[-1]} - view {object_desc}; pick {pick} \n'
        return desc

    def gen_atomic_trj_desc_mo(self, trj_tuple):
        trj, pick = trj_tuple

        # memory only
        desc = f"Here is the Agent's trajectory. The coordinates reflect the position of the Agent. Each time agent can move one step. " \
               f"Memory indicates all the past viewed food. Pick indicates the Agent's final choice.\n"
        memory_obs_list = []
        for pos in trj[:-1]:
            new_observables = get_observable_targets(self.grid_array, pos, self.object_pos, radius=1)
            for new_obs in new_observables:
                if new_obs not in memory_obs_list:
                    memory_obs_list.append(new_obs)
            if len(memory_obs_list) == 0:
                desc += f'{pos}\n'
            else:
                object_desc = ','.join([I2C_MAP[obs] for obs in memory_obs_list])
                desc += f'{pos} - memory {object_desc}\n'

        new_observables = get_observable_targets(self.grid_array, trj[-1], self.object_pos, radius=1)

        for new_obs in new_observables:
            if new_obs not in memory_obs_list:
                memory_obs_list.append(new_obs)

        object_desc = ','.join([I2C_MAP[obs] for obs in memory_obs_list])
        desc += f'{trj[-1]} - memory {object_desc}; pick {pick} \n'
        return desc

    def gen_atomic_trj_desc_both(self, trj_tuple):
        trj, pick = trj_tuple

        desc = f"Student A's Trajectory:\n" \
               f"Here is the student A's trajectory. The coordinates reflect the position of the A. Each time agent can move one step.\n"

        memory_obs_list = []
        for pos in trj[:-1]:
            new_observables = get_observable_targets(self.grid_array, pos, self.object_pos, radius=1)

            for new_obs in new_observables:
                if new_obs not in memory_obs_list:
                    memory_obs_list.append(new_obs)

            if len(new_observables) == 0 and len(memory_obs_list) == 0:
                desc += f'{pos}\n'
            elif len(new_observables) == 0:
                memory_desc = ','.join([I2C_MAP[obs] for obs in memory_obs_list])
                desc += f'{pos} memory {memory_desc}\n'
            else:
                memory_desc = ','.join([I2C_MAP[obs] for obs in memory_obs_list])
                object_desc = ','.join([I2C_MAP[obs] for obs in new_observables])
                desc += f'{pos} view {object_desc}; memory {memory_desc}\n'

        new_observables = get_observable_targets(self.grid_array, trj[-1], self.object_pos, radius=1)
        object_desc = ','.join([I2C_MAP[obs] for obs in new_observables])

        for new_obs in new_observables:
            if new_obs not in memory_obs_list:
                memory_obs_list.append(new_obs)

        memory_desc = ','.join([I2C_MAP[obs] for obs in memory_obs_list])
        desc += f'{trj[-1]} view {object_desc}; memory {memory_desc}; pick {pick} \n\n'
        return desc

    def gen_trajectory(self, image_index, preference):
        trj = [self.agent_pos]
        visited = []
        visited_step_list = []

        # old
        description = f"Here is the trajectory of Agent's exploration. Agent starts at {self.agent_pos}."

        check_all = False if preference[0] in range(self.num_objects) else True

        new_observables = get_observable_targets(self.grid_array, self.agent_pos, self.object_pos, radius=1)

        if len(new_observables) > 0:
            object_desc = ','.join([I2C_MAP[obs] for obs in new_observables])
            description = description[:-1] + ','
            description += f' and sees {object_desc}.'

        if preference[0] in new_observables:
            obs = preference[0]
            path = self.dijkstras_path(self.agent_pos, self.object_pos[obs])
            trj.extend(path[1:])

            trj_descs = []
            self.trj_direction_desc(path, trj_descs)

            for step_desc in trj_descs:
                description += f' A {step_desc},'

            description += f' and A picks up {I2C_MAP[obs]}.'
            if len(visited) == 0 or visited[-1] != obs:
                visited.append(obs)
                visited_step_list.append([obs])
            return description, (trj, I2C_MAP[obs]), visited, visited_step_list

        step_obs_lis = []
        for obs in new_observables:
            if obs not in visited:
                visited.append(obs)
                step_obs_lis.append(obs)
        if len(step_obs_lis) > 0:
            visited_step_list.append(step_obs_lis)

        trg_pos, visit = self.__next_target(self.agent_pos, visited)
        cur_pos = self.agent_pos
        while trg_pos:
            # visited.append(visit)
            path = self.dijkstras_path(cur_pos, trg_pos)
            trj_descs = []
            self.trj_direction_desc(path, trj_descs)
            # description += f' A moves as follows: {path[1:]}.'
            for step_desc in trj_descs:
                description += f' A {step_desc},'
            description = description[:-1] + '.'

            discover_list_desc = ''
            new_observables = get_observable_targets(self.grid_array, trg_pos, self.object_pos, radius=1)
            for obs in new_observables:
                if obs not in visited:
                    discover_list_desc += f'{I2C_MAP[obs]},'
            if discover_list_desc:
                description = description[:-1] + ','
                description += f' and sees {discover_list_desc[:-1]}.'

            cur_pos = trg_pos
            trj.extend(path[1:])

            if preference[0] in new_observables:
                obs = preference[0]
                path = self.dijkstras_path(cur_pos, self.object_pos[obs])
                trj.extend(path[1:])

                trj_descs = []
                self.trj_direction_desc(path, trj_descs)

                for step_desc in trj_descs:
                    description += f' A {step_desc},'

                description += f' and A picks up {I2C_MAP[obs]}.'
                if len(visited) == 0 or visited[-1] != obs:
                    visited.append(obs)
                    visited_step_list.append([obs])
                return description, (trj, I2C_MAP[obs]), visited, visited_step_list

            step_obs_list = []
            for obs in new_observables:
                if obs not in visited:
                    visited.append(obs)
                    step_obs_list.append(obs)
            if len(step_obs_list) > 0:
                visited_step_list.append(step_obs_list)

            trg_pos, visit = self.__next_target(cur_pos, visited)

        pick = ''

        if check_all:

            second = preference[1]

            if cur_pos != self.object_pos[second]:
                path = self.dijkstras_path(cur_pos, self.object_pos[second])
                trj.extend(path[1:])

                trj_descs = []
                self.trj_direction_desc(path, trj_descs)

                for step_desc in trj_descs:
                    description += f' A {step_desc},'

                if visited[-1] != second:
                    visited.append(second)
                    visited_step_list.append([second])
            else:
                assert second in visited
                assert len(visited) < len(preference)
                index = visited.index(second)
                visited.append(visited.pop(index))
                if len(visited) == len(visited_step_list):
                    visited_step_list.append(visited_step_list.pop(index))
                print(image_index, preference, 'cur_pos == self.object_pos[second]')

            description += f' and A picks up {I2C_MAP[second]}.'
            pick = I2C_MAP[second]

        assert pick != ''
        return description, (trj, pick), visited, visited_step_list

    def ground_truth_base_rule(self, preference, visited, visited_step_list):
        s1 = [I2C_MAP[p] for p in preference]
        s2 = [I2C_MAP[v] for v in visited]

        chosen = visited[-1]
        assert chosen == preference[0] or chosen == preference[1]
        assert len(visited) <= len(preference)

        def direct_go():
            n = len(visited_step_list)
            assert chosen in visited_step_list[n-1]
            if n >= 2:
                if chosen in visited_step_list[n-2]:
                    return True
            return False

        if len(visited) < len(preference) - 1:
            assert preference[0] in visited
            assert preference[0] == visited[-1]
            # other = ','.join([I2C_MAP[v] for v in visited[:-1]])
            other = ','.join([I2C_MAP[v] for v in preference[1:]])
            other = '{' + other + '}'
            return f'{I2C_MAP[chosen]} > {other}', FTSceneType.DIRECT

        if len(visited) == len(preference) - 1:
            other = ','.join([I2C_MAP[v] for v in visited[:-1]])
            other = '{' + other + '}'
            return f'{I2C_MAP[chosen]} > {other}', FTSceneType.FINAL

        if len(visited) == len(preference):
            # assert preference[0] not in visited
            # assert preference[1] == visited[-1]
            # if go_back:
            if not direct_go():
                # go back
                assert visited.count(chosen) == 2
                other = ','.join([I2C_MAP[v] for v in visited[:-1] if (v != chosen and v != preference[0])])
                other = '{' + other + '}'
                return f'{I2C_MAP[preference[0]]} > {I2C_MAP[chosen]} > {other}', FTSceneType.TURN
            else:
                other = ','.join([I2C_MAP[v] for v in visited[:-1] if v != chosen])
                other = '{' + other + '}'
                return f'{I2C_MAP[chosen]} > {other}', FTSceneType.FINAL
        return ''

    def gen_prompt(self, image_index, N=5):

        cnt = 0

        visited_preferences = set()
        visited_trj = set()
        loop_cnt = 0

        scene_name = f'{self.data_type}_{image_index}'
        self.render(f'./data/{self.data_type}/{scene_name}.png')

        while cnt < N:
            if loop_cnt >= 100:
                break
            loop_cnt += 1
            preference = np.random.choice(self.num_objects + 1, self.num_objects + 1, replace=False)
            if tuple(preference) in visited_preferences:
                continue

            trj_desc, trj_tuple, visited, visited_step_list = self.gen_trajectory(image_index, preference)
            trj = trj_tuple[0]

            if tuple(trj) in visited_trj:
                continue

            if trj.count(max(set(trj), key=trj.count)) >= 3:
                continue

            visited_preferences.add(tuple(preference))
            visited_trj.add(tuple(trj))

            # atomic_trj_desc = self.gen_atomic_trj_desc_vo(trj_tuple)
            # atomic_trj_desc = self.gen_atomic_trj_desc_mo(trj_tuple)
            atomic_trj_desc = self.gen_atomic_trj_desc_both(trj_tuple)
            gt, scene_type = self.ground_truth_base_rule(preference, visited, visited_step_list)
            image_name = f'{self.data_type}_{image_index}_{cnt}'
            # self.render_with_trj(f'./data/{self.data_type}/{image_name}.png', trj, preference)
            self.draw_ft_trj(f'./data/{self.data_type}/{image_name}.png', trj)

            gpt_prompt = self._prompt_for_gpt(atomic_trj_desc)
            human_prompt = self._prompt_for_human(atomic_trj_desc)

            cnt += 1
            yield scene_name, preference, gpt_prompt, human_prompt, gt, scene_type

    def _prompt_for_gpt(self, atomic_trj_desc):

        # 3(2+1),4(3+1),5(4+1)
        types_desc = ''
        if self.num_objects == 2:
            types_desc = 'X, Y and Z'
        elif self.num_objects == 3:
            types_desc = 'X, Y, Z and M'
        elif self.num_objects == 4:
            types_desc = 'X, Y, Z, M and N'

        prefix = ''

        setting = f"Setting:\n" \
                 f"Imagine a 5x5 grid representing a campus food truck area. " \
                 f"Student A, who attends school daily, is familiar with the food trucks in the school area. " \
                 f"He knows there are usually **five food trucks**, labeled {types_desc}, " \
                 f"but due to limited parking space, **only four trucks** can park each day.\n\n"

        constraint = f"Constraint:\n" \
                     f"Student A knows where the food trucks usually park, but he only finds out what specific food they're selling when they're close enough for him to see. " \
                     f"His **view range** is limited to his current grid and the eight grids around him - one in each direction: north, northeast, east, southeast, south, southwest, west, and northwest. " \
                     f"His **memory** includes all the foods that he has discovered till now.\n\n"

        action = f"Action:\n" \
                 f"Student A goes to the food truck square every day to buy lunch, " \
                 f"**so he has a clear preference for five specific foods while there are no equal food preferences.** " \
                 f"Assuming that Student A's behavior conforms to the rational person hypothesis, " \
                 f"he aims to maximize his benefits by choosing the food he likes the most among the available options.\n\n"

        prefix += setting
        prefix += constraint
        prefix += action

        # if self.one_shot:

        postfix = ''

        task = f"Task:\n" \
               f"Your task is to analyze Student A's trajectory and determine his preferences for the five types of food (**Note: Please include all five types of food in your final answer**). " \
               f"Please think carefully and present the preference order using \">\" symbols. " \
               f"If you cannot determine the preference order for some foods, group them in {{}}. " \
               f"If you believe there are multiple preference orders in a scenario, separate them using \";\".\n\n"

        answer_demo = f"Some answer examples:\n" \
                      f"**X > Y > {{Z, M, N}}**\n" \
                      f"It indicates that X is ranked first in preferences, superior to all foods, and Y is ranked second, only next to X, superior to all remaining foods, but the preference order of Z, M, and N cannot be determined.\n" \
                      f"**N>X>Y>Z, {{M}}**\n" \
                      f"It indicates that it can be determined that N>X>Y>Z, but M is uncertain, that is, it could be placed in any position.\n" \
                      f"**X>Y>Z>M>N; N>X>Y>M>Z**\n" \
                      f"It indicates that there may exist two responses that cannot be integrated."

        postfix += task
        postfix += answer_demo

        layout = f"Layout:\n" \
                 f"Below is one possible layout of the food truck area. The letter \"A\" stands for Student A, \"*\" stands for empty areas, and \"W\" stands for obstructed walls that block the student. " \
                 f"Other letters represent different kinds of food. We're assuming the top left corner is (0,0), top right is (4,0), bottom left is (0,4), and bottom right is (4,4).\n" \
                 f"{self.grid_layout_rep()}\n\n"
        total_prompt = ''
        total_prompt += prefix
        total_prompt += layout
        total_prompt += atomic_trj_desc
        total_prompt += postfix

        return total_prompt

    def _prompt_for_human(self, atomic_trj_desc):
        question = f"Question:\n" \
                   f"Please follow the instructions to answer the question. Below is one possible layout of the food truck area. The letter \"A\" stands for Student A, \"*\" stands for empty areas, and \"W\" stands for obstructed walls that block the student. Other letters represent different kinds of food.\n\n" \
                   f"We're assuming the top left corner is (0,0), top right is (4,0), bottom left is (0,4), and bottom right is (4,4). Here is student A's trajectory. The coordinates reflect the position of the A. Each time student A can move one step. Please determine the preference among all the five foods and provide your answer following the format.\n\n"

        layout = f"Layout:\n" \
                 f"{self.grid_layout_rep()}\n\n"

        trajectory = f"{atomic_trj_desc}"

        answer = 'Please determine the preference among all the five foods and provide your answer following the format.'

        return question + layout + trajectory + answer

    def _one_shot_example(self):
        example = f"Here is an example. The grid layout is shown below. The Agent is represented as 'A'. Empty spaces are marked with '*'. Obstructed walls are marked with 'W'. " \
               f"Different letters show various kinds of food. We're assuming the top left corner is (0,0), top right is (4,0), bottom left is (0,4), and bottom right is (4,4). " \
               f"The conclusion is a preference order of X, Y, Z, M, and N that you inferred based on the trajectory of the Agent. \">\" and \"=\" are used to represent the preference order of Agent towards different kinds of food.\n"
        example += "**Z*M\n" \
                   "****W\n" \
                   "AWWWW\n" \
                   "X***W\n" \
                   "WW*Y*\n"
        example += "Here is the trajectory of Agent's exploration. " \
                   "Agent starts at (0, 2), and sees X. A goes up to (0, 1), A goes right to (1, 1), and sees Z. " \
                   "A goes right to (3, 1), and sees M. A goes left to (0, 1), A goes down to (0, 3), A goes right to (2, 3), and sees Y. " \
                   "A goes left to (0, 3), A goes up to (0, 0), A goes right to (4, 0), and A picks up M. " \
                   "We can infer that the agent's preference is: N > M > {X,Z,Y}, the reasoning process is as follows.\n"
        example += "1)Agent starts at (0, 2) and sees X, but doesn't pick it up, so X is not Agent's favorite food.\n" \
                   "2)A goes up to (0, 1), no food information obtained.\n" \
                   "3)A goes right to (1, 1) and sees Z, but doesn't pick it up, so Z is not Agent's favorite food.\n" \
                   "4)A goes right to (3, 1) and sees M, but doesn't pick it up, so M is not Agent's favorite food.\n" \
                   "5)A goes left to (0, 1), no food information obtained.\n" \
                   "6)A goes down to (0, 3), no food information obtained.\n" \
                   "7)A goes right to (2, 3) and sees Y but doesn't pick it up, so Y is not Agent's favorite food. " \
                   "Now Agent knows all four kinds of food: X, Z, M, and Y. " \
                   "However, he does not pick up any of these foods, indicating that none of X, Z, M, and Y are the Agent's favorite food. " \
                   "Therefore, N is the Agent's favorite food.\n" \
                   "8)A goes left to (0, 3), A goes up to (0, 0), A goes right to (4, 0), and A picks up M. " \
                   "Agent returns to the location of M and picks it up, indicating that among the four kinds of food available: X, Z, M, and Y, Agent has a stronger preference for M. " \
                   "But we can not determine the preference order of X, Z, Y. Therefore, the final result is N > M > {X, Z, Y}.\n"
        return example

    def sample(self, N=5):

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        dataset_dict = defaultdict(lambda: defaultdict(dict))
        for i in tqdm(range(N)):
            while True:
                self.grid_array = np.ones((self.world_size, self.world_size), dtype=np.uint8)
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

                if self._valid():
                    # self.render(f'./data/{self.dataset_type}_{i}.png')
                    for k, (image_name, preference, gpt_prompt, human_prompt, gt, scene_type) in enumerate(self.gen_prompt(i, N=5)):
                        dataset_dict[image_name][k]['gpt_prompt'] = gpt_prompt
                        dataset_dict[image_name][k]['human_prompt'] = human_prompt
                        dataset_dict[image_name][k]['preference'] = '>'.join([I2C_MAP[p] for p in preference])
                        print(dataset_dict[image_name][k]['preference'])
                        dataset_dict[image_name][k]['gt'] = gt
                        dataset_dict[image_name][k]['scene_type'] = scene_type.value
                    break

        dataset_dict = dict(dataset_dict)
        ts = datetime.now().strftime('%m%d%H%M')

        with open(f'./data/{self.data_type}_{len(dataset_dict)}_{self.num_objects}_{"one_shot" if self.one_shot else "zero_shot"}_{ts}.json', 'w') as f:
            json.dump(dataset_dict, f)

def cases():
    # ft_gen = FTGenerator()
    # agent_pos = (3, 0)
    # object_pos = [(0, 3), (4, 1), (4, 4), (3, 3)]
    # obstacle_pos = [(0, 1), (1, 1), (2, 1),
    #                 (1, 3), (2, 3),
    #                 (0, 4), (1, 4), (2, 4), (3, 4)]
    # ft_gen.load_setting(agent_pos, object_pos, obstacle_pos)
    # for _ in ft_gen.gen_prompt(-1, N=20):
    #     print()

    # paper table
    # paper supp
    # agent_pos = (4, 0)
    # object_pos = [(1, 2), (1, 4), (3, 0), (4, 3)]
    # obstacle_pos = [(1, 0), (1, 1), (1, 3), (2, 3), (3, 3)]
    #
    # ft_gen.load_setting(agent_pos, object_pos, obstacle_pos)
    # for _ in ft_gen.gen_prompt(-1, N=20):
    #     print()

    # paper supp
    # agent_pos = (4, 4)
    # object_pos = [(2, 2), (3, 0), (1, 4), (0, 3)]
    # obstacle_pos = [(2, 3), (3, 3), (2, 4), (3, 4)]
    #
    # ft_gen.load_setting(agent_pos, object_pos, obstacle_pos)
    # for _ in ft_gen.gen_prompt(-1, N=20):
    #     print()

    # supp zero shot
    # agent_pos = (0, 4)
    # object_pos = [(2, 1), (3, 3), (0, 3), (3, 0)]
    # obstacle_pos = [(2, 2), (2, 3), (4, 1), (4, 2), (4, 3)]
    # ft_gen.load_setting(agent_pos, object_pos, obstacle_pos)
    # for _ in ft_gen.gen_prompt(-1, N=20):
    #     print()
    pass


if __name__ == '__main__':
    # test dataset
    np.random.seed(1234)
    ft_gen = FTGenerator()
    shutil.rmtree(ft_gen.image_dir)
    ft_gen.sample(100)

    # train dataset
    # np.random.seed(3456)
    # ft_gen = FTGenerator()
    # shutil.rmtree(ft_gen.image_dir)
    # ft_gen.sample(1000)

    # val dataset
    # np.random.seed(5678)
    # ft_gen = FTGenerator()
    # shutil.rmtree(ft_gen.image_dir)
    # ft_gen.sample(300)


