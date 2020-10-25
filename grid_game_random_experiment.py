import numpy as np
import pygame
import time
#from ccpomdp_class_new import CCPOMDP
from ccpomdp_class_combNode_LP_fullRound_prune import CCPOMDP
from grid_model_numba import grid_model
import IPython
import copy
import csv



def wall_counter(x, y, wall, grid_size):
    """for a given state(x,y) return number of walls around it (used to create the observation function)"""
    counter = 0
    if x == 0 or [x - 1, y] in wall:
        counter += 1
    if x == grid_size[0] - 1 or [x + 1, y] in wall:
        counter += 1
    if y == 0 or [x, y - 1] in wall:
        counter += 1
    if y == grid_size[1] - 1 or [x, y + 1] in wall:
        counter += 1
    return counter


#Grid_model = grid_model(grid_size=(3, 4), start=(0, 0), wall=[(1, 1)], risk=[(0, 2, 0.0), (0, 1, 0.5)], step_cost=-0.04,
#                         state_rewards=((2, 1, 5), (1, 3, -.001), (0, 2, .001)), sensor_accuracy=0.8, time_horizon=5, step_horizon=4, cc=0.1, optimization='Max')

#Grid_model = grid_model(grid_size=(3, 4), start=(1, 0), wall=[(2, 1),(0, 2)], risk=[(2, 2, 1.0), (0, 1, 1.0)], step_cost=-0.04,
#                         state_rewards=((1, 3, 5.0), (0, 3, -.001), (0, 2, .001)), sensor_accuracy=0.8, time_horizon=5, step_horizon=4, cc=0.3, optimization='Max')

#Grid_model = grid_model(grid_size=(3, 4), start=(0, 0), wall=[(1, 1)], risk=[(2, 2, 0.0), (0, 1, 0.5)], step_cost=0.0,
#                     state_rewards=[(2, 0, 5.0)], sensor_accuracy=0.8, time_horizon=6, step_horizon=6, soft_horizon=6, cc=0.3, optimization='max')
'''
Grid_model = grid_model(grid_size=(3, 4), start=(0, 0), wall=[(1, 1)], risk=[(2, 2, 0.0), (0, 1, 0.5)],
                        step_cost=0.0,
                        state_rewards=[(2, 1, 5.0)], sensor_accuracy=0.8, time_horizon=5, step_horizon=5,
                        soft_horizon=5, cc=0.1, optimization='Max')

policy = CCPOMDP(initial_belief=Grid_model.init_state, model=Grid_model)
grid_test(node_names=policy.node_names, model=Grid_model, sensor_accuracy=1.0)
'''
horizon = 5
grid_size = np.array([5, 6], dtype=np.int32).reshape(-1)
start = np.array([0, 0], dtype=np.int32).reshape(-1)
wall_list = np.array([[1, 4], [2, 2]], dtype=np.int32).reshape(-1, 2)
risk_list = np.array([[2, 3, 1.], [3, 2, 1.], [0, 1, 0.5], [1, 0, 0.1], [1, 1, 0.05]]).reshape(-1, 3)
state_rewards = np.array([[2, 1, 2.0], [3, 3, 5.0], [4, 5, 10.0]]).reshape(-1, 3)
# init model, and later just update values
Grid_model = grid_model(wall_list=wall_list, risk_list=risk_list, grid_size=grid_size, start=start,
                        step_cost=0.0, state_rewards=state_rewards, sensor_accuracy=0.8,
                        time_horizon=horizon, step_horizon=horizon, soft_horizon=horizon, cc=0.2,
                        optimization='Max')



# TODO: uncomment for first run
#with open('grid_game_experiment.csv', 'w', newline="") as file:
#    writer = csv.writer(file)
#    writer.writerow(['horizon', 'wall states', 'risky states', 'reward states', 'grid_size', 'risk', 'node_count', 'prune', 'node_count_prune', 'expand_tree_time',
#                     'prune_time', 'LP_time', 'LP_rounds_min', 'LP_rounds', 'ILP_time', 'LP_objective', 'ILP_objective', 'integral_solution', 'greedy_iterations', 'greedy_advanced_upper'])


"""
horizon_list = []
grid_size_list = []
risk_val_list = []
node_count_list = []
prune_list = []  #
node_count_prune_list = []
expand_tree_time = []
prune_time = []
LP_time = []
LP_rounds_min = []
LP_rounds_iteration = []
ILP_time_list = []
LP_obj_list = []
ILP_obj_list = []
integral_sol_list = []
greedy_iteration_min_list = []  # iteration count
greedy_advanced_list = []
"""
max_iteration = 0
try:
    j=0
    while j <100000:
        print('test case', j)

        horizon = np.random.randint(2, 7)
        grid_size = np.array([np.random.randint(3, 7), np.random.randint(3, 7)], dtype=np.int32)
        state_list = [[x, y] for x in range(grid_size[0]) for y in range(grid_size[1])]

        start_idx = np.random.choice(len(state_list))
        start = state_list[start_idx]
        state_list.pop(start_idx)
        start = np.asarray(start, np.int32)
        ##############################################################
        wall_count = np.random.randint(0, len(state_list)//3)
        wall_list = []
        for i in range(wall_count):
            wall = np.random.choice(len(state_list))
            wall_list.append(state_list[wall])
            state_list.pop(wall)
        wall_list = np.asarray(wall_list, dtype=np.int32)

        # TODO: check that start state is not traped, and avoid four wall states (use wall counter function verify)
        wall_set = [wall_counter(x, y,wall_list, grid_size) for x in range(grid_size[0]) for y in range(grid_size[1])]
        if 4 in wall_set:
            continue

        # add walls, and remaining state could have reward, risk or both

        risk_state_list = copy.copy(state_list)
        reward_state_list = copy.copy(state_list)
        #############################################################
        risk_count = np.random.randint(0, len(state_list) // 3)
        risk_list = []  # to avoid an empty list
        for i in range(risk_count):
            risk_idx = np.random.choice(len(risk_state_list))
            risk_list.append(risk_state_list[risk_idx]+[np.random.random()])
            risk_state_list.pop(risk_idx)
        risk_list = np.asarray(risk_list, np.float64)

        #############################################################
        reward_count = np.random.randint(0, len(state_list) // 2)
        state_rewards = []  # to avoid an empty list
        for i in range(reward_count):
            reward_idx = np.random.choice(len(reward_state_list))
            state_rewards.append(reward_state_list[reward_idx]+[np.random.random()*5])
            reward_state_list.pop(reward_idx)
        state_rewards = np.asarray(state_rewards, np.float64)

        cc = np.random.random()*.5

        wall_list = None if len(wall_list) == 0 else wall_list
        risk_list = None if len(risk_list) == 0 else risk_list
        state_rewards = None if len(state_rewards) == 0 else state_rewards

        step_cost = 0-np.random.random()*.1
        sensor_accuracy = .85

        step_cost = 0.  # TODO: remove
        #horizon = 2

        Grid_model.update_params(wall_list, risk_list, state_rewards, grid_size,
                             start, step_cost, sensor_accuracy,
                            horizon, horizon, horizon, cc, 'Max')

        #prune = np.random.choice([True, False])
        #prune = False

        prune = j%2 == 0  # prune half of cases
        rounding_min_iteration = 100 if j//2%2 == 0 else 0
        greedy_iteration =10 if j//2%4 == 0 else 0  # use greedy half of the cases once with prune and without
        greedy_advanced = j//2%8 == 0  # in greedy cases once advanced and once not

        prune = True  # TODO: remove
        rounding_min_iteration = 100
        greedy_iteration = 10 if j % 4 == 0 else 0
        greedy_advanced = False

        if horizon == 6 and len(Grid_model.obs_function(0)) >= 4:
            continue  # too much for the solver

        print(f'horizon:{horizon}, obs:{len(Grid_model.obs_function(0))}')
        p = CCPOMDP(initial_belief=Grid_model.init_state, model=Grid_model, prune=prune, rounding_min_iteration=rounding_min_iteration, greedy_iteration=greedy_iteration, greedy_advanced=greedy_advanced)

        if not p.feasible or p.tree_expand_time > 35:
            continue

        #['horizon', 'wall states', 'risky states', 'reward states', 'grid_size', 'risk', 'node_count', 'prune',
        # 'node_count_prune', 'expand_tree_time',
        # 'prune_time', 'LP_time', 'LP_rounds_min', 'LP_rounds', 'ILP_time', 'LP_objective', 'ILP_objective',
        # 'integral_solution', 'greedy_iterations(0 or 10)', 'greedy_advanced_upper']

        row = [horizon, 0 if wall_list is None else len(wall_list), 0 if risk_list is None else len(risk_list), 0 if state_rewards is None else len(state_rewards), grid_size, cc, p.node_count, prune, p.prune_node_count,
               p.tree_expand_time, p.prune_time, p.LP_time, rounding_min_iteration, p.round_iteration, p.ILP_time,
               p.obj_LP, p.obj_ILP, p.integral_sol, greedy_iteration, greedy_advanced]

        print(row)

        max_iteration = max(max_iteration, p.round_iteration)
        print('max iteration', max_iteration)
        j += 1
        #if p.obj_LP < p.obj_ILP/len(Grid_model.obs_function(0))**(horizon-1) and p.obj_ILP>0:
        #    IPython.embed(header='obj')
        with open('grid_game_experiment.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)



        """
        #if not policy.round_iteration:
        horizon_list.append(horizon)
        LP_time.append(np.round(policy.LP_time, decimals=3))
        #LP_iteration.append(policy.LP_iteration)
        ILP_time.append(np.round(policy.ILP_time, decimals=3))
        ILP_diff_count.append(policy.ILP_diff_count)
        obj_LP.append(np.round(policy.obj_LP, decimals=3))
        obj_ILP.append(np.round(policy.obj_ILP, decimals=3))
        integral_sol.append(policy.integral_sol)
        #round_iteration.append(policy.round_iteration)
        #print('max iteration:', max(round_iteration))
        print('avg LP to ILP:', sum(obj_LP) / sum(obj_ILP))
        print('avg time ratio:', sum(LP_time) / sum(ILP_time))

        """
except KeyboardInterrupt:
    print('error occurred')
"""
for i in range(len(LP_time)):
    print('feasible:{}, iter:{}, first_iter:{}, LP_Time:{}, ILP_time:{}, diff:{}, obj_LP:{}, obj_ILP:{}'.format(integral_sol[i], round_iteration[i], LP_iteration[i], LP_time[i], ILP_time[i], ILP_diff_count[i], obj_LP[i], obj_ILP[i]))

print('feasibility percentage:', sum(integral_sol) / len(integral_sol))
#print('max iteration:', max(round_iteration))
print('avg obj LP:', sum(obj_LP)/len(obj_LP))
print('avg obj ILP:', sum(obj_ILP)/len(obj_ILP))
print('avg LP to ILP:', sum(obj_LP)/sum(obj_ILP))

print('avg time ratio:', sum(LP_time)/sum(ILP_time))

min_obj_ratio = 1.
for i in range(len(obj_LP)):
    if obj_ILP[i] >= 0.0001:
        min_obj_ratio = min(min_obj_ratio, obj_LP[i]/obj_ILP[i])

print('min LP to ILP:', min_obj_ratio)
"""
IPython.embed(header='LP test Done')
