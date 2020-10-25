import IPython
import numpy as np
import pygame
import time
from ccpomdp_class_combNode_robust import CCPOMDP_robust
from ccpomdp_class_FPTAS import CCPOMDP_FPTAS
from grid_model_numba import grid_model
from numba import njit
pygame.init()


def policy_test(node_list, model, risk_error_range, risk_error_probability, count):
    objective_list = []  # list of objective per run
    risk_bool_list = []  # list of risk bool per run
    observation_probability = []  # counter for each observation
    for node in node_list:
        observation_probability.append(np.zeros(len(node.safe_post_probability), np.int32))
    print('running experiment')
    for _ in range(count):
        risk_bool, objective = grid_test(model, node_list, observation_probability, risk_error_range, risk_error_probability)
        objective_list.append(objective)
        risk_bool_list.append(risk_bool)
    print('computing risk')
    error_list = []
    for i, node in enumerate(node_list):
        if node.x_val == 1 and not node.terminal:
            error = np.sum(np.abs(observation_probability[i]/sum(observation_probability[i]) - node.safe_post_probability))/len(observation_probability[i])
            error_list.append(error)

    risk_probability = sum(risk_bool_list) / len(risk_bool_list)
    avg_objective = sum(objective_list) / len(objective_list)
    probability_error = sum(error_list) / len(error_list)
    print('Policy Test:-------')
    print('risk_probability', risk_probability)
    print('avg_objective', avg_objective)
    print('probability_error', probability_error)


def grid_test(model, node_list, observation_probability, risk_error_range, risk_error_probability):
    # grid has position of real robot
    # belief in in current_node.state
    risk_bool = False
    grid = np.copy(model.init_state)
    robot_location = np.where(grid == 1.0)
    loc_x = robot_location[0][0]
    loc_y = robot_location[1][0]
    #current_node = 'r'
    current_node_idx = 0
    observ = 0
    objective = 0.
    objective += model.compute_reward(grid, 0)
    #print(grid[::-1])
    observation_probability[current_node_idx][observ] += 1
    count = 0
    while not node_list[current_node_idx].terminal and count < 50:
        #best_action = node_names[current_node].best_action
        best_action = node_list[current_node_idx].best_action[observ]
        if int(best_action) == -1:
            print('no best action assigned')
            break

        #current_node += str(int(best_action))
        current_node_idx = node_list[current_node_idx].child_list[observ][best_action].BFS_index


        # move robot with given transition probability
        if int(best_action) == 0:
            act_list = np.array([0, 1, 2, 3])
        elif int(best_action) == 1:
            act_list = np.array([1, 0, 3, 2])
        elif int(best_action) == 2:
            act_list = np.array([2, 3, 1, 0])
        elif int(best_action) == 3:
            act_list = np.array([3, 2, 0, 1])

        act_choice = np.random.choice(act_list, p=model.trans_prob)

        if act_choice == 0 and loc_x != model.grid_size[0]-1 and not model.wall[loc_x+1][loc_y]:
            grid[loc_x][loc_y] = 0
            grid[loc_x+1][loc_y] = 1
            loc_x += 1
        elif act_choice == 1 and loc_x != 0 and not model.wall[loc_x-1][loc_y]:
            grid[loc_x][loc_y] = 0
            grid[loc_x-1][loc_y] = 1
            loc_x -= 1
        elif act_choice == 2 and loc_y != 0 and not model.wall[loc_x][loc_y-1]:
            grid[loc_x][loc_y] = 0
            grid[loc_x][loc_y-1] = 1
            loc_y -= 1
        elif act_choice == 3 and loc_y != model.grid_size[1]-1 and not model.wall[loc_x][loc_y+1]:
            grid[loc_x][loc_y] = 0
            grid[loc_x][loc_y+1] = 1
            loc_y += 1

        objective += model.compute_reward(grid, best_action)
        risk = model.calc_risk(grid, act_choice)
        #risk += risk_error_range/2 * np.random.random() if np.random.choice([True, False], p=[risk_error_probability, 1 - risk_error_probability]) else 0
        #risk = np.random.uniform(max(risk - risk_error_range/2, 0.), min(risk + risk_error_range/2, 1.))
        risk = (max(risk - risk_error_range/2, 0.)+ min(risk + risk_error_range/2, 1.))/2
        if np.random.choice([True, False], p=[risk, 1 - risk]):
            risk_bool = True
            break

        observ = grid_observ(x=loc_x, y=loc_y, sensor_accuracy=model.sensor_accuracy,
                             grid_size=model.grid_size, wall=model.wall, wall_set=model.wall_set)

        if not node_list[current_node_idx].terminal:
            observation_probability[current_node_idx][observ] += 1
        count += 1

    return risk_bool, objective


# state is the correct state
# observation depends on action
# for listen prob of 85% returns correct state
#@njit
def grid_observ(x, y, sensor_accuracy, grid_size, wall, wall_set):
    wall_count = 0
    if x == 0 or wall[x - 1][y]:
        wall_count += 1
    if x == grid_size[0]-1 or wall[x+1][y]:
        wall_count += 1
    if y == 0 or wall[x][y-1]:
        wall_count += 1
    if y == grid_size[1]-1 or wall[x][y+1]:
        wall_count += 1

    if np.random.choice([True, False], p=[sensor_accuracy, 1-sensor_accuracy]):
        return wall_count

    return np.random.choice(wall_set[wall_set != wall_count])






if __name__ == '__main__':
    horizon = 5
    grid_size = np.array([5, 6], dtype=np.int32)
    start = np.array([0, 0], dtype=np.int32)
    wall_list = np.array([[1, 4], [2, 2]], dtype=np.int32)
    risk_list = np.array([[2, 3, .9], [3, 2, .5], [0, 1, 0.05], [1, 0, 0.1], [1, 1, 0.1]])
    state_rewards = np.array([[1, 0, 1.0], [2, 2, 2.0], [3, 3, 5.0], [4, 5, 10.0]])


    horizon = 6
    grid_size = np.array([4, 4], dtype=np.int32)
    start = np.array([1, 0], dtype=np.int32)
    #wall_list = np.array([[1, 1], [2, 2]], dtype=np.int32)
    wall_list = None
    risk_list = np.array([[1, 2, 1.],[1, 3, 1.]])
    state_rewards = np.array([[0, 3, 10.], [1,1,2], [3, 3, 5.]])


    Grid_model = grid_model(wall_list=wall_list, risk_list=risk_list, grid_size=grid_size, start=start,
                            step_cost=0.0, state_rewards=state_rewards, sensor_accuracy=0.85,
                            time_horizon=horizon, step_horizon=horizon, soft_horizon=horizon, cc=0.25,
                            optimization='maximize')

    #IPython.embed(header='LP test Done')
    risk_error_range = 0.07  # risk range/2
    risk_error_probability = 0.5

    policy = CCPOMDP_robust(initial_belief=Grid_model.init_state, model=Grid_model, prune=True, risk_error_range=risk_error_range, risk_error_probability=risk_error_probability, robust=True)

    #policy_test(node_list=policy.BFS_tree, model=Grid_model, risk_error_range=0.1, risk_error_probability=0.2, count=10000)

    #p = CCPOMDP_FPTAS(Grid_model.init_state, Grid_model)

    IPython.embed(header='LP test Done')

