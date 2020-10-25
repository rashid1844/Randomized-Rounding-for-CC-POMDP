import numpy as np
import IPython
import time
from numba import jitclass, types, typeof, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

"""in case risk_list, reward_list or wall_list are empty, it must be set to None"""


spec = [
    ('action_list', typeof(['sample'])),
    ('trans_prob', types.float64[::1]),
    ('s_horizon', types.int32),
    ('soft_horizon', types.int32),
    ('grid_size', types.int32[::1]),
    ('reward_func', types.float64[:, ::1]),
    ('risk_func', types.float64[:, ::1]),
    ('cc', types.float64),
    ('reset_actions', types.boolean[::1]),
    ('optimization', types.unicode_type),
    ('init_state', types.float64[:, ::1]),
    ('wall', types.boolean[:, ::1]),
    ('wall_count', types.int32),
    ('sensor_accuracy', types.float64),
    ('obs_prob', types.float64[:, :, ::1]),
    ('wall_set', types.int32[::1])]


@jitclass(spec)
class grid_model:
    def __init__(self, wall_list, risk_list, state_rewards, grid_size, start,
                 step_cost=-0.04, sensor_accuracy=0.8, time_horizon=5, step_horizon=5, soft_horizon=3, cc=0.1, optimization='Max'):
        #self.initial_belief = np.array([0.5, 0.5])  # np.array
        self.action_list = ['up', 'down', 'left', 'right']  # up:increase value, down:decrease value
        self.trans_prob = np.array([0.8, 0.0, 0.1, 0.1])  # correct, opposite, left, right
        self.s_horizon = step_horizon
        self.soft_horizon = soft_horizon
        self.grid_size = grid_size
        self.reward_func = np.array([step_cost]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])  #np.full(self.grid_size, step_cost)
        if state_rewards is not None:
            for i in range(len(state_rewards)):
                self.reward_func[int(state_rewards[i][0])][int(state_rewards[i][1])] = state_rewards[i][2]

        self.risk_func = np.array([0.0]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])
        if risk_list is not None:
            for i in range(len(risk_list)):
                self.risk_func[int(risk_list[i][0])][int(risk_list[i][1])] = risk_list[i][2]
        self.cc = cc
        self.reset_actions = np.array([False, False, False, False])  # TODO reset_states
        self.optimization = optimization  # maximize or minimize
        self.init_state = np.array([0.0]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])
        self.init_state[start[0]][start[1]] = 1.0  # init state
        self.wall = np.array([False]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])
        self.wall_count = 0  # num of wall states
        if wall_list is not None:
            for i in range(len(wall_list)):
                self.wall[int(wall_list[i][0])][int(wall_list[i][1])] = True
                self.wall_count += 1
        self.sensor_accuracy = sensor_accuracy
        self.create_observation_function()
        #self.prior_timer = 0.
        self._jit_function_init()


        # calculates prior belief for a given action
        # and the previous belief: prior_b(s') = sum( trans_p(s'|s,a) * pre_b(s') ) 'for all s'
        # note: use action index, not action name


    def update_params(self, wall_list, risk_list, state_rewards, grid_size, start, step_cost=-0.04,
                      sensor_accuracy=0.8, time_horizon=5, step_horizon=5, soft_horizon=3, cc=0.1, optimization='Max'):
        self.s_horizon = step_horizon
        self.soft_horizon = soft_horizon
        self.grid_size = grid_size
        self.reward_func = np.array([step_cost]*self.grid_size[0]*self.grid_size[1], dtype=np.float64).reshape(self.grid_size[0], self.grid_size[1])
        if state_rewards is not None:
            for i in range(len(state_rewards)):
                self.reward_func[int(state_rewards[i][0])][int(state_rewards[i][1])] = state_rewards[i][2]

        self.risk_func = np.zeros((self.grid_size[0], self.grid_size[1]), np.float64)
        if risk_list is not None:
            for i in range(len(risk_list)):
                self.risk_func[int(risk_list[i][0])][int(risk_list[i][1])] = risk_list[i][2]
        self.cc = cc
        self.optimization = optimization  # maximize or minimize
        self.init_state = np.array([0.0]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])
        self.init_state[start[0]][start[1]] = 1.0  # init state
        self.wall = np.array([False]*self.grid_size[0]*self.grid_size[1]).reshape(self.grid_size[0], self.grid_size[1])
        self.wall_count = 0  # num of wall states
        if wall_list is not None:
            for i in range(len(wall_list)):
                self.wall[int(wall_list[i][0])][int(wall_list[i][1])] = True
                self.wall_count += 1
        self.sensor_accuracy = sensor_accuracy
        self.create_observation_function()






    def _jit_function_init(self):
        """run all actions for first time so they get compiled"""
        self.calc_safe_prior(self.init_state, 0)  # init function
        obss = self.obs_function(0)
        self.observation_probability(self.init_state, obss[0])
        self.compute_reward(self.init_state, 0)
        self.action_feasibility(self.init_state)
        self.calc_risk(self.init_state, 0)
        self.calc_safe_post(obss[0],self.init_state)
        print('model compiled')


    def obs_function(self, action_idx):
        return self.obs_prob

    def create_observation_function(self):
        wall_set = np.array([self.wall_counter(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])], dtype=np.int32)
        wall_set = np.unique(wall_set)
        self.wall_set = wall_set
        #obs_prob = np.array([], dtype=np.float64)
        remain_accuracy = (1-self.sensor_accuracy) / (len(wall_set)-1)

        for wall in list(wall_set):
            belief = np.zeros_like(self.init_state, dtype=np.float64)
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    if not self.wall[x][y]:
                        belief[x][y] = self.sensor_accuracy if self.wall_counter(x,y) == wall else remain_accuracy
            if wall_set[0] == wall:
                obs_prob = np.copy(belief).reshape(1, self.grid_size[0], self.grid_size[1])
            else:
                obs_prob = np.append(obs_prob, belief).reshape(-1, self.grid_size[0], self.grid_size[1])

        self.obs_prob = obs_prob

    def wall_counter(self, x, y):
        """for a given state(x,y) return number of walls around it (used to create the observation function)"""
        counter = 0
        if x == 0 or self.wall[x - 1][y]:
            counter += 1
        if x == self.init_state.shape[0]-1 or self.wall[x+1][y]:
            counter += 1
        if y == 0 or self.wall[x][y-1]:
            counter += 1
        if y == self.init_state.shape[1]-1 or self.wall[x][y+1]:
            counter += 1
        return counter


    def observation_probability(self, prior_belief, observ):
        return (prior_belief * observ).sum()


    def compute_reward(self, prior_belief, action):
        return (prior_belief * self.reward_func).sum()


    def action_feasibility(self, belief):
        return self.action_list

    def calc_risk(self, belief, action):
        return (belief * self.risk_func).sum()

    def calc_safe_prior(self, pre_safe_post, action):
        #s_time = time.time()
        safe_prior = np.zeros_like(pre_safe_post, dtype=np.float64)

        if action == 0:
            trans_prob = self.trans_prob
        elif action == 1:
            trans_prob = np.array([self.trans_prob[1], self.trans_prob[0], self.trans_prob[3], self.trans_prob[2]])
        elif action == 2:
            trans_prob = np.array([self.trans_prob[3], self.trans_prob[2], self.trans_prob[0], self.trans_prob[1]])
        elif action == 3:
            trans_prob = np.array([self.trans_prob[2], self.trans_prob[3], self.trans_prob[1], self.trans_prob[0]])

        for x in range(pre_safe_post.shape[0]):
            for y in range(pre_safe_post.shape[1]):
                # safe states
                if not self.wall[x][y]:  # TODO change it to all all states with risk less than threshold
                    if x == 0 or self.wall[x-1][y]:  # state bellow
                        safe_prior[x][y] += trans_prob[1] * pre_safe_post[x][y] * (1 - self.risk_func[x][y])
                    else:
                        safe_prior[x][y] += trans_prob[0] * pre_safe_post[x-1][y] * (1 - self.risk_func[x-1][y])

                    if x == pre_safe_post.shape[0]-1 or self.wall[x+1][y]:  # state above
                        safe_prior[x][y] += trans_prob[0] * pre_safe_post[x][y] * (1 - self.risk_func[x][y])
                    else:
                        safe_prior[x][y] += trans_prob[1] * pre_safe_post[x + 1][y] * (1 - self.risk_func[x+1][y])

                    if y == pre_safe_post.shape[1]-1 or self.wall[x][y+1]:   # state right
                        safe_prior[x][y] += trans_prob[3] * pre_safe_post[x][y] * (1 - self.risk_func[x][y])
                    else:
                        safe_prior[x][y] += trans_prob[2] * pre_safe_post[x][y+1] * (1 - self.risk_func[x][y+1])

                    if y == 0 or self.wall[x][y-1]:  # state left
                        safe_prior[x][y] += trans_prob[2] * pre_safe_post[x][y] * (1 - self.risk_func[x][y])
                    else:
                        safe_prior[x][y] += trans_prob[3] * pre_safe_post[x][y-1] * (1 - self.risk_func[x][y-1])

        safe_prior = safe_prior / (1 - self.calc_risk(pre_safe_post, action))
        if not 0.9999 <= safe_prior.sum() <= 1.0001:
            print('safe_prior_function error')
            #IPython.embed(header='safe_prior_function')
            #raise TypeError('safe prior prob doesnt sum to one')
        #self.prior_timer += time.time() - s_time
        return safe_prior


    def calc_safe_post(self, obs, safe_prior):
        safe_post = safe_prior * obs
        return safe_post / safe_post.sum()


if __name__ == "__main__":
    IPython.embed()

