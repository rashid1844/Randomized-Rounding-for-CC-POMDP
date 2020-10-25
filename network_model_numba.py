import numpy as np
import IPython
import time
from numba import jitclass, types, typeof, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

"""in case risk_list, reward_list or wall_list are empty, it must be set to None"""
from ccpomdp_class_combNode_LP_fullRound_prune import CCPOMDP

spec = [
    ('action_list', typeof(['sample'])),
    ('trans_prob', types.float64[:, :, ::1]),
    ('s_horizon', types.int32),
    ('soft_horizon', types.int32),
    ('grid_size', types.int32[::1]),
    ('reward_func', types.float64[:, ::1]),
    ('risk_func', types.float64[::1]),
    ('cc', types.float64),
    ('reset_actions', types.boolean[::1]),
    ('optimization', types.unicode_type),
    ('init_state', types.float64[::1]),
    ('obs_prob', types.float64[:, ::1]),
    ]

@jitclass(spec)
class network_model:
    def __init__(self, step_horizon=5, cc=0.1, optimization='min'):
        #self.initial_belief = np.array([0.5, 0.5])  # np.array
        self.action_list = ['unrestrict', 'steady', 'restrict', 'reboot']  # up:increase value, down:decrease value
        self.s_horizon = step_horizon
        self.soft_horizon = step_horizon

        self.risk_func = np.array([0., 0., 0., 0., 0., 0., 1.], np.float64)
        self.cc = cc
        self.reset_actions = np.array([False, False, False, False])  # TODO reset_states
        self.optimization = optimization  # maximize or minimize
        self.init_state = np.array([1., 0., 0., 0., 0., 0., 0.], np.float64)
        self.obs_prob = np.array([[1, 1, 1, 0.9, 0.7, 0.5, 0], [0, 0, 0, 0.1, 0.3, 0.5, 1]], np.float64)

        self.reward_func = np.array([[-20,0,20,40,60,80,-20],[-20,0,20,40,60,80,-20],[-20,0,20,40,60,80,-20],[-40,-40,-40,-40,-40,-40,-40]], np.float64)
        self.reward_func = np.array([[100,80,60,40,20,0,100],[100,80,60,40,20,0,100],[100,80,60,40,20,0,100],[120,120,120,120,120,120,120]], np.float64)
        #self.reward_func = np.array([[0,0,20,40,60,80,0],[0,0,20,40,60,80,0],[0,0,20,40,60,80,0],[0,0,0,0,0,0,0]], np.float64)

        self.trans_prob = np.array([[[0.5,0.3,0.1,0.1,0,0,0], [0.2,0.3,0.3,0.1,0.1,0,0], [0.1,0.1,0.3,0.3,0.1,0.1,0], [0,0.1,0.1,0.3,0.3,0.1,0.1],[0,0,0.1,0.1,0.3,0.3,0.2],[0,0,0,0.1,0.1,0.3,0.5],[0,0,0,0,0,0,1.],],
                            [[0.7,0.2,0.1,0.,0.,0.,0.],[0.3,0.4,0.2,0.1,0.,0.,0.],[0.1,0.2,0.4,0.2,0.1,0.,0.],[0.,0.1,0.2,0.4,0.2,0.1,0.],[0.,0.,0.1,0.2,0.4,0.2,0.1],[0.,0.,0.,0.1,0.2,0.4,0.3],[0.,0.,0.,0.,0.,0.,1.],],
                            [[0.8,0.1,0.1,0.,0.,0.,0.],[0.5,0.3,0.1,0.1,0.,0.,0.],[0.2,0.3,0.3,0.1,0.1,0.,0.],[0.1,0.1,0.3,0.3,0.1,0.1,0.],[0.1,0.,0.1,0.3,0.3,0.1,0.1],[0.,0.1,0.,0.1,0.3,0.3,0.2],[0.,0.,0.,0.,0.,0.,1.],],
                            [[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],],], np.float64)

        #self.prior_timer = 0.
        self._jit_function_init()



    def update_params(self, step_horizon=5, cc=0.1):
        self.s_horizon = step_horizon
        self.cc = cc


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



    def observation_probability(self, prior_belief, observ):
        #if (prior_belief * observ).sum() == 0:
        #    print('prob 0:',prior_belief, observ)
        return (prior_belief * observ).sum()


    def compute_reward(self, prior_belief, action):
        return (prior_belief * self.reward_func[action]).sum()


    def action_feasibility(self, belief):
        return self.action_list

    def calc_risk(self, belief, action):
        return (belief * self.risk_func).sum()

    def calc_safe_prior(self, pre_safe_post, action):
        #s_time = time.time()
        safe_prior = self.trans_prob[action][0] * pre_safe_post[0] + self.trans_prob[action][1] * pre_safe_post[1] + self.trans_prob[action][2] * pre_safe_post[2] + \
            self.trans_prob[action][3] * pre_safe_post[3] + self.trans_prob[action][4] * pre_safe_post[4] + self.trans_prob[action][5] * pre_safe_post[5] #+ self.trans_prob[action][6] * pre_safe_post[6] # commented as it's safe prior
        return safe_prior/safe_prior.sum()


    def calc_safe_post(self, obs, safe_prior):
        safe_post = safe_prior * obs
        return safe_post / safe_post.sum()


if __name__ == "__main__":
    horizon = 7  # prune used for horizon 7

    m = network_model(step_horizon=horizon, cc=.3)

    # IPython.embed(header='LP model Done')

    policy = CCPOMDP(initial_belief=m.init_state, model=m, prune=True,
                     rounding_min_iteration=100,  prev_state_reward=True)
    # policy = CCPOMDP_FPTAS(initial_belief=Network_model.init_state, model=Network_model, )

    #policy_test(node_list=policy.BFS_tree, model=Network_model, count=10000)

    # p = CCPOMDP_FPTAS(Grid_model.init_state, Grid_model)

    IPython.embed(header='LP test Done')


