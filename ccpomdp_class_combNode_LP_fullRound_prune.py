import numpy as np
from gurobipy import *
from grid_model import grid_model
import itertools
import random
import IPython
import time
import copy
from numba import njit
# node represents the root or any posterior
# must define parent with contractor
# name is combination of action and observation index starting at root 'r'

class Node(object):
    def __init__(self, parent=None, parent_idx=0, action_idx=None, safe_post_probability=[],
                 reward=0.0, reset=False, safe_prior=None, safe_post=[], BFS_index=0):
        self.parent = parent
        self.parent_idx = parent_idx  # observation of parent (index of observation in parent list)
        self.name = 'r' if not self.parent else self.parent.name + str(parent_idx) + str(action_idx)
        self.action_idx = action_idx  # index of action #TODO: replace with action name or
        self.child_list = []  # list of lists
        self.safe_post_probability = safe_post_probability  # list of obs probability  based on (prior and obs)
        self.total_safe_probability = 1. if not self.parent else self.parent.total_safe_probability * self.parent.safe_post_probability[parent_idx]
        self.reward = reward  # expected for the whole action
        self.risk = 0.0  # expected for the whole action
        self.soft_terminal = False
        self.terminal = False
        self.depth = 0 if not self.parent else 1 + self.parent.depth
        self.x_val = 0
        self.best_action = -1  # list of best action idx per obs (which child has x value of one)
        self.reset = reset  # if true means belief was reset at this state
        self.safe_prior = safe_prior
        self.safe_post = safe_post  # list
        self.parent_risk_prob_safe = 1.0  # probability of safely reaching the current node (without obs)
        # Total_safe_probability: is probability of reaching this node, assuming all nodes were safe so far
        # parent_risk_prob_safe: probability of reaching this node * probability of being safe till this node (1-risk)

        self.BFS_index = BFS_index


    def __eq__(self, other):
        return self.name == other.name

    def normalize(self):
        #norm = sum(self.safe_post_probability)
        self.safe_post_probability = np.array(self.safe_post_probability) / sum(self.safe_post_probability)
        #self.total_probability /= norm
        #self.reward /= norm


    def node_info(self):
        print('name', self.name)
        if self.parent:
            print('parent', self.parent.name)
        for child in self.child_list:
            print('child', child)
        #print('belief', self.belief)
        print('reward', self.reward)
        print('safe_prior', self.safe_prior)
        print('safe_post', self.safe_post)
        print('risk', self.risk)

    def find_best_action(self):
        """Finds the best action by looking for the child with x_val==1"""
        if not self.terminal:
            self.best_action = []
            if self.child_list:
                for obs_list in self.child_list:
                    for child in obs_list:
                        if child.x_val == 1:
                            self.best_action.append(child.action_idx)
                            break


"""
full prune at the end:
1- create list of Min_risk
2- create list of Min_risk_list
3- calc min_risk function using idx lists(after converting to np) (njit)
4- prune function (njit) based on last depth nodes (and parents if prunes)
5- create a list of prune idx and remove them from all lists backward

"""
@njit
def prune_all_nodes(node_count, terminal_node_count, parent_idx_list, child_list_idx_list, parent_act_idx_list, cc, Min_risk_idx_list, Min_risk_list_idx_list):
    """prunes all nodes of expanded tree, by checking last depth nodes, and if prune check parents, and add all nodes to prune list"""

    prune_list = np.full(node_count, False, np.bool_)
    terminal_start = node_count - terminal_node_count
    for idx in range(node_count-1, terminal_start-1, -1):
        if not prune_list[idx] and prune(idx, cc, parent_idx_list, Min_risk_idx_list, Min_risk_list_idx_list, parent_act_idx_list, prune_list, child_list_idx_list, True):
            prune_list[idx] = True
            parent_idx = idx
            while prune(parent_idx_list[parent_idx], cc, parent_idx_list, Min_risk_idx_list, Min_risk_list_idx_list, parent_act_idx_list, prune_list, child_list_idx_list, False):
                prune_list[parent_idx_list[parent_idx]] = True
                if prune_list[0]:
                    break
                parent_idx = parent_idx_list[parent_idx]
            #prune_list[parent_idx] = True

            if prune_list[0]:
                break  # root is pruned so infeasibile

            if parent_idx >= terminal_start:
                continue
            temp_prune_list = [parent_idx]

            while len(temp_prune_list) > 0:
                temp_idx = temp_prune_list.pop(0)
                child_list_list = [0 for _ in range(0)]
                for obs_list in range(len(child_list_idx_list[temp_idx])):
                    if np.sum(child_list_idx_list[temp_idx][obs_list])*-1 != len(child_list_idx_list[temp_idx][obs_list]):
                        child_list_list += list(filter(lambda a: a != -1, list(child_list_idx_list[temp_idx][obs_list])))
                temp_prune_list += [v for v in child_list_list if v < terminal_start and not prune_list[v]]
                prune_list[np.array(child_list_list, np.int32)] = True

    return prune_list



@njit
def prune(idx, cc, parent_idx_list, Min_risk_idx_list, Min_risk_list_idx_list, parent_act_idx_list, prune_list, child_list_idx_list, terminal_bool):
    """test if a given node should be pruned based on Min_risk or if child_obs has no actions"""
    if not terminal_bool:
        for obs in range(len(child_list_idx_list[idx])):
            if np.sum(child_list_idx_list[idx][obs]) * -1 != len(child_list_idx_list[idx][obs]):
                if 0 == np.sum(np.array([1 for act_idx in child_list_idx_list[idx][obs] if not prune_list[act_idx] and act_idx != -1], np.int32)):
                    return True


    prev_parent = int(idx)
    parent = int(parent_idx_list[idx])
    risk_sum = np.float64(Min_risk_idx_list[idx])
    while prev_parent != 0:
        risk_sum += Min_risk_idx_list[parent] - Min_risk_list_idx_list[parent][parent_act_idx_list[prev_parent]]  # parent_min_risk - min(min_risk of child obs)
        if parent == 0:
            break
        prev_parent = int(parent_idx_list[prev_parent])
        parent = int(parent_idx_list[parent])

    return risk_sum > cc

@njit
def calc_min_risk(node_count, terminal_node_count,risk_list, child_list_idx_list, obs_count):
    """Min_risk is the risk of current node and sum of minimum MinRisk of each observation"""
    Min_risk_idx_list = np.zeros(node_count, np.float64)
    #Min_risk_list_idx_list = np.array([[0.]*obs_count]*(node_count - terminal_node_count), np.float64)
    Min_risk_list_idx_list = np.zeros((len(child_list_idx_list), obs_count), np.float64)

    for idx in range(node_count-1, -1, -1):
        Min_risk = np.float64(risk_list[idx])
        if idx < len(child_list_idx_list):
            for obs_idx in range(len(child_list_idx_list[idx])):
                if len(child_list_idx_list[idx][obs_idx]) * (-1) != np.sum(child_list_idx_list[idx][obs_idx]):
                    Min_risk_list_idx_list[idx][obs_idx] = np.min(np.array([Min_risk_idx_list[child_idx] for child_idx in child_list_idx_list[idx][obs_idx] if child_idx != -1], np.float64))
            Min_risk += np.sum(Min_risk_list_idx_list[idx])
        Min_risk_idx_list[idx] = np.float64(Min_risk)

    return Min_risk_idx_list, Min_risk_list_idx_list


#########################################################################################################################################################################################
#########################################################################################################################################################################################


class CCPOMDP:
    def __init__(self, initial_belief, model=None, prune=False, rounding_min_iteration=0, greedy_iteration=0, greedy_advanced=False):  # action_list, trans_prob, obs_prob, t_horizon, s_horizon, reward_func, risk_func,duration_func, cc, reset_actions):
        self.initial_belief = initial_belief  # np.array
        self.model = model

        self.rounding_min_iteration = rounding_min_iteration
        self.greedy_iteration = greedy_iteration
        self.greedy_advanced = greedy_advanced

        self.tree_expand_time = 0
        self.LP_time = 0
        self.ILP_time = 0
        self.ILP_diff_count = 0
        self.obj_LP = 0
        self.obj_ILP = 0
        self.round_iteration = 0
        self.prune = prune
        self.model_feasibility = True
        if model is None:
            raise TypeError('Model not defined')
        self.open_nodes = []
        #self.node_names = {}
        self.BFS_tree = []

        self.integral_sol = False
        self.feasible = False
        #self.node_depth_list = [[]] * (self.model.s_horizon)  # TODO: remove
        #str_depth_list = 'self.node_depth_list = [' + '[],' * (self.model.s_horizon-1) + '[]]'
        #exec(str_depth_list)  # to create an empty list of lists
        self.depth_idx_list = np.array([-1] * (self.model.s_horizon+2), np.int32)  # returns the min BFS idx for each depth, so for depth_i (list[i] to list[i+1])

        ### roundized rounding lists
        self.parent_idx_list = []  # list[node_idx] = node_parent_idx
        self.child_list_idx_list = []  # list[node_idx] = child_list_idx (2d [obs])
        # TODO: make above list work with different child sizes by setting defualt to -1
        self.risk_list = []  # list[node.idx] = risk
        self.reward_list = []  # list[node.idx] = reward
        self.parent_act_idx_list = []  # list[node.idx] = node.parent_idx (parent observation
        self.prune_idx_list = np.arange(0)

        #self.test_case_tree()
        self.test = False  # test prune
        if self.test:
            self.test_prune()
        else:
            self.expand_tree()  # the main code
            if self.model_feasibility:
                self.gurobi_solver_LP()
                if self.model_feasibility:
                    #print('LP policy check', self.full_policy_check())
                    assert self.full_policy_check()
                self.gurobi_solver_ILP()
                if self.model_feasibility:
                    #print('ILP policy check', self.full_policy_check())
                    assert self.full_policy_check()

    def test_prune(self):
        """Tests the prune by comparing the objective before and after pruning"""
        self.prune = False
        self.open_nodes = []
        #self.node_names = {}
        self.BFS_tree = []
        self.expand_tree()
        self.gurobi_solver_ILP()
        no_prune_obj = self.obj_ILP

        self.depth_idx_list = np.array([-1] * (self.model.s_horizon+2), np.int32)  # returns the min BFS idx for each depth, so for depth_i (list[i] to list[i+1])
        self.parent_idx_list = []  # list[node_idx] = node_parent_idx
        self.child_list_idx_list = []  # list[node_idx] = child_list_idx (2d [obs])
        self.risk_list = []  # list[node.idx] = risk
        self.reward_list = []  # list[node.idx] = reward
        self.parent_act_idx_list = []  # list[node.idx] = node.parent_idx (parent observation

        self.prune = True
        self.open_nodes = []
        #self.node_names = {}
        self.BFS_tree = []
        self.expand_tree()
        self.gurobi_solver_ILP()
        prune_obj = self.obj_ILP
        print('prune obj, no_prune:{}, prune:{}'.format(no_prune_obj, prune_obj))
        if np.round(no_prune_obj, decimals=5) != np.round(prune_obj, decimals=5):
            print('prune error, no_prune:{}, prune:{}'.format(no_prune_obj, prune_obj))


    def node_idx_f(self, idx):
        """function is used to get correct index after pruning, as it's slower to do it for child_list directly"""
        if self.prune:
            return self.prune_idx_list[idx]
        return idx

    def del_prune_list(self, prune_list, node_count, terminal_node_count):
        """
        delete nodes in prune_list from all lists
        delete function deletes based on index not value, but as we are deleting index values it works"""
        prune_list = np.where(prune_list == True)[0]  # convert to list of indices

        temp_prune_idx_list = np.arange(len(self.BFS_tree))
        temp_prune_idx_list = np.delete(temp_prune_idx_list, prune_list)
        self.prune_idx_list = np.arange(len(self.BFS_tree))
        for i, idx in enumerate(temp_prune_idx_list):  # maps old index with new index
            self.prune_idx_list[idx] = i

        # del child_list
        #self.child_list_idx_list = self.child_list_idx_list.tolist()
        for idx in prune_list:
            #if self.parent_idx_list[idx] in prune_list:
            #    continue
            temp_child_list = self.child_list_idx_list[self.parent_idx_list[idx]][self.parent_act_idx_list[idx]]
            temp_child_list[np.where(temp_child_list == idx)[0]] = -1
            self.child_list_idx_list[self.parent_idx_list[idx]][self.parent_act_idx_list[idx]] = temp_child_list

        self.child_list_idx_list = np.delete(self.child_list_idx_list, prune_list, axis=0)

        self.parent_idx_list = np.delete(self.parent_idx_list, prune_list)
        for i in range(len(self.parent_idx_list)):
            self.parent_idx_list[i] = self.prune_idx_list[self.parent_idx_list[i]]

        self.risk_list = np.delete(self.risk_list, prune_list)
        self.reward_list = np.delete(self.reward_list, prune_list)
        self.parent_act_idx_list = np.delete(self.parent_act_idx_list, prune_list)
        self.BFS_tree = np.asarray(self.BFS_tree, object)
        self.BFS_tree = np.delete(self.BFS_tree, prune_list)

        for depth in range(1, len(self.depth_idx_list)-2):
            self.depth_idx_list[depth+1:] -= np.sum((self.depth_idx_list[depth] < prune_list) & (prune_list < self.depth_idx_list[depth+1]))
        self.depth_idx_list[-1] = len(self.BFS_tree)

        for idx, node in enumerate(self.BFS_tree):
            node.BFS_index = idx

        # check child_list
        #for idx in range(len(self.child_list_idx_list)):
        #    for obs in self.child_list_idx_list[idx]:
        #        if len(obs) == 0:  # not action
        #            print(f'error node:{idx} obs:{obs} has no actions')


    # FIFO: in terms of node expansion (similar to depth first search)
    # note: use action index, not action name

    # Steps: expands initial node, adds it to open_node list then goes into the loop
    # the loop: takes last item in open_nodes, explores all actions, for each action explores all observations
    # adds new node if they are not terminal to open_nodes. final pops current node from open_nodes list

    def expand_tree(self):
        s_time = time.time()
        b0 = Node(safe_post=[self.initial_belief],  safe_post_probability=[1.], BFS_index=0)
        #self.node_names[b0.name] = b0
        self.open_nodes.append(b0)
        self.BFS_tree.append(b0)

        self.parent_idx_list.append(-1)
        self.risk_list.append(b0.risk)
        self.reward_list.append(b0.reward)
        self.parent_act_idx_list.append(-1)
        self.depth_idx_list[0] = 0  # for root
        max_obs = len(self.model.obs_function(0))
        max_act = len(self.model.action_feasibility(self.initial_belief))
        print('max_obs', max_obs)
        #b0.parent_risk_prob_safe = 1.0
        BFS_index = 1
        while self.open_nodes:
            node = self.open_nodes.pop(0)
            # obs1 for action in new_node for each obs in node,
            node_child_list_idx = []
            # node observations
            for obs_idx in range(len(node.safe_post)):
                obs_child_list = []
                obs_child_list_idx = []
                # observation actions
                for node_action_idx, node_action in enumerate(self.model.action_feasibility(node.safe_post[obs_idx])):  # TODO: switch from action_idx to act name
                    max_act = max(max_act, node_action_idx+1)
                    safe_prior = self.model.calc_safe_prior(node.safe_post[obs_idx], node_action_idx)
                    new_node = Node(parent=node, parent_idx=obs_idx, action_idx=node_action_idx, safe_prior=safe_prior, safe_post_probability=[], safe_post=[], BFS_index=BFS_index)
                    new_node.parent_risk_prob_safe = node.parent_risk_prob_safe * (1 - self.model.calc_risk(node.safe_post[obs_idx], new_node.action_idx)) * node.safe_post_probability[obs_idx]
                    new_node.terminal = new_node.depth >= self.model.s_horizon  # or new_node.total_duration >= self.model.t_horizon
                    new_node.soft_terminal = True if new_node.depth >= self.model.soft_horizon else False
                    new_node.risk = self.model.calc_risk(safe_prior, new_node.action_idx) * new_node.parent_risk_prob_safe
                    new_node.reward = self.model.compute_reward(safe_prior, node_action_idx) * new_node.parent_risk_prob_safe  # TODO: pick reward based on prev_safe post or safe_prior
                    #new_node.reward = self.model.compute_reward(node.safe_post[obs_idx], node_action_idx) * new_node.parent_risk_prob_safe
                    if self.depth_idx_list[new_node.depth] == -1:  # has not been set yet, so this node is first node in this depth
                        self.depth_idx_list[new_node.depth] = new_node.BFS_index
                    # new_node observations
                    if not new_node.terminal or not new_node.soft_terminal:
                        for obs_index1, obs in enumerate(self.model.obs_function(node_action_idx)):    # observation action observations
                            safe_post = self.model.calc_safe_post(obs, safe_prior)
                            safe_probability = self.model.observation_probability(safe_prior, obs)
                            if safe_probability == 0.:
                                continue  # TODO: double check

                            #new_reward = self.model.compute_reward(safe_prior, node_action_idx) * new_node.parent_risk_prob_safe * safe_probability
                            #Update
                            new_node.safe_post_probability.append(safe_probability)
                            #new_node.reward += new_reward
                            new_node.safe_post.append(safe_post)
                        max_obs = max(max_obs, obs_index1+1)
                        new_node.normalize()
                        #post_risk = sum(self.model.calc_risk(new_node.safe_post[i], new_node.action_idx) * new_node.safe_post_probability[i] for i in range(len(new_node.safe_post))) * new_node.parent_risk_prob_safe
                        #if abs(post_risk-new_node.risk) > 0.0001:
                        #    print('risk error', abs(post_risk-new_node.risk))


                    #if self.prune:
                    #    new_node.calc_min_risk()
                    #self.node_names[new_node.name] = new_node
                    if self.model.reset_actions[node_action_idx]:
                        new_node.safe_post = b0.safe_post
                        new_node.reset = True

                    self.BFS_tree.append(new_node)
                    self.parent_idx_list.append(new_node.parent.BFS_index)

                    self.risk_list.append(new_node.risk)
                    self.reward_list.append(new_node.reward)
                    #self.node_depth_list[new_node.depth-1].append(new_node)
                    self.parent_act_idx_list.append(new_node.parent_idx)
                    BFS_index += 1

                    if not new_node.terminal:
                        self.open_nodes.append(new_node)
                    obs_child_list.append(new_node)
                    obs_child_list_idx.append(new_node.BFS_index)
                node.child_list.append(obs_child_list)
                node_child_list_idx.append(obs_child_list_idx)

            # TO deal with ununiform problems
            for act_list in node_child_list_idx:
                while len(act_list) != max_act:
                    act_list.append(-1)
            while len(node_child_list_idx) != max_obs:
                node_child_list_idx.append([-1]*max_act)


            self.child_list_idx_list.append(node_child_list_idx)
            # prune function
            #if self.prune:
            #    self.prune_node(node)

        print('tree expanded')
        print('number of nodes', len(self.BFS_tree))
        print('tree expantion time', time.time() - s_time)
        self.node_count = len(self.BFS_tree)
        self.tree_expand_time = time.time() - s_time
        # print(self.risk_list)
        s_time = time.time()

        self.depth_idx_list[self.model.s_horizon+1] = len(self.BFS_tree)  # end of last depth

        #if self.prune:
        #    self.full_prune()
        #    print('number of nodes', len(self.BFS_tree))
        #    print('prune time', time.time() - s_time)

        ########################################################################################
        self.reward_list = np.asarray(self.reward_list, np.float64)
        self.risk_list = np.asarray(self.risk_list, np.float64)
        self.parent_idx_list = np.asarray(self.parent_idx_list, np.int32)
        #self.child_list_idx_list[0] = [list(range(1, 1+len(self.model.action_feasibility(b0.safe_post[0]))))] + [[-1] * len(self.model.action_feasibility(b0.safe_post[0]))] * (len(self.model.obs_function(0)) - 1)
        self.child_list_idx_list = np.asarray(self.child_list_idx_list, np.int32)
        self.parent_act_idx_list = np.asarray(self.parent_act_idx_list, np.int32)
        ########################################################################################
        if self.prune:
            terminal_node_count = len(self.BFS_tree) - self.depth_idx_list[-2]
            r_time = time.time()
            Min_risk_idx_list, Min_risk_list_idx_list = calc_min_risk(len(self.BFS_tree), terminal_node_count, self.risk_list, self.child_list_idx_list, max_obs)
            print('calc_min_risk time', time.time()-r_time)
            r_time = time.time()

            print('depth list', self.depth_idx_list)

            prune_list = prune_all_nodes(len(self.BFS_tree), terminal_node_count, self.parent_idx_list, self.child_list_idx_list,
                                         self.parent_act_idx_list, self.model.cc, Min_risk_idx_list, Min_risk_list_idx_list)
            print('prune_all_nodes time', time.time()-r_time)
            r_time = time.time()
            if prune_list[0]:
                print('tree fully pruned')
                self.model_feasibility = False
            else:
                print('prune list size', np.sum(prune_list==True))
                self.del_prune_list(prune_list, len(self.BFS_tree), terminal_node_count)
                print('del_prune_list time', time.time()-r_time)

                print('new number of nodes', len(self.BFS_tree))
                print('prune time', time.time() - s_time)
                #raise RuntimeError('prune done')
        self.prune_node_count = len(self.BFS_tree)
        self.prune_time = time.time() - s_time
        #self.child_list_idx_list = np.asarray(self.child_list_idx_list, object)
        #self.child_list_idx_list[0] = [list(range(self.depth_idx_list[1], self.depth_idx_list[2]))]

        #print('node name', 'node risk', 'node reward')
        #for n in self.BFS_tree:
        #print(n.name, n.risk, n.reward)

        #for i, node in enumerate(self.node_names.items()):
        #    if node[1].reward is float('nan') or node[1].risk is float('nan'):
        #        IPython.embed(header='nan node')
        #    if i==16:
        #        IPython.embed(header='nan node')


    #def calc_risk_safe_prob(self, node):
    #    prod = 1.0
    #    while node.parent:
    #        prod *= (1 - self.model.calc_risk(node.parent.safe_post[node.parent_idx], node.parent.action_idx)) * node.safe_probability[node.parent_idx]
    #        node = node.parent
    #    return prod

    def gurobi_solver_LP(self):
        try:
            s_time = time.time()
            m = Model('pomdp_LP')
            # define x value
            gurobi_var = []

            gurobi_var = m.addVars((len(self.BFS_tree)), lb=0., ub=1., obj=1.0, vtype=GRB.SEMICONT)

            #for i, node in enumerate(self.BFS_tree):
                #gurobi_var.append(m.addVar(vtype=GRB.CONTINUOUS, name=node.name))
            #    gurobi_var.append(m.addVar(0.0, 1.0, 1.0, vtype=GRB.SEMICONT, name=node.name))  # semi continous with range [0.-1.] and defaut 1. (low, upper, objtive)
            #    #if node.x_val != -1:  #use saved values
            #    #    m.addConstr(gurobi_var[node.name] == node.x_val)
                #if node.risk > self.model.cc:
                #    m.addConstr(gurobi_var[i] == 0)

            print('time1', time.time()-s_time)

            # Objective
            if 'max' in self.model.optimization.lower():
                m.setObjective(quicksum(gurobi_var[i] * node.reward for i, node in enumerate(self.BFS_tree)), GRB.MAXIMIZE)
            else:
                m.setObjective(quicksum(gurobi_var[i] * node.reward for i, node in enumerate(self.BFS_tree)), GRB.MINIMIZE)

            print('time2', time.time()-s_time)

            # risk constraint

            m.addConstr(quicksum(gurobi_var[i] * node.risk for i, node in enumerate(self.BFS_tree)) <= self.model.cc)
            #m.addConstr((self.risk_list * gurobi_var).sum() <= self.model.cc)
            # force select root node
            m.addConstr(gurobi_var[0] == 1.0)


            print('time3', time.time()-s_time)

            # ensures that one child is selected if parent is selected
            #for index in range(len(self.BFS_tree)):
            #    if not self.BFS_tree[index].terminal:
            #        m.addConstrs((quicksum(gurobi_var[child.BFS_index] for child in child_list) == gurobi_var[index]) for child_list in self.BFS_tree[index].child_list)
            # sum of observation children per observation = value of action (observation parent)
            m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in self.child_list_idx_list[0][0] if child_idx != -1) == gurobi_var[0])
            for index in range(1, len(self.BFS_tree)):
                if not self.BFS_tree[index].terminal:
                    if not self.BFS_tree[index].soft_terminal:
                        for child_list in self.child_list_idx_list[index]:
                            if np.sum(child_list)*-1 != len(child_list):
                                m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in child_list if child_idx != -1) == gurobi_var[index])
                                #m.addConstr(quicksum(gurobi_var[child.BFS_index] for child in child_list) <= 1)
                    else:
                        for child_list in self.child_list_idx_list[index]:
                            m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in child_list if child_idx != -1) <= gurobi_var[index])
                            #m.addConstr(sum(gurobi_var[child.name] for child in child_list) <= 1)
                            print('wrong constraint')

            print('time4', time.time()-s_time)

            # Robust version
            # for each comb add comb upper risk + remaining depth avg risk
            #gamma_percent = 0.3
            #gamma = int(math.ceil(gamma_percent * self.model.s_horizon))
            #comb_list = list(itertools.combinations(range(self.model.s_horizon), gamma))
            #for comb in comb_list:  # TODO: add risk_down, risk_up in each node for robust
            #    m.addConstr(sum(gurobi_var[node.name].x * node.risk_up for d in comb for node in self.node_depth_list[d]) <= self.model.cc)


            m.optimize()

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))


        if m.status == 3:  # model is infeasibile
            self.model_feasibility = False
            return None

        print('LP_time', m.Runtime)
        #IPython.embed(header='lp')
        #gurobi_var_list = m.getVars()
        gurobi_var_list = np.array(m.getAttr('x'), dtype=np.float64)

        self.integral_sol = True
        self.feasible = False

        for v in gurobi_var_list:
            if abs(v-0) > 10**-8 and abs(v-1) > 10**-8:
                self.integral_sol = False
                break
        if self.integral_sol:
            self.obj_LP = m.objVal
            max_obj_var_list = gurobi_var_list
            self.feasible = True
            print('lp integral solution')

        else:
            obj_val = m.objVal
            max_obj = float('-inf') if 'max' in self.model.optimization.lower() else float('inf')  # max objective per iteration
            max_obj_var_list = None  # list of obj per iteration
            iteration = 0  # iteration counter
            depth_exp_risk = np.zeros(self.model.s_horizon+1, np.float64)  # expected risk per depth
            for i in range(self.model.s_horizon+1):
                #depth_exp_risk[i] = sum(node.risk * gurobi_var_list[node.BFS_index] for node in self.node_depth_list[i])
                try:
                    depth_exp_risk[i] = sum(self.risk_list[idx] * gurobi_var_list[idx] for idx in range(self.depth_idx_list[i], self.depth_idx_list[i+1]))
                except:
                    print(self.depth_idx_list)
                    raise RuntimeError('depth_idx_list error')

            #self.reward_list = np.asarray(self.reward_list, np.float64)
            #self.risk_list = np.asarray(self.risk_list, np.float64)
            #self.parent_idx_list = np.asarray(self.parent_idx_list, np.int32)
            #self.child_list_idx_list[0] = [[1, 2, 3, 4]] + [[-1, -1, -1, -1]] * (len(self.model.obs_function(0)) - 1)  # TODO: fix root has one obs so we assume obs of 3
            #self.child_list_idx_list = np.asarray(self.child_list_idx_list, np.int32)
            #self.parent_act_idx_list = np.asarray(self.parent_act_idx_list, np.int32)
            #opt_factor = len(self.model.obs_function(0))**(self.model.s_horizon-1)
            opt_factor = 2**(self.model.s_horizon-1)

            #for _ in range(k):
            while (not self.feasible) or (iteration < self.rounding_min_iteration) or (max_obj < obj_val/opt_factor and max_obj > 0 and self.prune):  #TODO: optimallity condition
                iteration += 1
                #print('k val', iteration)
                gurobi_list_copy = np.copy(gurobi_var_list)

                if iteration <= self.greedy_iteration:
                    #gurobi_list_copy, risk_sum, objective = self.randomized_rounding_greedy(gurobi_list_copy, depth_exp_risk, greedy_upper_exp_bool)
                    gurobi_list_copy, risk_sum, objective = randomized_rounding_greedy(gurobi_list_copy, self.model.cc,
                            self.risk_list, self.reward_list, self.parent_idx_list,
                            self.child_list_idx_list, self.parent_act_idx_list, self.model.s_horizon,
                            depth_exp_risk, self.greedy_advanced, self.depth_idx_list, self.prune, self.prune_idx_list)
                else:
                    #gurobi_list_copy, risk_sum, objective = self.randomized_rounding(gurobi_list_copy)
                    gurobi_list_copy, risk_sum, objective = randomized_rounding(gurobi_list_copy, self.model.cc,
                                                        self.risk_list, self.reward_list, self.parent_idx_list,
                                                                    self.child_list_idx_list, self.parent_act_idx_list, self.prune, self.prune_idx_list)

                if risk_sum <= self.model.cc:
                    self.feasible = True
                    if 'max' in self.model.optimization.lower() and objective >= max_obj or 'min' in self.model.optimization.lower() and objective <= max_obj:
                        max_obj = objective
                        max_obj_var_list = gurobi_list_copy


            print('k val', iteration)
            self.round_iteration = iteration
            self.obj_LP = max_obj

        if self.feasible:
            for i, val in enumerate(max_obj_var_list):
                self.BFS_tree[i].x_val = int(np.round(val))

        for node in self.BFS_tree:
            node.find_best_action()

        self.LP_time = time.time() - s_time
        print('LP Done', self.LP_time)
        print('LP obj', self.obj_LP)
        print('LP feasibility', self.feasible)
        self.LP_var_list = max_obj_var_list
        self.LP_var_list_org = gurobi_var_list



    def gurobi_solver_ILP(self):
        try:
            s_time = time.time()
            m = Model('pomdp_ILP')

            # define x value
            #gurobi_var = {}
            #for key in self.node_names.keys():
            #    gurobi_var[key] = m.addVar(vtype=GRB.BINARY, name=key)

            gurobi_var = m.addVars((len(self.BFS_tree)), vtype=GRB.BINARY)

            #for node in self.BFS_tree:
            #    gurobi_var[node.name] = m.addVar(vtype=GRB.BINARY, name=node.name)
                #if node.x_val != -1:  #use saved values
                #    m.addConstr(gurobi_var[node.name] == node.x_val)
            #    if node.risk > self.model.cc:
            #        m.addConstr(gurobi_var[node.name] == 0)

            # Objective
            if 'max' in self.model.optimization.lower():
                m.setObjective(quicksum(gurobi_var[i] * node.reward for i, node in enumerate(self.BFS_tree)), GRB.MAXIMIZE)
            else:
                m.setObjective(quicksum(gurobi_var[i] * node.reward for i, node in enumerate(self.BFS_tree)), GRB.MINIMIZE)

            # risk constraint
            m.addConstr(quicksum(gurobi_var[i] * node.risk for i, node in enumerate(self.BFS_tree)) <= self.model.cc)

            # force select root node
            m.addConstr(gurobi_var[0] == 1.0)

            # ensures that one child is selected if parent is selected
            m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in self.child_list_idx_list[0][0] if child_idx != -1) == gurobi_var[0])
            for index in range(1,len(self.BFS_tree)):
                if not self.BFS_tree[index].terminal:
                    if not self.BFS_tree[index].soft_terminal:
                        for child_list in self.child_list_idx_list[index]:
                            if np.sum(child_list) * -1 != len(child_list):
                                m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in child_list if child_idx != -1) == gurobi_var[index])
                                # m.addConstr(quicksum(gurobi_var[child.BFS_index] for child in child_list) <= 1)
                    else:
                        for child_list in self.child_list_idx_list[index]:
                            m.addConstr(quicksum(gurobi_var[self.node_idx_f(child_idx)] for child_idx in child_list if child_idx != -1) <= gurobi_var[index])
                            # m.addConstr(sum(gurobi_var[child.name] for child in child_list) <= 1)
                            print('wrong constraint')

            m.optimize()

            if m.status == 3:  # model is infeasibile
                self.model_feasibility = False
                return None

            print('ILP_time', m.Runtime)

            #setting output
            for i, val in enumerate(m.getAttr('x')):
                #if self.BFS_tree[i].x_val != int(np.round(val)):
                #    print('LP ')
                self.BFS_tree[i].x_val = int(np.round(val))

            self.ILP_var_list = np.array(m.getAttr('x'), np.float64)

            #sets best action
            for node in self.BFS_tree:
                node.find_best_action()
            #    # TODO: cane be automated to be aquired as request by node

            self.ILP_time = time.time() - s_time
            print("ILP Done, Time:{}".format(self.ILP_time))
            self.obj_ILP = m.objVal
            print(f'LP_obj:{self.obj_LP}, ILP_obj:{self.obj_ILP}')
            print('obj ratio', 1. if self.obj_LP == self.obj_ILP or self.obj_ILP == 0. else self.obj_LP/self.obj_ILP)

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))


    def full_policy_check(self):
        # to check if the policy is complete
        open_node = []
        for idx in self.child_list_idx_list[0][0]:
            if idx != -1 and self.BFS_tree[self.node_idx_f(idx)].x_val == 1:
                open_node.append(self.node_idx_f(idx))

        while len(open_node) > 0:
            node_idx = open_node.pop(0)
            if self.BFS_tree[node_idx].terminal:
                continue
            for child_list in self.child_list_idx_list[node_idx]:
                if np.sum(child_list)*-1 != len(child_list):  # in case obs number is not uniform, we make sure it's not a full -1 obs
                    action_count = sum(self.BFS_tree[self.node_idx_f(idx)].x_val for idx in child_list if idx != -1)  # each obervation must have an action
                    if action_count != 1:
                        print('policy node', node_idx, self.BFS_tree[node_idx].name)
                        return False
                for idx in child_list:
                    if idx != -1 and self.BFS_tree[self.node_idx_f(idx)].x_val == 1:
                        open_node.append(self.node_idx_f(idx))  # add action to open list
                        break
        return True



@njit
def randomized_rounding(gurobi_list_copy, cc, risk_list, reward_list, parent_list, child_list_idx_list, parent_act_idx_list, prune, prune_idx_list):  # prune_idx_list[old_idx] = new idx after prune (used for child list rather fully updating it)
    round_bool = np.full(len(gurobi_list_copy), False, np.bool_)
    round_bool[0] = True
    risk_sum = 0.
    objective = 0.
    for index, var in enumerate(gurobi_list_copy):
        if risk_sum > cc:
            break
        # Case: parent not selected
        if index != 0 and gurobi_list_copy[parent_list[index]] == 0.:
            #self.BFS_tree[index].x_val = 0
            gurobi_list_copy[index] = 0.
        # Case: integral sol
        #elif abs(var - 1.) < 10**-5 or abs(var - 0.) < 10**-5:
        elif round_bool[index]:
            gurobi_list_copy[index] = np.round(var)
            if gurobi_list_copy[index] == 1.:
                risk_sum += risk_list[index]
                objective += reward_list[index]
        else:
            round_child_list = np.copy(child_list_idx_list[parent_list[index]][parent_act_idx_list[index]])#self.BFS_tree[index].parent.child_list[self.BFS_tree[index].parent_idx]  # list of children to be rounded
            if prune:
                round_child_list = np.array([prune_idx_list[idx] for idx in round_child_list if idx != -1], np.int32)  # convert old idx to ne idx
            else:
                round_child_list = np.array([idx for idx in round_child_list if idx != -1], np.int32)  # convert old idx to ne idx

            round_var_list = np.array([gurobi_list_copy[idx] for idx in round_child_list], np.float64)  # list of there fractional sol

            #if len(np.where(index == round_child_list)[0]) == 0:
            #    print('error randround')
            #    print(round_child_list)
            #    print(index)
            if np.sum(round_var_list) == 0.:  # in case parent was zero so all children will be zero
                round_var_list[:] = 1.
            else:
                round_var_list += np.sum(round_var_list)/10  # TODO: adds 10% to each node to allow possibility of being selected

            round_var_list /= np.sum(round_var_list)
            # convert prob list to cumulative
            for i in range(1, len(round_var_list)):
                round_var_list[i] += round_var_list[i - 1]
            randnum = np.random.random()

            # choose item
            for i in range(len(round_var_list)):
                if randnum <= round_var_list[i]:
                    node_choice_idx = round_child_list[i]
                    break

            if node_choice_idx == index:  # if not then it will be added in the node's iteration
                risk_sum += risk_list[node_choice_idx]
                objective += reward_list[node_choice_idx]

            for idx in round_child_list:
                gurobi_list_copy[idx] = 0.
                round_bool[idx] = True
            gurobi_list_copy[node_choice_idx] = 1.

    return gurobi_list_copy, risk_sum, objective



@njit
def randomized_rounding_greedy(gurobi_list_copy, cc, risk_list, reward_list, parent_list, child_list_idx_list, parent_act_idx_list, s_horizon, depth_exp_risk, greedy_upper_exp_bool, depth_idx_list, prune, prune_idx_list):
    depth_remain_risk = np.zeros(s_horizon+1, np.float64)  # remaining risk per depth after actions are selected (expected risk - selected action)
    risk_sum = 0.  # per iteration
    objective = 0.  # per iteration
    selected_exp_risk_list = np.array([-1] * len(depth_idx_list), np.float64)

    for index, var in enumerate(gurobi_list_copy):
        if risk_sum > cc:
            break

        # Case: parent not selected
        if index != 0 and gurobi_list_copy[parent_list[index]] == 0.:
            gurobi_list_copy[index] = 0.
        # Case: integral sol
        elif abs(var - 1.) < 10**-5 or abs(var - 0.) < 10**-5:
        #elif var == 0 or var == 1:
            gurobi_list_copy[index] = np.round(var)
            if var == 1.:
                risk_sum += risk_list[index]
                objective += reward_list[index]
        # Case: fractional sol
        else:
            round_child_list = np.copy(child_list_idx_list[parent_list[index]][parent_act_idx_list[index]])  # list of children to be rounded
            if prune:
                round_child_list = np.array([prune_idx_list[idx] for idx in round_child_list if idx != -1], np.int32)  # converts child idx to correct value and remove pruned nodes (-1)
            else:
                round_child_list = np.array([idx for idx in round_child_list if idx != -1], np.int32)  # converts child idx to correct value and remove pruned nodes (-1)

            #if not np.isin(index,round_child_list):
            #    print('error randround')

            round_var_list = np.array([gurobi_list_copy[idx] for idx in round_child_list])  # list of there fractional sol
            candidate_round_idx = np.array([i for i in range(len(round_var_list)) if round_var_list[i] > 0.], np.int32)  # idx of nodes with prob>0
            for i in range(len(depth_idx_list) - 1):
                if depth_idx_list[i] <= index < depth_idx_list[i + 1]:
                    depth = i
                    break

            if selected_exp_risk_list[depth] == -1:
                selected_exp_risk = np.sum(np.array([risk_list[idx] * gurobi_list_copy[idx] for idx in range(depth_idx_list[depth], depth_idx_list[depth + 1]) if gurobi_list_copy[parent_list[idx]] == 1.], np.float64))  # exp risk for nodes with selected parents
                selected_exp_risk_list[depth] = selected_exp_risk
            else:
                selected_exp_risk = selected_exp_risk_list[depth]

            child_exp_risk = np.sum(np.array([risk_list[round_child_list[i]] * round_var_list[i] for i in candidate_round_idx], np.float64))  # exp risk of action set (children set)
            if selected_exp_risk != 0.:  # incase exp risk = 0
                exp_risk_w_upper = child_exp_risk * (depth_exp_risk[depth] + np.sum(depth_remain_risk[:depth])) / selected_exp_risk  # expected risk + unused risk from upper depth
                exp_risk = child_exp_risk * (depth_exp_risk[depth] / selected_exp_risk)  # WITHOUT upper remain
            else:
                exp_risk_w_upper = 0.
                exp_risk = 0.

            capacity = exp_risk_w_upper if greedy_upper_exp_bool else exp_risk

            if np.sum(np.array([1 for i in candidate_round_idx if risk_list[round_child_list[i]] <= capacity], np.int32))+0 > 0:  # if atleast one node is lest than risk
                for i in candidate_round_idx:
                    if risk_list[round_child_list[i]] > capacity:
                        round_var_list[i] = 0.  # set probability of zero to pick a node with risk>exp
            else:
                # if all actions are infeasible
                min_index = np.argmin(np.array([risk_list[round_child_list[idx]] for idx in candidate_round_idx], np.float64))
                round_var_list[candidate_round_idx[min_index]] = 1.

            round_var_list /= np.sum(round_var_list)

            # convert prob list to cumulative
            for i in range(1, len(round_var_list)):
                round_var_list[i] += round_var_list[i - 1]
            randnum = np.random.random()

            # choose item
            for i in range(len(round_var_list)):
                if randnum <= round_var_list[i]:
                    node_choice_idx = round_child_list[i]
                    break

            if node_choice_idx == index:  # if not then it will be added in the node's iteration
                risk_sum += risk_list[node_choice_idx]
                objective += reward_list[node_choice_idx]

            depth_remain_risk[depth] += exp_risk - risk_list[node_choice_idx]

            for idx in round_child_list:
                gurobi_list_copy[idx] = 0  # set to zero so they don't get rounded
            gurobi_list_copy[node_choice_idx] = 1

    return gurobi_list_copy, risk_sum, objective








if __name__ == '__main__':
    Grid_model = grid_model(grid_size=(3, 4), start=(0, 0), wall=[(1, 1)], risk=[(2, 2, 0.0), (0, 1, 0.5)],
                            step_cost=0.0,
                            state_rewards=[(2, 1, 10.0)], sensor_accuracy=0.81, time_horizon=4, step_horizon=4,
                            soft_horizon=4, cc=0.8, optimization='Max')

    policy = CCPOMDP(initial_belief=Grid_model.init_state, model=Grid_model)
    # grid_test(node_names=policy.node_names, model=Grid_model, sensor_accuracy=1.0)

    IPython.embed(header='LP test Done')

