import os
import gc
from datetime import datetime
import math
import pickle
import tracemalloc
import time
import random as rd
import collections
from copy import deepcopy
import multiprocessing as mp
import numpy as np
# from scipy.stats import norm
# import warnings
# warnings.filterwarnings("error")
# Exploration constant
import psutil as psutil
import sys

from utils.general_utils import handle_dict_list
from utils.memory_utils import mcts_state_to_list, display_top

c_PUCT = 0.001
# Dirichlet noise alpha parameter.
NOISE_VAR = 0.00004  # 0.00001 to 0.00005

SPLIT_POOL = None
PROCESS_NUMBER = None


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    # def revert_virtual_loss(self, up_to=None): pass
    #
    # def add_virtual_loss(self, up_to=None): pass
    #
    # def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


# class MCTSNodeDiscreteAction:
#
#     def __init__(self, state, n_actions_types, TreeEnv, var_list, random_seed):
#         self.var_list = var_list  # record the variance of each subset
#         self.TreeEnv = TreeEnv
#
#         self.depth = 0
#         parent = DummyNode()
#
#         self.parent = parent
#         self.action = "DiscreteAction"  # continuous actions
#         self.state = state
#
#         self.children = {}
#
#         # self.subset_split_flag = [True if len(state[j]) > 0 else False for j in range(len(state))]
#         for j in range(len(state)):
#             assert len(state[j]) > 0
#
#         self.child_N = []
#         self.child_W = []
#
#         for i in range(n_actions_types):
#             self.child_N.append(0)
#             self.child_W.append(0)
#
#         self.is_expanded = True


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, state, n_actions_types, var_list,
                 random_seed, action, parent, level, ignored_dim=[]):
        """
        :param state: State that the node should hold.
        :param n_actions_types: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: the action (split) in our Monte Carlo Regression Tree
        :param parent: Parent node.
        """
        self.var_list = var_list  # record the variance of each subset
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.action = action  # continuous actions
        self.state = state
        self.n_actions_types = n_actions_types
        self.children = {}
        # self.children_state_pair = {}  # to prevent duplicated split resulting the same state (could be removed)
        # self.check_split_state_pair = {}  # applied during select_expand_action(), will be clean each time
        self.split_var_index_dict = {}  # for progressive widening

        # self.subset_split_flag = [True if len(state[j]) > 0 else False for j in range(len(state))]
        for j in range(len(state)):
            # if len(state[j]) <= 0:
            #     print("Error caught")
            assert len(state[j]) > 0
        self.child_N = []
        self.child_W = []
        for j in range(len(state)):
            # if self.subset_split_flag[j]:
            self.child_W.append([collections.defaultdict(float) for i in range(n_actions_types)])
            self.child_N.append([collections.defaultdict(float) for i in range(n_actions_types)])
            # else:
            #     self.child_W.append(None)
            #     self.child_N.append(None)

        self.is_expanded = False
        self.random_seed = random_seed
        self.ignored_dim = ignored_dim
        self.level = level

    @property
    def N(self):
        if self.action is not None:
            action_values = self.action.split('_')
            subset_index = int(action_values[0])
            dim = int(action_values[1])
            split_value = float(action_values[2])
            return self.parent.child_N[subset_index][dim][split_value]
        else:
            return self.parent.child_N[None]

    @N.setter
    def N(self, value):
        if self.action is not None:
            action_values = self.action.split('_')
            subset_index = int(action_values[0])
            dim = int(action_values[1])
            split_value = float(action_values[2])
            self.parent.child_N[subset_index][dim][split_value] = value
        else:
            # for Dummy node
            self.parent.child_N[self.action] = value

    @property
    def W(self):
        if self.action is not None:
            action_values = self.action.split('_')
            subset_index = int(action_values[0])
            dim = int(action_values[1])
            split_value = float(action_values[2])
            return self.parent.child_W[subset_index][dim][split_value]
        else:
            return self.parent.child_W[None]

    @W.setter
    def W(self, value):
        if self.action is not None:
            action_values = self.action.split('_')
            subset_index = int(action_values[0])
            dim = int(action_values[1])
            split_value = float(action_values[2])
            self.parent.child_W[subset_index][dim][split_value] = value
        else:
            # for Dummy node
            self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def U(self):
        """
        Returns the current U of the node.
        """
        if self.action is not None:
            return c_PUCT * np.sqrt(math.log(self.parent.N + 1) / (1 + self.N))
        else:
            return float(0)

    def child_Q(self, subset_number):
        child_Q_return = []
        for dim in range(self.n_actions_types):
            if dim not in self.ignored_dim:
                child_W_subdim = []
                child_N_subdim = []
                for split_values in sorted(self.child_W[subset_number][dim].keys()):
                    child_W_subdim.append(self.child_W[subset_number][dim][split_values])
                    child_N_subdim.append(self.child_N[subset_number][dim][split_values])
                child_Q_return.append(np.asarray(child_W_subdim) / (1 + np.asarray(child_N_subdim)))
            else:
                child_Q_return.append(None)
        return child_Q_return

    def child_U(self, subset_number):
        child_U_return = []
        for dim in range(self.n_actions_types):
            if dim not in self.ignored_dim:
                child_N_subdim = []
                for split_values in sorted(self.child_W[subset_number][dim].keys()):
                    child_N_subdim.append(self.child_N[subset_number][dim][split_values])
                child_U_return.append(c_PUCT * np.sqrt(math.log(1 + self.N) / (1 + np.asarray(child_N_subdim))))
            else:
                child_U_return.append(None)
        return child_U_return

    def child_action_score(self, subset_number):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        child_Q = self.child_Q(subset_number)
        child_U = self.child_U(subset_number)

        max_split_numbers = 0
        for dim in range(len(self.child_N[subset_number])):
            if dim not in self.ignored_dim and len(child_Q[dim]) > max_split_numbers:
                max_split_numbers = len(child_Q[dim])
        child_action_score_return = np.ones(shape=[self.n_actions_types, max_split_numbers]) * float('-inf')
        # child_action_score_return = []
        for dim in range(self.n_actions_types):
            if dim not in self.ignored_dim:
                # self.random_seed += 1
                # noise_U = np.random.normal(child_U[dim], NOISE_VAR)
                noise_U = child_U[dim]
                # print("seed is {0} with number {1}".format(self.random_seed, str(list(noise_U))))
                child_action_score_return[dim, :len(noise_U)] = (child_Q[dim] + noise_U)
                # child_action_score_return.append((child_Q[dim] + noise_U))
        return child_action_score_return

    def select_leaf(self, k, TreeEnv, max_exploration_depth = 5, dim_per_split=None,
                    original_var=None, avg_timer_record=None):
        """
         {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}
        """
        current = self
        exploration_depth = 0
        while exploration_depth<max_exploration_depth:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                if TreeEnv.data_all is not None:
                    start_time = time.time()
                    current.select_expand_action(dim_per_split=dim_per_split, reference_data=TreeEnv.data_all,
                                                 original_var=original_var,
                                                 add_estimate=True, apply_global_variance=True)
                    end_time = time.time()
                    used_time = end_time - start_time
                    # print(used_time)
                    avg_timer_record.get('expand')[0] += 1
                    avg_timer_record.get('expand')[1] += used_time
                break
            start_time = time.time()
            topK_value_index_list = []
            # TODO: add the maximum exploration depth
            sorted_split_var = sorted(current.split_var_index_dict.keys(), reverse=True)
            split_flag = True
            if sorted_split_var[0] == 0:
                split_flag = False  ## we make no progress even with the largest variance reduction
                break
            for split_value in sorted_split_var:
                if len(topK_value_index_list) < k:
                    # print(current.split_var_index_dict.get(split_value))
                    topK_value_index_list += current.split_var_index_dict.get(split_value)

            child_action_score_all = np.ones([len(current.state), self.n_actions_types, len(TreeEnv.data_all)]) * float('-inf')
            for j in range(len(current.state)):
                # if current.subset_split_flag[j]:
                child_action_score = current.child_action_score(subset_number=j)
                child_action_score = np.asarray(child_action_score)
                action_score_shape = child_action_score.shape
                # assert len(action_score_shape) == 2
                # try:
                child_action_score_all[j, :action_score_shape[0], :action_score_shape[1]] = child_action_score
                # except:
                #     print(j)
                #     print(action_score_shape)
                #     print(current.state[j])
                #     print(child_action_score)
                #     print(current.child_N[j])
                #     print(current.child_W[j])
                #     print(child_action_score_all)
                #     with open('./tmp_sub_action_score.pkl', 'wb') as f:
                #         pickle.dump(child_action_score, f)
                #     raise ValueError("check")

            topK_action_score = [child_action_score_all[topK_value_index] for topK_value_index in topK_value_index_list]
            best_move_index = topK_value_index_list[np.argmax(topK_action_score)]
            best_subset_index = best_move_index[0]
            best_dim_index = best_move_index[1]
            best_split_value = sorted(current.child_W[best_subset_index][best_dim_index].keys())[best_move_index[2]]
            end_time = time.time()
            used_time = end_time - start_time
            avg_timer_record.get('action_score')[0] += 1
            avg_timer_record.get('action_score')[1] += used_time
            # best_move_index = np.unravel_index(child_action_score_all.argmax(), child_action_score_all.shape)
            action = "{0}_{1}_{2}".format(str(best_subset_index), str(best_dim_index), str(best_split_value))
            start_time = time.time()
            if exploration_depth + 1 < max_exploration_depth:
                current = current.find_or_add_child(action=action, TreeEnv=TreeEnv)
            end_time = time.time()
            used_time = end_time - start_time
            avg_timer_record.get('add_node')[0] += 1
            avg_timer_record.get('add_node')[1] += used_time
            exploration_depth += 1
        return current, avg_timer_record

    @staticmethod
    def check_splitting_options_by_dim(dim, subset_data_dim,
                                       state, subset_index,
                                       delta_data_all, dim_per_split,
                                       var_list, original_var,
                                       add_estimate, apply_global_variance,
                                       check_split_state_pair):
        check_split_state_pair_new = {}
        child_N_subset_dim = collections.defaultdict(float)
        child_W_subset_dim = collections.defaultdict(float)
        # split_values = np.sort(subset_data_dim)
        # split_gap = len(split_values) / dim_per_split
        state_subset = state[subset_index]
        total_length = len(delta_data_all)
        split_value_weight_list= []

        assert len(state_subset) == len(subset_data_dim)
        index_dim_value = np.stack([np.asarray(state_subset), subset_data_dim], axis=1)
        # sort_index_dim_value = np.sort(np.stack([np.asarray(state_subset), subset_data_dim]), axis=1)
        sort_index_dim_value = index_dim_value[index_dim_value[:, 1].argsort()]
        # check_all_flag = False
        # if split_gap < 1:
        #     dim_per_split = len(split_values)
        #     check_all_flag = True
        split_subset_1_record = []
        split_subset_2_record = sort_index_dim_value[:, 0].tolist()
        split_subset_2_record = [int(i) for i in split_subset_2_record]  # float to int
        sort_scanned_number = 0

        split_subset_1_square_sum = 0
        split_subset_1_sum = 0
        split_subset_2_square_sum = np.sum([(delta_data_all[data_index])**2 for data_index in split_subset_2_record])
        split_subset_2_sum = np.sum([delta_data_all[data_index] for data_index in split_subset_2_record])

        for split_index in range(len(sort_index_dim_value)):
            skip_flag = False
            # if check_all_flag:
            #     split_value = round(split_values[split_index].item(), 6)
            # else:
            #     split_value = round(float(split_values[int(split_index * split_gap)]), 6)
            split_value = sort_index_dim_value[split_index][1]
            var_weighted_sum = float(0)


            for scan_index in range(sort_scanned_number, len(sort_index_dim_value)):
                if sort_index_dim_value[scan_index][1] < split_value:
                    split_subset_1_record.append(int(sort_index_dim_value[scan_index][0]))
                    split_subset_2_record.pop(0)

                    split_subset_1_square_sum += (delta_data_all[int(sort_index_dim_value[scan_index][0])]**2)
                    split_subset_1_sum += delta_data_all[int(sort_index_dim_value[scan_index][0])]

                    split_subset_2_square_sum -= (delta_data_all[int(sort_index_dim_value[scan_index][0])]**2)
                    split_subset_2_sum -= delta_data_all[int(sort_index_dim_value[scan_index][0])]

                    # split_subset_delta_1.append(delta_data_all[int(sort_index_dim_value[scan_index][0])])
                    # split_subset_delta_2.pop(0)

                else:
                    sort_scanned_number = scan_index
                    break
            # split_subset_delta_1 = []
            # split_subset_delta_2 = []
            # for i in range(len(state_subset)):  # create subsplit
            #     if subset_data_dim[i] < split_value:
            #         # split_subset_1.append(state_subset[i])
            #         split_subset_delta_1.append(delta_data_all[state_subset[i]])
            #     else:
            #         # split_subset_2.append(state_subset[i])
            #         split_subset_delta_2.append(delta_data_all[state_subset[i]])

            new_state = []
            for state_index in range(len(state)):  # generate new state
                if state_index == subset_index:
                    new_state.append(split_subset_1_record)
                    new_state.append(split_subset_2_record)
                else:
                    new_state.append(state[state_index])

            new_state_list = mcts_state_to_list(new_state)

            action_new = "{0}_{1}_{2}".format(str(subset_index), str(dim), str(split_value))
            # if action_new == '20_4_-0.056968':
            #     print('find you!')
            if check_split_state_pair.get(new_state_list) is not None:
                action = check_split_state_pair.get(new_state_list)
                if action_new != action: # prevent the duplicated split (different split methods but have the same state)
                    var_weighted_sum = float('inf')
                else:
                    var_weighted_sum = float('inf')
                    skip_flag = True
            else:
                check_split_state_pair.update({new_state_list: action_new})
                check_split_state_pair_new.update({new_state_list: action_new})

            if len(split_subset_1_record) == 0 or len(split_subset_2_record) == 0:
                var_weighted_sum = float('inf')  # prevent the empty set

            if add_estimate and var_weighted_sum != float('inf') and not skip_flag:
                if len(split_subset_1_record) > 0:  # compute the greedy variance estimates
                    # mu1, std1 = norm.fit(split_subset_delta_1)
                    var1 = split_subset_1_square_sum/float(len(split_subset_1_record)) - \
                           (split_subset_1_sum/float(len(split_subset_1_record)))**2
                    # var1_tmp = np.var(split_subset_delta_1)
                    # if abs(var1- var1_tmp) > 0.0000001:
                    #     print("catch you")

                    var_weighted_sum += (float(len(split_subset_1_record)) / total_length) * var1
                    # if var1 > 1e-6:
                    #     log_var1 = math.log(var1)
                    # else:
                    #     log_var1 = math.log(1e-6)
                    # std_weighted_sum += (float(len(split_subset_delta_1)) / total_length) * log_var1
                if len(split_subset_2_record) > 0:
                    # mu2, std2 = norm.fit(split_subset_delta_2)
                    var2 = split_subset_2_square_sum/float(len(split_subset_2_record)) - \
                           (split_subset_2_sum/float(len(split_subset_2_record)))**2
                    # var2_tmp = np.var(split_subset_delta_2)
                    # if abs(var2- var2_tmp) > 0.0000001:
                    #     print("catch you")
                    var_weighted_sum += (float(len(split_subset_2_record)) / total_length) * var2
                    # if var2 > 1e-6:
                    #     log_var2 = math.log(var2)
                    # else:
                    #     log_var2 = math.log(1e-6)
                    # std_weighted_sum += (float(len(split_subset_delta_2)) / total_lvar2_tmpength) * log_var2

                if not apply_global_variance:
                    weight_var_reduction = len(state_subset) / total_length * var_list[
                        subset_index] - var_weighted_sum
                    weight_log_var = math.log(var_weighted_sum)
                else:
                    for var_index in range(len(var_list)):
                        if var_index != subset_index:
                            var_add = var_list[var_index]
                            if var_add < 1e-6:
                                var_add = 1e-6
                            var_weighted_sum += (float(len(state[var_index])) / total_length) * var_add
                            # var_weighted_sum += float(len(state[var_index])) / total_length * math.log(var_add)

                    weight_var_reduction = original_var - var_weighted_sum
                    weight_log_var = math.log(var_weighted_sum)
            else:
                weight_var_reduction = original_var - var_weighted_sum
                weight_log_var = var_weighted_sum
            if not skip_flag:
                child_N_subset_dim[split_value] = 0
                child_W_subset_dim[split_value] = weight_var_reduction
                split_value_weight_list.append([split_value, weight_var_reduction, skip_flag])
                # child_W_subset_dim[split_value] = weight_std_reduction
                # split_value_weight_list.append([split_value, weight_std_reduction, skip_flag])


        return child_N_subset_dim, child_W_subset_dim, check_split_state_pair, check_split_state_pair_new, split_value_weight_list



    def select_expand_action(self, dim_per_split, reference_data, original_var, add_estimate=False,
                             apply_global_variance=False, max_expand_state=10):
        """
        explore the possible split for each node and add greedy estimate (like the normal decision tree)
        :param dim_per_split: the number of split testes for each latent dimension
        :param reference_data: the data to be split
        :param add_estimate: 1/0 flag to decide whether to add the estimates
        :return:
        """
        # TODO: think about how to accelerate it, maybe parallel?

        check_split_state_pair = {}
        self.is_expanded = True
        split_dimension = self.n_actions_types
        delta_data_all = []
        for data_sub_line in reference_data:
            delta_data_all.append(data_sub_line[-1])

        state_lengths = []
        for subset in self.state:
            state_lengths.append(len(subset))
        expand_state_indices = np.asarray(state_lengths).argsort()[-max_expand_state:][::-1]

        for subset_index in range(len(self.state)):
            if subset_index not in expand_state_indices:
                continue
            subset = self.state[subset_index]
            subset_data = None
            for data_index in subset:
                data_line = np.expand_dims(np.asarray(reference_data[data_index][0]),axis=0)
                if subset_data is not None:
                    subset_data = np.concatenate([subset_data, data_line])
                else:
                    subset_data = data_line
            global SPLIT_POOL
            if SPLIT_POOL is not None:
                global PROCESS_NUMBER
                # for dim in range(split_dimension):
                check_dim = 0
                assert (split_dimension-len(self.ignored_dim))%PROCESS_NUMBER == 0
                run_process_time = int((split_dimension-len(self.ignored_dim))/PROCESS_NUMBER)
                for i in range(run_process_time):
                    results = []
                    processed_dims = []
                    launch_number = 0
                    while launch_number < PROCESS_NUMBER:
                        if check_dim not in self.ignored_dim:
                            processed_dims.append(check_dim)
                            subset_data_dim = subset_data[:, check_dim]
                            results.append(SPLIT_POOL.apply_async( self.check_splitting_options_by_dim,
                                                            args=(check_dim, subset_data_dim,
                                                               self.state, subset_index,
                                                               delta_data_all, dim_per_split,
                                                               self.var_list, original_var,
                                                               add_estimate, apply_global_variance,
                                                               check_split_state_pair)))
                            launch_number+=1
                        check_dim += 1

                    split_results = [p.get() for p in results]
                    for split_result_index in range(len(split_results)):
                        split_result = split_results[split_result_index]
                        child_N_subset_dim, child_W_subset_dim, \
                        check_split_state_pair_return, check_split_state_pair_new,\
                        split_value_weight_list  = split_result
                        dim = processed_dims[split_result_index]

                        if len(check_split_state_pair) == 0:
                            check_split_state_pair.update(check_split_state_pair_return)
                        else:
                            for state_list_str in check_split_state_pair_new.keys():
                                if check_split_state_pair.get(state_list_str) is not None:  # remove duplicate splits
                                    action = check_split_state_pair_return[state_list_str].split('_')
                                    # child_W_subset_dim_tmp = deepcopy(child_W_subset_dim)
                                    child_W_subset_dim[float(action[2])] = float('-inf')
                                    eliminate_flag = False
                                    for split_value_weight_record in split_value_weight_list:
                                        if split_value_weight_record[0] == float(action[2]):
                                            split_value_weight_record[1] = float('-inf')
                                            eliminate_flag = True
                                            break
                                    assert eliminate_flag == True

                                else:
                                    check_split_state_pair.update({state_list_str:check_split_state_pair_return[state_list_str]})
                        self.child_N[subset_index][dim] = child_N_subset_dim
                        self.child_W[subset_index][dim] = child_W_subset_dim


                        sorted_child_W_subset_dim = sorted(self.child_W[subset_index][dim].keys())
                        for split_value_weight_record in split_value_weight_list:
                            if add_estimate and split_value_weight_record[1] != float('-inf') and not \
                            split_value_weight_record[-1]:
                                # the sequence of split value is from small to large
                                split_value_index = sorted_child_W_subset_dim.index(split_value_weight_record[0])
                                if self.split_var_index_dict.get(split_value_weight_record[1]) is None:
                                    self.split_var_index_dict.update(
                                        {split_value_weight_record[1]: [(subset_index, dim, split_value_index)]})
                                else:
                                    self.split_var_index_dict.get(split_value_weight_record[1]).append(
                                        (subset_index, dim, split_value_index))

            else:
                for dim in range(split_dimension):
                    if dim not in self.ignored_dim:
                        subset_data_dim = subset_data[:, dim]
                        child_N_subset_dim, child_W_subset_dim, \
                        check_split_state_pair, check_split_state_pair_new,\
                        split_value_weight_list = self.check_splitting_options_by_dim(dim, subset_data_dim,
                                                       self.state, subset_index,
                                                       delta_data_all, dim_per_split,
                                                       self.var_list, original_var,
                                                       add_estimate, apply_global_variance,
                                                       check_split_state_pair)
                        self.child_N[subset_index][dim] = child_N_subset_dim
                        self.child_W[subset_index][dim] = child_W_subset_dim

                    # subset_data_dim = subset_data[:, dim]
                    # split_values = np.sort(subset_data[:, dim])
                    # split_gap = len(split_values) / dim_per_split
                    # split_value_weight_list = []
                    # for split_index in range(dim_per_split):
                    #     skip_flag = False
                    #     split_value = round(float(split_values[int(split_index * split_gap)]), 6)
                    #     std_weighted_sum = float(0)
                    #
                    #     split_subset_1 = []
                    #     split_subset_delta_1 = []
                    #     split_subset_2 = []
                    #     split_subset_delta_2 = []
                    #     for i in range(len(subset)):  # create subsplit
                    #         if subset_data[i, dim] < split_value:
                    #             split_subset_1.append(subset[i])
                    #             split_subset_delta_1.append(reference_data[subset[i]][-1])
                    #         else:
                    #             split_subset_2.append(subset[i])
                    #             split_subset_delta_2.append(reference_data[subset[i]][-1])
                    #
                    #     new_state = []
                    #     for state_index in range(len(self.state)):  # generate new state
                    #         if state_index == subset_index:
                    #             new_state.append(split_subset_1)
                    #             new_state.append(split_subset_2)
                    #         else:
                    #             new_state.append(self.state[state_index])
                    #
                    #     new_state_list = mcts_state_to_list(new_state)
                    #
                    #     action_new = "{0}_{1}_{2}".format(str(subset_index), str(dim), str(split_value))
                    #     if self.check_split_state_pair.get(new_state_list) is not None:
                    #         action = self.check_split_state_pair.get(new_state_list)
                    #         if action_new != action:
                    #             # prevent the duplicated split (different split methods but have the same state)
                    #             std_weighted_sum = float('inf')
                    #         else:
                    #             skip_flag = True
                    #     else:
                    #         self.check_split_state_pair.update({new_state_list: action_new})
                    #
                    #     if len(split_subset_1) == 0 or len(split_subset_2) == 0:
                    #         std_weighted_sum = float('inf')  # prevent the empty set
                    #
                    #     if add_estimate and std_weighted_sum != float('inf') and not skip_flag:
                    #         if len(split_subset_delta_1) > 0:  # compute the greedy variance estimates
                    #             mu1, std1 = norm.fit(split_subset_delta_1)
                    #             std_weighted_sum += float(len(split_subset_delta_1)) / total_length * std1
                    #         if len(split_subset_delta_2) > 0:
                    #             mu2, std2 = norm.fit(split_subset_delta_2)
                    #             std_weighted_sum += float(len(split_subset_delta_2)) / total_length * std2
                    #
                    #         if not apply_global_variance:
                    #             weight_std_reduction = len(subset) / total_length * self.var_list[
                    #                 subset_index] - std_weighted_sum
                    #         else:
                    #
                    #             for var_index in range(len(self.var_list)):
                    #                 if var_index != subset_index:
                    #                     std_weighted_sum += float(len(self.state[var_index])) / total_length * \
                    #                                         self.var_list[var_index]
                    #             weight_std_reduction = original_var - std_weighted_sum
                    #     else:
                    #         weight_std_reduction = -std_weighted_sum
                    #     if not skip_flag:
                    #         self.child_N[subset_index][dim][split_value] = 0
                    #         self.child_W[subset_index][dim][split_value] = weight_std_reduction
                    #
                    #     split_value_weight_list.append([split_value, weight_std_reduction, skip_flag])

                        sorted_child_W_subset_dim = sorted(self.child_W[subset_index][dim].keys())
                        for split_value_weight_record in split_value_weight_list:
                            if add_estimate and split_value_weight_record[1] != float('-inf') and not split_value_weight_record[-1]:
                                # the sequence of split value is from small to large
                                split_value_index = sorted_child_W_subset_dim.index(split_value_weight_record[0])
                                if self.split_var_index_dict.get(split_value_weight_record[1]) is None:
                                    self.split_var_index_dict.update(
                                        {split_value_weight_record[1]: [(subset_index, dim, split_value_index)]})
                                else:
                                    self.split_var_index_dict.get(split_value_weight_record[1]).append(
                                        (subset_index, dim, split_value_index))

        del check_split_state_pair  # release memory
        # print('finish expanding a node')

    def find_or_add_child(self, action, TreeEnv):
        """
        find the children node or add new node
        :param action: the selected action in format "subset_dim_splitvalue"
        :return:
        """

        if action not in self.children:
            # if action == '20_4_-0.056968':
            #     print('find you!')
            # Obtain state following given action.
            new_state, new_var_list = TreeEnv.next_state(self.state, action, self.var_list)
            new_state_list = mcts_state_to_list(new_state)
            # assert self.children_state_pair.get(new_state_list) is None
            # self.children_state_pair.update({new_state_list: action})
            self.children[action] = MCTSNode(state=new_state, n_actions_types=self.n_actions_types,
                                              var_list=new_var_list, random_seed=self.random_seed,
                                             action=action, parent=self, level=self.level+1,
                                             ignored_dim=self.ignored_dim)
        return self.children[action]

    # def add_virtual_loss(self, up_to):
    #     """
    #     Propagate a virtual loss up to a given node.
    #     :param up_to: The node to propagate until.
    #     """
    #     self.n_vlosses += 1
    #     self.W -= 1
    #     if self.parent is None or self is up_to:
    #         return
    #     self.parent.add_virtual_loss(up_to)
    #
    # def revert_virtual_loss(self, up_to):
    #     """
    #     Undo adding virtual loss.
    #     :param up_to: The node to propagate until.
    #     """
    #     self.n_vlosses -= 1
    #     self.W += 1
    #     if self.parent is None or self is up_to:
    #         return
    #     self.parent.revert_virtual_loss(up_to)

    # def revert_visits(self, up_to):
    #     """
    #     Revert visit increments.
    #     Sometimes, repeated calls to select_leaf return the same node.
    #     This is rare and we're okay with the wasted computation to evaluate
    #     the position multiple times by the dual_net. But select_leaf has the
    #     side effect of incrementing visit counts. Since we want the value to
    #     only count once for the repeatedly selected node, we also have to
    #     revert the incremented visit counts.
    #     :param up_to: The node to propagate until.
    #     """
    #     self.N -= 1
    #     if self.parent is None or self is up_to:
    #         return
    #     self.parent.revert_visits(up_to)

    # def incorporate_estimates(self, action_probs, value, up_to):
    #     """
    #     Call if the node has just been expanded via `select_leaf` to
    #     incorporate the prior action probabilities and state value estimated
    #     by the neural network.
    #     :param action_probs: Action probabilities for the current node's state
    #     predicted by the neural network.
    #     :param value: Value of the current node's state predicted by the neural
    #     network.
    #     :param up_to: The node to propagate until.
    #     """
    #     # A done node (i.e. episode end) should not go through this code path.
    #     # Rather it should directly call `backup_value` on the final node.
    #     # TODO: Add assert here
    #     # Another thread already expanded this node in the meantime.
    #     # Ignore wasted computation but correct visit counts.
    #     if self.is_expanded:
    #         self.revert_visits(up_to=up_to)
    #         return
    #     self.original_prior = self.child_prior = action_probs
    #     # This is a deviation from the paper that led to better results in
    #     # practice (following the MiniGo implementation).
    #     self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
    #     self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    # def is_done(self):
    #     return self.TreeEnv.is_done_state(self.state, self.depth)

    # def inject_noise(self):
    #     dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
    #     self.child_prior = self.child_prior * 0.75 + dirch * 0.25
    #
    # def visits_as_probs(self, squash=False):
    #     """
    #     Returns the child visit counts as a probability distribution.
    #     :param squash: If True, exponentiate the probabilities by a temperature
    #     slightly large than 1 to encourage diversity in early steps.
    #     :return: Numpy array of shape (n_actions).
    #     """
    #     probs = self.child_N
    #     if squash:
    #         probs = probs ** .95
    #     return probs / np.sum(probs)

    def print_tree(self, TreeEnv, writer=None, print_indent=0):
        return_value = TreeEnv.get_return(self.state, self.depth, is_training=True)
        # node_string = "\033[94m|" + "----" * level
        node_string = "----" * print_indent
        # node_string += "Node: action={0}, N={1}, Q={2}, " \
        #                "return={3}|\033[0m".format(self.action, self.N, round(self.Q, 6), round(return_value, 6))
        node_string += "Level{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}|".format(self.level, self.action, self.N,
                                                                                    round(self.Q, 6), round(self.U, 6),
                                                                                    return_value, self.state)

        # node_string += ",state:{}".format(self.state)
        if writer is None:
            print(node_string)
        else:
            writer.write(node_string + "\n")
        for _, child in sorted(self.children.items()):
            child.print_tree(TreeEnv, writer, print_indent + 1)

    def select_action_by_n(self, TreeEnv, level=0, selected_node=None,
                           node_child_dict_all=None, node_str_dict_all=None,
                           mother_str=None, if_iterates_till_leaf=False):
        child_actions = []
        child_visit_number = []
        for action, child in sorted(self.children.items()):
            child_actions.append(action)
            child_visit_number.append(float(child.N))

        return_value = TreeEnv.get_return(self.state, self.depth, is_training=True)
        node_string = "----" * level
        node_string += "|Node: action={0}, N={1}, " \
                       "Q={2}, U={3}, return={4}|".format(self.action, self.N,
                                                          round(self.Q, 6), round(self.U, 6),
                                                          return_value, self.state)
        # print(node_string)
        if self.action is not None:
            split_subset_index = int(self.action.split('_')[0])

        if len(child_actions) > 0:
            child_visit_probability = np.asarray(child_visit_number) / sum(child_visit_number)
            # selected_index = np.random.choice(len(sub_child_visit_number), 1, p=child_visit_probability)[0]
            selected_index = np.argmax(child_visit_probability)
            selected_action = child_actions[selected_index]
            if selected_node is not None:
                selected_node.children[selected_action] = deepcopy(self.children[selected_action])
                next_selected_node = selected_node.children[selected_action]
            else:
                next_selected_node = None

            selected_split_subset_index = int(selected_action.split('_')[0])
            mother_str_child = ','.join(list(map(str, self.state[selected_split_subset_index])))

            if if_iterates_till_leaf:
                _, _, node_child_dict_all, node_str_dict_all, final_splitted_states = \
                    self.children[selected_action].select_action_by_n(TreeEnv=TreeEnv,
                                                                      level=level + 1,
                                                                      selected_node=next_selected_node,
                                                                      node_child_dict_all=node_child_dict_all,
                                                                      node_str_dict_all=node_str_dict_all,
                                                                      mother_str=mother_str_child,
                                                                      if_iterates_till_leaf=if_iterates_till_leaf)
            else:
                final_splitted_states = self.state
        else:
            final_splitted_states = self.state
        if self.action is None:  # root node
            root_state_str = ','.join(list(map(str, self.state[0])))
            node_string_root = "{0}|Node: action=root, return={2}, subset={3}|".format(
                '',
                self.action,
                return_value,
                root_state_str
            )
            node_str_dict_all.update({root_state_str: node_string_root})
            # print(node_str_dict_all.keys())

        else:
            subset_list = self.state
            subset_left = subset_list[split_subset_index]
            subset_left = ','.join(list(map(str, subset_left)))
            subset_right = subset_list[split_subset_index + 1]
            subset_right = ','.join(list(map(str, subset_right)))
            node_child_dict_all.update({mother_str: [subset_left, subset_right]})

            # print(node_str_dict_all.keys())

            action_details = self.action.split('_')

            node_string_left = "{0}|Node: subset{1}-dim{2}<={3}, return={4}, subset={5}|".format(
                "",
                action_details[0],
                action_details[1],
                action_details[2],
                return_value,
                subset_left
            )
            node_str_dict_all.update({subset_left: node_string_left})

            node_string_right = "{0}|Node: subset{1}-dim{2}>{3}, return={4}, subset={5}|".format(
                "",
                action_details[0],
                action_details[1],
                action_details[2],
                return_value,
                subset_right
            )
            node_str_dict_all.update({subset_right: node_string_right})

        return selected_action, child_visit_probability, node_child_dict_all, node_str_dict_all, final_splitted_states,


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, TreeEnv, tree_save_dir=None, simulations_per_round=20):
        """
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_round: Number of traversals through the tree
        before performing a step.
        """
        # self.agent_netw = agent_netw
        self.mcts_save_dir = '../mimic_learner/save_tmp/mcts_save' if tree_save_dir == '' else tree_save_dir
        self.TreeEnv = TreeEnv
        # self.simulations_per_move = simulations_per_move
        self.simulations_per_round = simulations_per_round
        self.temp_threshold = None  # Overwritten in initialize_search

        self.qs = None
        self.rewards = None
        self.searches_pi = None
        self.states = None

        self.root = None
        self.original_var = None
        self.random_seed = None

        self.moved_node_str_dict_all = None
        self.moved_node_child_dict_all = None

    def initialize_search(self, random_seed, init_state, init_var_list, n_action_types, ignored_dim):
        self.random_seed = random_seed
        self.root = MCTSNode(state=init_state, n_actions_types=n_action_types, var_list=init_var_list,
            random_seed=self.random_seed, action=None, parent=None, level=1, ignored_dim=ignored_dim)
        # state, n_actions_types, var_list, random_seed
        self.original_var = init_var_list[0]
        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.states = []
        self.moved_node_str_dict_all = {}
        self.moved_node_child_dict_all = {}

    def tree_search(self, k, original_var, avg_timer_record, TreeEnv, log_file, process, pid, apply_rollout=False):
        print('\nCPUCT is {0}, pid is {1} and k is {2}\n'.format(c_PUCT, pid, k), file=log_file)
        # if num_parallel is None:
        self.TreeEnv = TreeEnv
        num_parallel = self.simulations_per_round
        leaves = []
        np.random.seed(self.random_seed)
        print_gap = 1
        select_time_t0 = time.time()
        while len(leaves) < num_parallel:
            if len(leaves) % print_gap == 0:
                avg_select_time = (time.time()-select_time_t0)/print_gap if len(leaves)>0 else 0
                select_time_t0 = time.time()
                mme = float(process.memory_info().rss) / 1000000
                print("Working on simulations {0} currently "
                      "using memory {1} with avg time {2}".format(
                    str(len(leaves)), str(mme), str(avg_select_time)) ,file=log_file)
                if log_file is not None:
                    log_file.flush()

            # print("_"*50)
            leaf, avg_timer_record = self.root.select_leaf(k=k, TreeEnv=TreeEnv,
                                                           original_var=original_var,
                                                           avg_timer_record=avg_timer_record)
            # mme = process.memory_info().rss
            # print("Search {0} using total memory {1}".format(str(len(leaves)), str(mme)), file=log_file)

            if apply_rollout:
                value = self.TreeEnv.regression_tree_rollout(leaf.state)
            else:
                value = self.TreeEnv.get_return(leaf.state, leaf.depth, is_training=True)
            # mme = process.memory_info().rss
            # print("Return {0} using total memory {1}".format(str(len(leaves)), str(mme)), file=log_file)
            start_time = time.time()
            leaf.backup_value(value, up_to=self.root)
            # mme = process.memory_info().rss
            # print("Backup {0} using total memory {1}".format(str(len(leaves)), str(mme)), file=log_file)
            end_time = time.time()
            used_time = end_time - start_time
            avg_timer_record.get('back_up')[0] += 1
            avg_timer_record.get('back_up')[1] += used_time
            # Discourage other threads to take the same trajectory via virtual loss
            # leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        return self, avg_timer_record

    def save_mcts(self, mode, current_simulations, action_id):

        file_name = self.mcts_save_dir + '_action{3}_{0}_plays{1}_{2}.pkl'.format(mode,
                                                                                  current_simulations,
                                                                                  datetime.today().strftime('%Y-%m-%d-%H'),
                                                                                  action_id)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

        return file_name

    def select_final_actions(self, mode, action_id, TreeEnv):
        mcts_selected = MCTS(None)
        # mcts_empty.root = deepcopy(self.root)

        node_str_dict_all = {}
        node_child_dict_all = {}

        root = self.root
        # return_value = self.TreeEnv.get_return(root.state, root.depth)
        root_state_str = ','.join(list(map(str, root.state[0])))
        # node_string_root = "{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}, subset={6}|".format(
        #     '',
        #     root.action, root.N,
        #     round(root.Q, 6),
        #     round(root.U, 6),
        #     return_value,
        #     root_state_str
        # )
        # node_str_dict_all.update({root_state_str: node_string_root})

        _, _, node_child_dict_all, node_str_dict_all, final_splitted_states = \
            self.root.select_action_by_n(TreeEnv=TreeEnv, level=0, selected_node=None,
                                         node_child_dict_all=node_child_dict_all,
                                         node_str_dict_all=node_str_dict_all,
                                         mother_str=root_state_str,
                                         if_iterates_till_leaf=True)
        print('\n The final binary tree is:')
        PBT = PrintBinaryTree(node_str_dict_all, node_child_dict_all)
        PBT.print_final_binary_tree(root_state_str, indent_number=0)

        if mcts_selected.root is not None:
            mcts_selected.save_mcts(mode=mode, current_simulations='@final', action_id=action_id)

        return final_splitted_states

    # def pick_action(self):
    #     """
    #     Selects an action for the root state based on the visit counts.
    #     """
    #     if self.root.depth > self.temp_threshold:
    #         action = np.argmax(self.root.child_N)
    #     else:
    #         cdf = self.root.child_N.cumsum()
    #         cdf /= cdf[-1]
    #         selection = rd.random()
    #         action = cdf.searchsorted(selection)
    #         assert self.root.child_N[action] != 0
    #     return action

    def take_move(self, TreeEnv, log_file, process, round_counter, saved_nodes_dir):
        root = self.root
        if round_counter > 1:
            mother_state_str = ','.join(list(map(str, root.parent.state[0])))
        else:
            mother_state_str = None
        selected_action, child_visit_probability, node_child_dict_all, node_str_dict_all, final_splitted_states = \
            root.select_action_by_n(TreeEnv=TreeEnv, level=0, selected_node=None,
                                         node_child_dict_all=self.moved_node_child_dict_all,
                                         node_str_dict_all=self.moved_node_str_dict_all,
                                         mother_str=mother_state_str)

        # generate prediction value
        state_prediction = []
        for subset in root.state:
            state_target_values = []
            for data_index in subset:
                state_target_values.append(float(self.TreeEnv.data_all[data_index][-1]))
            state_prediction.append(sum(state_target_values)/len(state_target_values))
        self.root.state_prediction=state_prediction
        self.searches_pi.append(child_visit_probability)
        self.qs.append(self.root.Q)
        self.states.append(root.state)

        candidate_actions2remove = []
        # Resulting state becomes new root of the tree.
        for candidate_action in root.children.keys():
            if candidate_action == selected_action:
                self.root = root.children[selected_action]
            else:
                candidate_actions2remove.append(candidate_action)
        # mme_b = process.memory_info().rss
        for candidate_action in candidate_actions2remove:
            del root.children[candidate_action]
            gc.collect()
        mme_a = float(process.memory_info().rss)/1000000
        print("Clean total memory usage is {0} at {1}".format(mme_a, datetime.now().strftime('%Y-%m-%d %H:%M:%S')), file=log_file)
        root.parent.parent=None
        # mme_b = process.memory_info().rss
        # del root.child_N
        # del root.child_W
        save_node = MCTSNode(state=root.state, n_actions_types=root.n_actions_types, var_list=root.var_list,
                             random_seed=root.random_seed, action=root.action, parent=None, level=root.level,
                             ignored_dim=root.ignored_dim)
        save_node.state_prediction = state_prediction
        save_node.child_N = root.child_N

        tree_predictions = [None for i in range(len(TreeEnv.data_all))]
        for subset_index in range(len(save_node.state)):
            subset = save_node.state[subset_index]
            for data_index in subset:
                tree_predictions[data_index] = save_node.state_prediction[subset_index]
        ae_all = []
        se_all = []
        for data_index in range(len(tree_predictions)):
            if tree_predictions[data_index] is not None:
                real_value = TreeEnv.data_all[data_index][-1]
                predicted_value = tree_predictions[data_index]
                ae = abs(real_value-predicted_value)
                ae_all.append(ae)
                mse = ae**2
                se_all.append(mse)
        mae = np.mean(ae_all)
        mse = np.mean(se_all)
        rmse = (mse)**0.5

        print("The children of root is {0}".format(root.children), file=log_file)
        with open(saved_nodes_dir+'/node_counter_{0}_{1}.pkl'
                .format(round_counter, datetime.today().strftime('%Y-%m-%d-%H')), 'wb') as f:
            pickle.dump(obj=save_node, file=f)
        del save_node

        mme_a = float(process.memory_info().rss) / 1000000
        print("Deep copy total memory usage is {0} at {1}".format(mme_a, datetime.now().strftime('%Y-%m-%d %H:%M:%S')), file=log_file)

        return_value = self.TreeEnv.get_return(self.root.state, self.root.depth, is_training=True)


        self.rewards.append(return_value)
        print("Moving from level {0} with var {1} to level {2} with var {3} by taking "
              "action: {4}, prob: {5}, Q: {6} and return: {7}, rmse:{8}, mae:{9}".format(
            root.level,
            root.var_list,
            self.root.level,
            self.root.var_list,
            selected_action,
            self.searches_pi[-1],
            self.qs[-1],
            self.rewards[-1],
            rmse,
            mae
        ), file = log_file)


    # def transfer_final_binary_tree(self):
    #
    #     node_str_dict_all = {}
    #     node_child_dict_all = {}
    #
    #     root = self.root
    #     return_value = self.TreeEnv.get_return(root.state, root.depth)
    #     node_string_root = "{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}, subset={6}|\n".format(
    #         '',
    #         root.action, root.N,
    #         round(root.Q, 6),
    #         round(root.U, 6),
    #         return_value,
    #         root.state[0]
    #     )
    #
    #     mother_str = ','.join(list(map(str, root.state[0])))
    #     node_str_dict_all.update({root.state[0]: node_string_root})
    #     current_node = root.children.values()[0]
    #     while len(current_node.children.values()) > 1:
    #         split_subset_index = current_node.action.split('_')[0]
    #
    #         return_value = self.TreeEnv.get_return(current_node.state, current_node.depth)
    #
    #         subset_list = current_node.state
    #
    #         subset_left = subset_list[split_subset_index]
    #         subset_left = ','.join(list(map(str, subset_left)))
    #         subset_right = subset_list[split_subset_index + 1]
    #         subset_right = ','.join(list(map(str, subset_right)))
    #         node_child_dict_all.update({mother_str: [subset_left, subset_right]})
    #
    #         node_string_left = "{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}, subset={6}|\n".format(
    #             "",
    #             current_node.action, current_node.N,
    #             round(current_node.Q, 6),
    #             round(current_node.U, 6),
    #             return_value,
    #             subset_left
    #         )
    #         node_str_dict_all.update({subset_left: node_string_left})
    #
    #         node_string_right = "{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}, subset={6}|\n".format(
    #             "",
    #             current_node.action, current_node.N,
    #             round(current_node.Q, 6),
    #             round(current_node.U, 6),
    #             return_value,
    #             subset_right
    #         )
    #
    #         node_str_dict_all.update({subset_right: node_string_right})
    #
    #         assert len(current_node.children.values()) == 1
    #         mcts_children = current_node.children.values()[0]
    #         current_node = mcts_children
    #
    #     PBT = PrintBinaryTree(node_str_dict_all, node_child_dict_all)
    #     PBT.print_final_binary_tree(mother_str, indent_number=0)


class PrintBinaryTree:
    def __init__(self, node_str_dict_all, node_child_dict_all):
        self.node_str_dict_all = node_str_dict_all
        self.node_child_dict_all = node_child_dict_all

    def print_final_binary_tree(self, mother_str, indent_number):
        print("---" * indent_number + self.node_str_dict_all.get(mother_str))
        if self.node_child_dict_all.get(mother_str) is not None:
            for child_str in self.node_child_dict_all.get(mother_str):
                # for child_str in self.node_child_dict_all.get(mother_str):
                self.print_final_binary_tree(child_str, indent_number + 1)


def iterative_handle_nodes(merged_node, thread_node, origin_node, level, ignored_dim):
    update_sum = 0
    for s_index in range(level):
        assert len(merged_node.child_N) == level
        assert len(thread_node.child_N) == level
        for dim in range(merged_node.n_actions_types):
            if dim not in ignored_dim:
                for action in thread_node.child_N[s_index][dim].keys():
                    W_sum = 0
                    N_sum = 0
                    if action in merged_node.child_N[s_index][dim].keys():
                        if thread_node.child_N[s_index][dim][action] > 0:
                            W_sum += float(thread_node.child_W[s_index][dim][action])
                            N_sum += thread_node.child_N[s_index][dim][action]
                        if merged_node.child_N[s_index][dim][action] > 0:
                            W_sum += float(merged_node.child_W[s_index][dim][action])
                            N_sum += merged_node.child_N[s_index][dim][action]
                        if origin_node is not None:
                            if action in origin_node.child_N[s_index][dim].keys():
                                if origin_node.child_N[s_index][dim][action] > 0:
                                    W_sum -= float(origin_node.child_W[s_index][dim][action])
                                    N_sum -= origin_node.child_N[s_index][dim][action]
                        if W_sum > 0:
                            merged_node.child_W[s_index][dim][action] = W_sum
                        if N_sum > 0:
                            merged_node.child_N[s_index][dim][action] = N_sum
                    else:
                        W_sum += thread_node.child_W[s_index][dim][action]
                        N_sum += thread_node.child_N[s_index][dim][action]
                        if origin_node is not None:
                            if action in origin_node.child_N[s_index][dim].keys():
                                if origin_node.child_N[s_index][dim][action] > 0:
                                    W_sum -= float(origin_node.child_W[s_index][dim][action])
                                    N_sum -= origin_node.child_N[s_index][dim][action]
                        merged_node.child_W[s_index][dim][action] = W_sum
                        merged_node.child_N[s_index][dim][action] = N_sum
    if origin_node is not None:
        all(map(thread_node.children_state_pair.pop, origin_node.children_state_pair))
    merged_node.children_state_pair.update(thread_node.children_state_pair)
    if origin_node is not None:
        handle_dict_list(thread_node.split_var_index_dict, origin_node.split_var_index_dict, option='substract')
    handle_dict_list(merged_node.split_var_index_dict, thread_node.split_var_index_dict, option='add')

    for thread_child_action in thread_node.children.keys():
        if thread_child_action in merged_node.children.keys():
            origin_node_child = None
            if origin_node is not None:
                if thread_child_action in origin_node.children.keys():
                    origin_node_child = origin_node.children[thread_child_action]
                else:
                    pass
            update_sum += iterative_handle_nodes(merged_node.children[thread_child_action],
                                                 thread_node.children[thread_child_action],
                                                 origin_node_child,
                                                 level + 1, ignored_dim)
        else:
            if origin_node is not None:
                assert thread_child_action not in origin_node.children.keys()
            merged_node.children[thread_child_action] = thread_node.children[thread_child_action]
    return update_sum + 1


def merge_mcts(mcts_threads, mcts_origin, ignored_dim):
    mcts_merged = mcts_threads[0]
    # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].Q)
    # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].W)
    # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].N)
    # for mcts in mcts_threads:
    #     print("\n tree printed is: ")
    #     mcts.root.print_tree()
    update_sum = 0
    for i in range(1, len(mcts_threads)):
        merged_node = mcts_merged.root.parent
        thread_node = mcts_threads[i].root.parent
        origin_node = mcts_origin.root.parent
        level = 0
        for action in thread_node.child_N.keys():
            merged_node.child_W[action] = float(
                thread_node.child_W[action] + merged_node.child_W[action] - origin_node.child_W[action])
            # (thread_node.child_N[action] + merged_node.child_N[action])
            merged_node.child_N[action] += (thread_node.child_N[action] - origin_node.child_N[action])
        update_sum = iterative_handle_nodes(mcts_merged.root, mcts_threads[i].root, mcts_origin.root, level + 1, ignored_dim)
        # print(mcts_threads[i].root.Q)
        # print(list(mcts_merged.root.children.keys())[0])
        # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].Q)
        # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].W)
        # print(mcts_merged.root.children[list(mcts_merged.root.children.keys())[1]].N)

    print("update_sum is {0}".format(update_sum))
    mcts_threads_new = []
    for i in range(0, len(mcts_threads)):
        mcts_threads_new.append(deepcopy(mcts_merged))
        mcts_threads_new[i].random_seed = i

    mcts_origin_new = deepcopy(mcts_merged)
    del mcts_threads
    del mcts_origin
    gc.collect()
    return mcts_threads_new, mcts_origin_new


def test_mcts(saved_nodes_dir, TreeEnv, action_id):
    nodes_dirs = os.listdir(saved_nodes_dir)
    testing_moved_nodes = []
    for counter in range(len(nodes_dirs)):
        for nodes_dir in nodes_dirs:
            if "node_counter_{0}_".format(counter+1) in nodes_dir:
                with open(saved_nodes_dir+nodes_dir, 'rb') as f:
                    print("reading saved node {0}".format(counter))
                    mcts_node = pickle.load(f)
                    mcts_node.states = []
                    del mcts_node.split_var_index_dict
                    # del mcts_node.children_state_pair
                    del mcts_node.children
                    del mcts_node.child_N
                    del mcts_node.child_W
                    del mcts_node.parent
                    testing_moved_nodes.append(mcts_node)
                break

    # TODO: do you need to record the parent node?
    # print('\n The final binary tree is:')
    # mother_str = ','.join(list(map(str, mcts_read.moved_nodes[0].state[0])))
    # PBT = PrintBinaryTree(mcts_read.moved_node_str_dict_all, mcts_read.moved_node_child_dict_all)
    # PBT.print_final_binary_tree(mother_str=mother_str, indent_number=0)

    return testing_moved_nodes




def execute_episode_single(num_simulations, TreeEnv, tree_writer,
                           mcts_saved_dir, max_k, init_state,
                           init_var_list, action_id, ignored_dim,
                           shell_round_number, shell_saved_model_dir, log_file,
                           apply_split_parallel=False, save_gap=2, play=None):
    pid = os.getpid()
    process = psutil.Process(pid)  # supervise the usage of memory

    next_end_flag = False
    if play is None:
        simulations_per_round = 200 # 1000
    else:
        simulations_per_round = play
    if apply_split_parallel:
        global SPLIT_POOL
        global PROCESS_NUMBER
        PROCESS_NUMBER = 10-len(ignored_dim)
        print("Parallel process number is {0}".format(PROCESS_NUMBER))
        SPLIT_POOL = mp.Pool(processes=PROCESS_NUMBER)
    print('Process number is {0}'.format(PROCESS_NUMBER), file=log_file)
    tracemalloc.start()
    avg_timer_record = {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}
    k = 5
    global c_PUCT
    initial_c_puct = str(c_PUCT).replace('.', '_')

    c_puct_step_size = float(c_PUCT)/ (num_simulations / simulations_per_round)

    if shell_round_number is None:  # running in linux shell
        round_counter = 0
        mcts = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=simulations_per_round)
        # init_state, init_var_list = TreeEnv.initial_state()
        # n_action_types = TreeEnv.n_action_types
        mcts.initialize_search(random_seed=0, init_state=init_state, init_var_list=init_var_list,
                               n_action_types=TreeEnv.n_action_types, ignored_dim=ignored_dim)
        from tqdm import tqdm
        pbar = tqdm(total=num_simulations)
        current_simulations = 0
        saved_nodes_dir = None
    else:
        # shell_saved_model_dir = mcts_saved_dir+'_tmp_shell_saved.pkl'
        round_counter = shell_round_number*save_gap
        k = k + shell_round_number*save_gap
        k = k if k < max_k else max_k
        c_PUCT -= c_puct_step_size*(shell_round_number*save_gap)
        # pbar.update(simulations_per_round*round_counter)
        if round_counter > 0:
            with open(shell_saved_model_dir, 'rb') as f:
                mcts = pickle.load(f)
            mcts.simulations_per_round = simulations_per_round
            saved_nodes_dir = mcts.saved_nodes_dir
        else:
            mcts = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=simulations_per_round)
            mcts.initialize_search(random_seed=0, init_state=init_state, init_var_list=init_var_list,
                                   n_action_types=TreeEnv.n_action_types, ignored_dim=ignored_dim)
            saved_nodes_dir = mcts_saved_dir.replace('saved_model','saved_nodes')+'_action'+str(action_id)+'_CPUCT'\
                              +str(initial_c_puct)+'_'+datetime.today().strftime('%Y-%m-%d')
            mcts.saved_nodes_dir=saved_nodes_dir
            if not os.path.exists(saved_nodes_dir):
                os.mkdir(saved_nodes_dir)

        current_simulations = simulations_per_round * round_counter

        print('launch mcts for round {0}\n'.format(round_counter), file = log_file)
        if log_file is not None:
            log_file.flush()

    # pre_simulations = mcts.root.N  # dummy node records the total simulation number
    # counter_pre_simulations = 0
    # We want `num_simulations` simulations per action not counting simulations from previous actions.
    while current_simulations <  num_simulations:

        if current_simulations % (simulations_per_round*save_gap) == 0:
            # mcts.root.print_tree(TreeEnv, tree_writer)
            if next_end_flag:  # skip the first save
                for timer_key in avg_timer_record.keys():
                    print("Avg Time of {0} is {1}".
                          format(timer_key, float(avg_timer_record[timer_key][1]) / avg_timer_record[timer_key][0]),
                          file=log_file)
                # mcts_file_name = mcts.save_mcts(current_simulations=current_simulations,
                #                                 mode='single', action_id=action_id)

            if shell_round_number is not None and next_end_flag:
                with open(shell_saved_model_dir, 'wb') as f:
                    pickle.dump(mcts, f)
                print('finishing mcts before round {0}\n'.format(round_counter), file = log_file)
                break
            next_end_flag = True

        start_time = time.time()
        mcts.tree_search(k, original_var=mcts.original_var,
                         avg_timer_record=avg_timer_record,
                         TreeEnv=TreeEnv, log_file=log_file,
                         process=process, pid=pid)
        # global c_PUCT
        c_PUCT -= c_puct_step_size
        round_counter += 1
        k = k + 1 if k < max_k else k
        end_time = time.time()
        print('Single thread time is {0}\n'.format(str(end_time - start_time)), file = log_file)
        current_simulations = simulations_per_round * round_counter
        if shell_round_number is None:
            pbar.update(simulations_per_round)
        print('Current simulations number is {0}\n'.format(current_simulations), file = log_file)
        # counter_pre_simulations = current_simulations
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)
        mcts.root.print_tree(TreeEnv, writer=log_file)
        if log_file is not None:
            log_file.flush()
        mcts.take_move(TreeEnv, log_file, process, round_counter, saved_nodes_dir)
        if log_file is not None:
            log_file.flush()



# def execute_episode_parallel(num_simulations, TreeEnv, tree_writer,
#                              mcts_saved_dir, max_k, init_state, init_var_list, action_id, ignored_dim):
#     tracemalloc.start()
#     from tqdm import tqdm
#     pbar = tqdm(total=num_simulations)
#     avg_timer_record = {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}
#     mcts_pool = mp.Pool(processes=5)
#     mcts_threads = []
#
#     # init_state, init_var_list = TreeEnv.initial_state()
#     n_action_types = TreeEnv.n_action_types
#
#     for i in range(5):
#         mcts_thread = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=20)
#         mcts_thread.initialize_search(random_seed=i + 1, init_state=init_state,
#                                       init_var_list=init_var_list, n_action_types=n_action_types)
#         mcts_threads.append(mcts_thread)
#     mcts_origin = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=20)
#     mcts_origin.initialize_search(random_seed=0, init_state=init_state,
#                                   init_var_list=init_var_list, n_action_types=n_action_types)
#     k = 5
#     while True:
#         pre_simulations = mcts_threads[0].root.N  # dummy node records the total simulation number
#         current_simulations = 0
#         counter_pre_simulations = 0
#         # We want `num_simulations` simulations per action not counting simulations from previous actions.
#         while current_simulations < pre_simulations + num_simulations:
#
#             if current_simulations % 100 == 0:
#                 mcts_origin.root.print_tree(TreeEnv, tree_writer)
#                 if current_simulations > 0:
#                     for timer_key in avg_timer_record.keys():
#                         print("Avg Time of {0} is {1}".
#                               format(timer_key, float(avg_timer_record[timer_key][1]) / avg_timer_record[timer_key][0]),
#                               file = sys.stderr)
#                 mcts_origin.save_mcts(current_simulations=current_simulations, mode='parallel', action_id=action_id)
#
#             start_time = time.time()
#             results = []
#             for i in range(5):
#                 results.append(mcts_pool.apply_async(mcts_threads[i].tree_search,
#                                                 args=(k, mcts_threads[i].original_var, avg_timer_record, TreeEnv)))
#             k = k + 1 if k < max_k else k
#             mcts_results = [p.get() for p in results]
#             mcts_threads = [results[0] for results in mcts_results]
#             avg_timer_record = mcts_results[0][1]
#             mcts_threads, mcts_origin = merge_mcts(mcts_threads, mcts_origin, ignored_dim)
#             end_time = time.time()
#             print('multithread time is {0}'.format(str(end_time - start_time)), file = sys.stderr)
#             current_simulations = mcts_threads[0].root.parent.child_N[None]
#             pbar.update(current_simulations - counter_pre_simulations)
#             print('current simulations number is {0}'.format(current_simulations), file = sys.stderr)
#             counter_pre_simulations = current_simulations
#
#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)
#
#         mcts_origin.root.print_tree(TreeEnv)
#
#         print('\n The extracted tree is:')
#         mcts_origin.select_final_actions(mode='parallel', action_id=action_id, TreeEnv=TreeEnv)
#
#         break
