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
from utils.general_utils import handle_dict_list
from utils.memory_utils import mcts_state_to_list, display_top

c_PUCT = 0.005
# Dirichlet noise alpha parameter.
NOISE_VAR = 0.00004  # 0.00001 to 0.00005

SPLIT_POOL = None
PROCESS_NUMBER = 4


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
        self.children_state_pair = {}  # to prevent duplicated split resulting the same state (could be removed)
        # self.check_split_state_pair = {}  # applied during select_expand_action(), will be clean each time
        self.split_var_index_dict = {}  # for progressive widening

        # self.subset_split_flag = [True if len(state[j]) > 0 else False for j in range(len(state))]
        for j in range(len(state)):
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
                noise_U = np.random.normal(child_U[dim], NOISE_VAR)
                # print("seed is {0} with number {1}".format(self.random_seed, str(list(noise_U))))
                child_action_score_return[dim, :len(noise_U)] = (child_Q[dim] + noise_U)
                # child_action_score_return.append((child_Q[dim] + noise_U))
        return child_action_score_return

    def select_leaf(self, k, TreeEnv, dim_per_split=100, reference_data=None, original_var=None, avg_timer_record=None):
        """
         {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}
        """
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                if reference_data is not None:
                    start_time = time.time()
                    current.select_expand_action(dim_per_split=dim_per_split, reference_data=reference_data,
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

            sorted_split_var = sorted(current.split_var_index_dict.keys(), reverse=True)
            split_flag = True
            if sorted_split_var[0] == 0:
                split_flag = False  ## we make no progress even with the largest variance reduction
                break
            for split_value in sorted_split_var:
                if len(topK_value_index_list) < k:
                    topK_value_index_list += current.split_var_index_dict.get(split_value)

            child_action_score_all = np.ones([len(current.state), self.n_actions_types, dim_per_split]) * float('-inf')
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
            current = current.find_or_add_child(action=action, TreeEnv=TreeEnv)
            end_time = time.time()
            used_time = end_time - start_time
            avg_timer_record.get('add_node')[0] += 1
            avg_timer_record.get('add_node')[1] += used_time
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
        split_values = np.sort(subset_data_dim)
        split_gap = len(split_values) / dim_per_split
        state_subset = state[subset_index]
        total_length = len(delta_data_all)
        split_value_weight_list= []
        for split_index in range(dim_per_split):
            skip_flag = False
            split_value = round(float(split_values[int(split_index * split_gap)]), 6)
            std_weighted_sum = float(0)

            split_subset_1 = []
            split_subset_delta_1 = []
            split_subset_2 = []
            split_subset_delta_2 = []
            for i in range(len(state_subset)):  # create subsplit
                if subset_data_dim[i] < split_value:
                    split_subset_1.append(state_subset[i])
                    split_subset_delta_1.append(delta_data_all[state_subset[i]])
                else:
                    split_subset_2.append(state_subset[i])
                    split_subset_delta_2.append(delta_data_all[state_subset[i]])

            new_state = []
            for state_index in range(len(state)):  # generate new state
                if state_index == subset_index:
                    new_state.append(split_subset_1)
                    new_state.append(split_subset_2)
                else:
                    new_state.append(state[state_index])

            new_state_list = mcts_state_to_list(new_state)

            action_new = "{0}_{1}_{2}".format(str(subset_index), str(dim), str(split_value))
            if check_split_state_pair.get(new_state_list) is not None:
                action = check_split_state_pair.get(new_state_list)
                if action_new != action:
                    # prevent the duplicated split (different split methods but have the same state)
                    std_weighted_sum = float('inf')
                else:
                    skip_flag = True
            else:
                check_split_state_pair.update({new_state_list: action_new})
                check_split_state_pair_new.update({new_state_list: action_new})

            if len(split_subset_1) == 0 or len(split_subset_2) == 0:
                std_weighted_sum = float('inf')  # prevent the empty set

            if add_estimate and std_weighted_sum != float('inf') and not skip_flag:
                if len(split_subset_delta_1) > 0:  # compute the greedy variance estimates
                    # mu1, std1 = norm.fit(split_subset_delta_1)
                    var1 = np.var(split_subset_delta_1)
                    std_weighted_sum += (float(len(split_subset_delta_1)) / total_length) * var1
                if len(split_subset_delta_2) > 0:
                    # mu2, std2 = norm.fit(split_subset_delta_2)
                    var2 = np.var(split_subset_delta_2)
                    std_weighted_sum += (float(len(split_subset_delta_2)) / total_length) * var2

                if not apply_global_variance:
                    weight_std_reduction = len(state_subset) / total_length * var_list[
                        subset_index] - std_weighted_sum
                else:

                    for var_index in range(len(var_list)):
                        if var_index != subset_index:
                            std_weighted_sum += float(len(state[var_index])) / total_length * \
                                                var_list[var_index]
                    weight_std_reduction = original_var - std_weighted_sum
            else:
                weight_std_reduction = -std_weighted_sum
            if not skip_flag:
                child_N_subset_dim[split_value] = 0
                child_W_subset_dim[split_value] = weight_std_reduction

            split_value_weight_list.append([split_value, weight_std_reduction, skip_flag])


        return child_N_subset_dim, child_W_subset_dim, check_split_state_pair, check_split_state_pair_new, split_value_weight_list



    def select_expand_action(self, dim_per_split, reference_data, original_var, add_estimate=False,
                             apply_global_variance=False):
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

        for subset_index in range(len(self.state)):
            subset = self.state[subset_index]
            subset_data = None
            for data_index in subset:
                data_line = np.expand_dims(np.concatenate([reference_data[data_index][0],
                                                           reference_data[data_index][3]]), axis=0)
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
                                if check_split_state_pair.get(state_list_str) is not None:
                                    action = check_split_state_pair_return[state_list_str].split('_')
                                    child_W_subset_dim[float(action[2])] = float('-inf')
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
            # Obtain state following given action.
            new_state, new_var_list = TreeEnv.next_state(deepcopy(self.state), action,
                                                              deepcopy(self.var_list))
            new_state_list = mcts_state_to_list(new_state)
            assert self.children_state_pair.get(new_state_list) is None
            self.children_state_pair.update({new_state_list: action})
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
        return_value = TreeEnv.get_return(self.state, self.depth)
        # node_string = "\033[94m|" + "----" * level
        node_string = "----" * print_indent
        # node_string += "Node: action={0}, N={1}, Q={2}, " \
        #                "return={3}|\033[0m".format(self.action, self.N, round(self.Q, 6), round(return_value, 6))
        node_string += "Level{0}|Node: action={1}, N={2}, Q={3}, U={4}, return={5}|".format(self.level, self.action, self.N,
                                                                                    round(self.Q, 6), round(self.U, 6),
                                                                                    return_value, self.state)

        # node_string += ",state:{}".format(self.state)
        print(node_string)
        if writer is not None:
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

        return_value = TreeEnv.get_return(self.state, self.depth)
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
        self.tree_save_dir = '../mimic_learner/save_tmp/mcts_save' if tree_save_dir =='' else tree_save_dir
        self.TreeEnv = TreeEnv
        # self.simulations_per_move = simulations_per_move
        self.simulations_per_round = simulations_per_round
        self.temp_threshold = None  # Overwritten in initialize_search

        self.qs = None
        self.rewards = None
        self.searches_pi = None
        self.moved_nodes = None
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
        self.moved_nodes = []
        self.states = []
        self.moved_node_str_dict_all = {}
        self.moved_node_child_dict_all = {}

    def tree_search(self, k, original_var, avg_timer_record, TreeEnv):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """

        print('\nCPUCT is {0}, pid is {1} and k is {2}'.format(c_PUCT, os.getpid(), k))
        # if num_parallel is None:
        self.TreeEnv = TreeEnv
        num_parallel = self.simulations_per_round
        leaves = []
        np.random.seed(self.random_seed)
        while len(leaves) < num_parallel:
            # print("_"*50)
            leaf, avg_timer_record = self.root.select_leaf(k=k, TreeEnv=TreeEnv,
                                                           reference_data=self.TreeEnv.data_all,
                                                           original_var=original_var,
                                                           avg_timer_record=avg_timer_record)
            value = self.TreeEnv.get_return(leaf.state, leaf.depth)
            start_time = time.time()
            leaf.backup_value(value, up_to=self.root)
            end_time = time.time()
            used_time = end_time - start_time
            avg_timer_record.get('back_up')[0] += 1
            avg_timer_record.get('back_up')[1] += used_time
            # Discourage other threads to take the same trajectory via virtual loss
            # leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        # for timer_key in avg_timer_record.keys():
        #     print("Avg Time of {0} is {1}".
        #           format(timer_key, float(avg_timer_record[timer_key][1]) / avg_timer_record[timer_key][0]))
        # self.root.print_tree(tree_writer)
        return self, avg_timer_record

    def save_mcts(self, mode, current_simulations, action_id):

        file_name = self.tree_save_dir + '_action{3}_{0}_plays{1}_{2}.pkl'.format(mode,
                                                                        current_simulations,
                                                                        datetime.today().strftime('%Y-%m-%d-%H'),
                                                                                 action_id)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

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

    def take_move(self, TreeEnv):
        root = self.root
        if len(self.moved_nodes) > 0:
            mother_state_str = ','.join(list(map(str, self.moved_nodes[-1].state[0])))
        else:
            mother_state_str = None
        selected_action, child_visit_probability, node_child_dict_all, node_str_dict_all, final_splitted_states = \
            root.select_action_by_n(TreeEnv=TreeEnv, level=0, selected_node=None,
                                         node_child_dict_all=self.moved_node_child_dict_all,
                                         node_str_dict_all=self.moved_node_str_dict_all,
                                         mother_str=mother_state_str)
        self.moved_nodes.append(self.root)
        self.searches_pi.append(child_visit_probability)
        self.qs.append(self.root.Q)
        self.states.append(root.state)

        # Resulting state becomes new root of the tree.
        self.root = root.children[selected_action]
        return_value = self.TreeEnv.get_return(self.root.state, self.root.depth)
        self.rewards.append(return_value)

        print("Moving from level {0} with var {1} to level {2} with var {3} by taking "
              "action: {4}, prob: {5}, Q: {6} and reward: {7}".format(
            self.moved_nodes[-1].level,
            self.moved_nodes[-1].var_list,
            self.root.level,
            self.root.var_list,
            selected_action,
            self.searches_pi[-1],
            self.qs[-1],
            self.rewards[-1],
        ))

        del self.root.parent.children

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


def test_mcts(model_dir, TreeEnv, action_id):
    with open(model_dir, 'rb') as f:
        mcts_read = pickle.load(f)
    mcts_read.root.print_tree(TreeEnv)
    # final_splitted_states = mcts_read.select_final_actions(mode='testing', action_id=action_id, TreeEnv=TreeEnv)
    final_splitted_states = mcts_read.moved_nodes[-1].state
    # TODO: add me if you want the final a binary tree
    # print('\n The final binary tree is:')
    # mother_str = ','.join(list(map(str, mcts_read.moved_nodes[0].state[0])))
    # PBT = PrintBinaryTree(mcts_read.moved_node_str_dict_all, mcts_read.moved_node_child_dict_all)
    # PBT.print_final_binary_tree(mother_str=mother_str, indent_number=0)

    return final_splitted_states, mcts_read.moved_nodes

    # TODO: add explanation of transferring latent variables to image (Nah, I am working on it.).


def execute_episode_single(num_simulations, TreeEnv, tree_writer,
                           mcts_saved_dir, max_k, init_state,
                           init_var_list, action_id, ignored_dim, apply_split_parallel=False):


    simulations_per_round = 2000 # 2000
    if apply_split_parallel:
        global SPLIT_POOL
        global PROCESS_NUMBER
        SPLIT_POOL = mp.Pool(processes=PROCESS_NUMBER)
    tracemalloc.start()
    from tqdm import tqdm
    pbar = tqdm(total=num_simulations)
    avg_timer_record = {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}

    mcts = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=simulations_per_round)
    # init_state, init_var_list = TreeEnv.initial_state()
    n_action_types = TreeEnv.n_action_types
    mcts.initialize_search(random_seed=0, init_state=init_state, init_var_list=init_var_list,
                           n_action_types=n_action_types, ignored_dim=ignored_dim)
    k = 5
    round_counter = 0

    global c_PUCT
    c_puct_step_size = float(c_PUCT)/ (num_simulations / simulations_per_round)

    # pre_simulations = mcts.root.N  # dummy node records the total simulation number
    current_simulations = 0
    # counter_pre_simulations = 0
    # We want `num_simulations` simulations per action not counting simulations from previous actions.
    while current_simulations <  num_simulations:

        if current_simulations % 6000 == 0:
            # mcts.root.print_tree(TreeEnv, tree_writer)
            if current_simulations > 0:
                for timer_key in avg_timer_record.keys():
                    print("Avg Time of {0} is {1}".
                          format(timer_key, float(avg_timer_record[timer_key][1]) / avg_timer_record[timer_key][0]))
            mcts.save_mcts(current_simulations=current_simulations, mode='single', action_id=action_id)

        start_time = time.time()
        mcts.tree_search(k, original_var=mcts.original_var, avg_timer_record=avg_timer_record, TreeEnv=TreeEnv)
        global c_PUCT
        c_PUCT -= c_puct_step_size
        round_counter += 1
        k = k + 1 if k < max_k else k
        end_time = time.time()
        print('single thread time is {0}'.format(str(end_time - start_time)))
        current_simulations = simulations_per_round * round_counter
        pbar.update(simulations_per_round)
        print('current simulations number is {0}'.format(current_simulations))
        # counter_pre_simulations = current_simulations
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)
        mcts.root.print_tree(TreeEnv)
        mcts.take_move(TreeEnv)


def execute_episode_parallel(num_simulations, TreeEnv, tree_writer,
                             mcts_saved_dir, max_k, init_state, init_var_list, action_id, ignored_dim):
    tracemalloc.start()
    from tqdm import tqdm
    pbar = tqdm(total=num_simulations)
    avg_timer_record = {'expand': [0, 0], 'action_score': [0, 0], 'add_node': [0, 0], 'back_up': [0, 0]}
    mcts_pool = mp.Pool(processes=5)
    mcts_threads = []

    # init_state, init_var_list = TreeEnv.initial_state()
    n_action_types = TreeEnv.n_action_types

    for i in range(5):
        mcts_thread = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=20)
        mcts_thread.initialize_search(random_seed=i + 1, init_state=init_state,
                                      init_var_list=init_var_list, n_action_types=n_action_types)
        mcts_threads.append(mcts_thread)
    mcts_origin = MCTS(None, tree_save_dir=mcts_saved_dir, simulations_per_round=20)
    mcts_origin.initialize_search(random_seed=0, init_state=init_state,
                                  init_var_list=init_var_list, n_action_types=n_action_types)
    k = 5
    while True:
        pre_simulations = mcts_threads[0].root.N  # dummy node records the total simulation number
        current_simulations = 0
        counter_pre_simulations = 0
        # We want `num_simulations` simulations per action not counting simulations from previous actions.
        while current_simulations < pre_simulations + num_simulations:

            if current_simulations % 100 == 0:
                mcts_origin.root.print_tree(TreeEnv, tree_writer)
                if current_simulations > 0:
                    for timer_key in avg_timer_record.keys():
                        print("Avg Time of {0} is {1}".
                              format(timer_key, float(avg_timer_record[timer_key][1]) / avg_timer_record[timer_key][0]))
                mcts_origin.save_mcts(current_simulations=current_simulations, mode='parallel', action_id=action_id)

            start_time = time.time()
            results = []
            for i in range(5):
                results.append(mcts_pool.apply_async(mcts_threads[i].tree_search,
                                                args=(k, mcts_threads[i].original_var, avg_timer_record, TreeEnv)))
            k = k + 1 if k < max_k else k
            mcts_results = [p.get() for p in results]
            mcts_threads = [results[0] for results in mcts_results]
            avg_timer_record = mcts_results[0][1]
            mcts_threads, mcts_origin = merge_mcts(mcts_threads, mcts_origin, ignored_dim)
            end_time = time.time()
            print('multithread time is {0}'.format(str(end_time - start_time)))
            current_simulations = mcts_threads[0].root.parent.child_N[None]
            pbar.update(current_simulations - counter_pre_simulations)
            print('current simulations number is {0}'.format(current_simulations))
            counter_pre_simulations = current_simulations

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

        mcts_origin.root.print_tree(TreeEnv)

        print('\n The extracted tree is:')
        mcts_origin.select_final_actions(mode='parallel', action_id=action_id, TreeEnv=TreeEnv)

        break
