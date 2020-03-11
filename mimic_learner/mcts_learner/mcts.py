import math
import random as rd
import collections
from copy import deepcopy

import numpy as np
from scipy.stats import norm
# import warnings
# warnings.filterwarnings("error")
# Exploration constant
from utils.memory_utils import mcts_state_to_list

c_PUCT = 0.005  # 1.38
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5


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


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, state, n_actions_types, TreeEnv, var_list, action=None, parent=None):
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
        self.TreeEnv = TreeEnv
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
        self.check_split_state_pair = {}  # applied during select_expand_action(), will be clean each time
        self.split_var_index_dict = {}  # for progressive widening
        self.n_vlosses = 0

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

    def child_Q(self, subset_number):
        child_Q_return = []
        for dim in range(self.n_actions_types):
            child_W_subdim = []
            child_N_subdim = []
            for split_values in sorted(self.child_W[subset_number][dim].keys()):
                child_W_subdim.append(self.child_W[subset_number][dim][split_values])
                child_N_subdim.append(self.child_N[subset_number][dim][split_values])
            child_Q_return.append(np.asarray(child_W_subdim) / (1 + np.asarray(child_N_subdim)))
        return child_Q_return

    def child_U(self, subset_number):
        child_U_return = []
        for dim in range(self.n_actions_types):
            child_N_subdim = []
            for split_values in sorted(self.child_W[subset_number][dim].keys()):
                child_N_subdim.append(self.child_N[subset_number][dim][split_values])
            child_U_return.append(c_PUCT * np.sqrt(math.log(1 + self.N) / (1 + np.asarray(child_N_subdim))))
        return child_U_return

    def child_action_score(self, subset_number):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        child_Q = self.child_Q(subset_number)
        child_U = self.child_U(subset_number)
        child_action_score_return = []
        for dim in range(len(self.child_N[subset_number])):
            child_action_score_return.append(child_Q[dim] + child_U[dim])
        return child_action_score_return

    def select_leaf(self, k=5, dim_per_split=100, reference_data=None, original_var=None):
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                if reference_data is not None:
                    current.select_expand_action(dim_per_split=dim_per_split, reference_data=reference_data,
                                                 original_var=original_var,
                                                 add_estimate=True, apply_global_variance=True)
                    # TODO write progressive widening for k
                break

            topK_value_index_list = []
            for split_value in sorted(current.split_var_index_dict.keys(), reverse=True):
                if len(topK_value_index_list) < k:
                    topK_value_index_list += current.split_var_index_dict.get(split_value)

            child_action_score_all = np.ones([len(current.state), self.n_actions_types, dim_per_split]) * float('-inf')
            for j in range(len(current.state)):
                # if current.subset_split_flag[j]:
                child_action_score = current.child_action_score(subset_number=j)
                child_action_score = np.asarray(child_action_score)
                action_score_shape = child_action_score.shape
                child_action_score_all[j, :action_score_shape[0], :action_score_shape[1]] = child_action_score

            topK_action_score = [child_action_score_all[topK_value_index] for topK_value_index in topK_value_index_list]
            best_move_index = topK_value_index_list[np.argmax(topK_action_score)]
            best_subset_index = best_move_index[0]
            best_dim_index = best_move_index[1]
            best_split_value = sorted(current.child_W[best_subset_index][best_dim_index].keys())[best_move_index[2]]
            # best_move_index = np.unravel_index(child_action_score_all.argmax(), child_action_score_all.shape)
            action = "{0}_{1}_{2}".format(str(best_subset_index), str(best_dim_index), str(best_split_value))
            current = current.find_or_add_child(action=action)
        return current

    def select_expand_action(self, dim_per_split, reference_data, original_var, add_estimate=False,
                             apply_global_variance=False):
        """
        explore the possible split for each node and add greedy estimate (like the normal decision tree)
        :param dim_per_split: the number of split testes for each latent dimension
        :param reference_data: the data to be split
        :param add_estimate: 1/0 flag to decide whether to add the estimates
        :return:
        """

        total_length = len(reference_data)
        self.is_expanded = True
        split_dimension = self.n_actions_types
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
            for dim in range(split_dimension):
                split_values = np.sort(subset_data[:, dim])
                split_gap = len(split_values) / dim_per_split
                for split_index in range(dim_per_split):
                    skip_flag = False
                    split_value = round(float(split_values[int(split_index * split_gap)]), 6)
                    std_weighted_sum = float(0)

                    split_subset_1 = []
                    split_subset_delta_1 = []
                    split_subset_2 = []
                    split_subset_delta_2 = []
                    for i in range(len(subset)):  # create subsplit
                        if subset_data[i, dim] < split_value:
                            split_subset_1.append(subset[i])
                            split_subset_delta_1.append(reference_data[subset[i]][-1])
                        else:
                            split_subset_2.append(subset[i])
                            split_subset_delta_2.append(reference_data[subset[i]][-1])

                    new_state = []
                    for state_index in range(len(self.state)):  # generate new state
                        if state_index == subset_index:
                            new_state.append(split_subset_1)
                            new_state.append(split_subset_2)
                        else:
                            new_state.append(self.state[state_index])

                    new_state_list = mcts_state_to_list(new_state)

                    action_new = "{0}_{1}_{2}".format(str(subset_index), str(dim), str(split_value))
                    if self.check_split_state_pair.get(new_state_list) is not None:
                        action = self.check_split_state_pair.get(new_state_list)
                        if action_new != action:
                            # prevent the duplicated split (different split methods but have the same state)
                            std_weighted_sum = float('inf')
                        else:
                            skip_flag = True
                    else:
                        self.check_split_state_pair.update({new_state_list: action_new})

                    if len(split_subset_1) == 0 or len(split_subset_2) == 0:
                        std_weighted_sum = float('inf')  # prevent the empty set

                    if add_estimate and std_weighted_sum != float('inf') and not skip_flag:
                        if len(split_subset_delta_1) > 0:  # compute the greedy variance estimates
                            mu1, std1 = norm.fit(split_subset_delta_1)
                            std_weighted_sum += float(len(split_subset_delta_1)) / total_length * std1
                        if len(split_subset_delta_2) > 0:
                            mu2, std2 = norm.fit(split_subset_delta_2)
                            std_weighted_sum += float(len(split_subset_delta_2)) / total_length * std2

                        if not apply_global_variance:
                            weight_std_reduction = len(subset) / total_length * self.var_list[
                                subset_index] - std_weighted_sum
                        else:

                            for var_index in range(len(self.var_list)):
                                if var_index != subset_index:
                                    std_weighted_sum += float(len(self.state[var_index])) / total_length * \
                                                        self.var_list[var_index]
                            weight_std_reduction = original_var - std_weighted_sum
                    else:
                        weight_std_reduction = -std_weighted_sum
                    if not skip_flag:
                        self.child_N[subset_index][dim][split_value] = 0
                        self.child_W[subset_index][dim][split_value] = weight_std_reduction

                    if add_estimate and std_weighted_sum != float('inf') and not skip_flag:
                        # the sequence of split value is from small to large
                        split_value_index = sorted(self.child_W[subset_index][dim]).index(split_value)
                        if self.split_var_index_dict.get(-std_weighted_sum) is None:
                            self.split_var_index_dict.update(
                                {-std_weighted_sum: [(subset_index, dim, split_value_index)]})
                        else:
                            self.split_var_index_dict.get(-std_weighted_sum).append(
                                (subset_index, dim, split_value_index))

        del self.check_split_state_pair  # release memory
        print('finish expanding a node')

    def find_or_add_child(self, action):
        """
        find the children node or add new node
        :param action: the selected action in format "subset_dim_splitvalue"
        :return:
        """

        if action not in self.children:
            # Obtain state following given action.
            new_state, new_var_list = self.TreeEnv.next_state(deepcopy(self.state), action,
                                                              deepcopy(self.var_list))
            new_state_list = mcts_state_to_list(new_state)
            assert self.children_state_pair.get(new_state_list) is None
            self.children_state_pair.update({new_state_list: action})
            self.children[action] = MCTSNode(state=new_state, n_actions_types=self.n_actions_types,
                                             TreeEnv=self.TreeEnv, var_list=new_var_list,
                                             action=action, parent=self)
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

    def is_done(self):
        return self.TreeEnv.is_done_state(self.state, self.depth)

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

    def print_tree(self, level=0):
        return_value = self.TreeEnv.get_return(self.state, self.depth)
        node_string = "\033[94m|" + "----" * level
        node_string += "Node: action={0}, N={1}, Q={2}, " \
                       "return={3}|\033[0m".format(self.action, self.N, round(self.Q, 6), round(return_value, 6))
        node_string += ",state:{}".format(self.state)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level + 1)


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, TreeEnv, data, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=8):
        """
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.
        """
        # self.agent_netw = agent_netw
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None  # Overwritten in initialize_search

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.data_all = data

        self.root = None
        self.original_var = None

    def initialize_search(self, data):
        init_state, init_var_list = self.TreeEnv.initial_state(data)
        n_action_types = self.TreeEnv.n_action_types
        self.root = MCTSNode(init_state, n_action_types, self.TreeEnv, init_var_list)
        self.original_var = init_var_list[0]
        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

    def tree_search(self, original_var, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []

        while len(leaves) < num_parallel:
            # print("_"*50)
            leaf = self.root.select_leaf(reference_data=self.data_all, original_var=original_var)
            value = self.TreeEnv.get_return(leaf.state, leaf.depth)
            leaf.backup_value(value, up_to=self.root)
            # Discourage other threads to take the same trajectory via virtual loss
            # leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        # if leaves:
        #     for leaf in leaves:
        #         leaf.revert_virtual_loss(up_to=self.root)
        #         # leaf.incorporate_estimates(action_prob, value, up_to=self.root)
        # self.root.print_tree()
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action

    def take_action(self, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        """
        # Store data to be used as experience tuples.
        ob = self.TreeEnv.get_obs_for_states([self.root.state])
        self.obs.append(ob)
        self.searches_pi.append(
            self.root.visits_as_probs())  # TODO: Use self.root.position.n < self.temp_threshold as argument
        self.qs.append(self.root.Q)
        reward = (self.TreeEnv.get_return(self.root.children[action].state, self.root.children[action].depth)
                  - sum(self.rewards))
        self.rewards.append(reward)

        # Resulting state becomes new root of the tree.
        self.root = self.root.find_or_add_child(action)
        del self.root.parent.children


def execute_episode(num_simulations, TreeEnv, data):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(TreeEnv, data)
    mcts.initialize_search(data)

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    # first_node = mcts.root.select_leaf()

    while True:
        pre_simulations = mcts.root.N  # dummy node records the total simulation number
        current_simulations = 0
        # We want `num_simulations` simulations per action not counting simulations from previous actions.
        while current_simulations < pre_simulations + num_simulations:
            mcts.tree_search(original_var=mcts.original_var)
            current_simulations = mcts.root.parent.child_N[None]

        mcts.root.print_tree()
        # print("_"*100)

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
           in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)
    return (obs, mcts.searches_pi, ret, total_rew, mcts.root.state)
