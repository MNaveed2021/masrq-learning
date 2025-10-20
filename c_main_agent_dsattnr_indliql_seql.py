# This script is amended (with comments, print() & code modifications) by me for COMM with SEQL for RWARE.
# https://github.com/uoe-agents/seac/blob/master/seql/agent.py

import sys
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from a1_marl_utils_orig_seql import hard_update, soft_update, onehot_from_logits
from b_main_model_dsrqn_seql import DSRQNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class DSRQAgent:
    """
    Class for individual IQL agent
    """

    def __init__(self, observation_size, action_size, params):
        """
        Initialise parameters for agent
        :param observation_size: dimensions of observations
        :param action_size: dimensions of actions
        :param params: parsed arglist parameter list
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.params = params

        self.epsilon = params.epsilon
        self.epsilon_anneal_slow = params.epsilon_anneal_slow
        if self.epsilon_anneal_slow:
            self.goal_epsilon = params.goal_epsilon
            self.epsilon_decay = params.epsilon_decay
            self.decay_factor = params.decay_factor
            self.current_decay = params.decay_factor
        else:
            self.decay_factor = params.decay_factor

        # create Q-Learning networks
        self.model = DSRQNetwork(observation_size, action_size, params.hidden_dim)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.target_model = DSRQNetwork(observation_size, action_size, params.hidden_dim)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        hard_update(self.target_model, self.model)

        # create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr)

        if params.seed is not None:
            random.seed(params.seed)
            np.random.seed(params.seed)
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.t_step = 0

    def step(self, obs, explore=False, available_actions=None): ###################################################
        #print("Calling Agent.step()")
        """
        Take a step forward in environment for a minibatch of observations
        :param obs (PyTorch Variable): Observations for this agent
        :param explore (boolean): Whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                binary values indicating whether action is applicable
        :return: action (PyTorch Variable) Actions for this agent
        """
        #print("Agent.step() Obs:", obs)

        qvals = self.model(obs) # Originally

        self.t_step += 1

        if available_actions is not None:
            assert self.discrete_actions
            available_mask = torch.ByteTensor(list(map(lambda a: a == 1, available_actions)))
            negative_tensor = torch.ones(qvals.shape) * -1e9
            negative_tensor[:, available_mask] = qvals[:, available_mask]
            qvals = negative_tensor
        if explore:
            action = onehot_from_logits(qvals, self.epsilon)  # Originally == "action"
        else:
            # use small epsilon in evaluation even
            action = onehot_from_logits(qvals, 0.01)  # Originally == "action"
        #action.append(agent_action)
        #print("Agent.step() Action", action)

        if self.epsilon_anneal_slow:
            self.current_decay *= self.decay_factor
            self.epsilon = max(0.1 + (self.epsilon_decay - self.current_decay)/ self.epsilon_decay, self.goal_epsilon)
        else:
            self.epsilon *= self.decay_factor

        return action
