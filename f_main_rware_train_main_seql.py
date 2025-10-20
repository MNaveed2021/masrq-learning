# THIS SCRIPT IS AMENDED (with comments, print() & code modifications) BY ME
# https://github.com/uoe-agents/seac/blob/master/seql/train.py
# https://github.com/uoe-agents/seac/blob/master/seql/rware_train.py
# PyTorch requires_grad:
# https://www.educba.com/pytorch-requires_grad/
# Scalar Variable Types in Python:
# https://iq.opengenus.org/scalar-variable-types-in-python/
# numpy.mean() in Python
# https://www.geeksforgeeks.org/numpy-mean-in-python/
# https://www.sharpsightlabs.com/blog/numpy-axes-explained/

import argparse
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import rware
from a2_wrappers_seql import RecordEpisodeStatistics, TimeLimit
from a_baseline_buffer_orig_seql import MARLReplayBuffer
from e_main_iql_dsattnr_marlagent_seql import IQL
from g_training_utils_orig_seql import Logger, ModelSaver
# from i_visual_utils_seql import plot_learning_curve

#from b1_rwarehouse_main import Warehouse
#from b1_rwarehouse_main import RewardType

USE_CUDA = False  # True if torch.cuda.is_available()  # Originally False
TARGET_TYPES = ["simple", "double", "our-double", "our-clipped"]

class RwareTrain:
    def __init__(self):
        self.parser = argparse.ArgumentParser("MARL (Q-Learning) experiments for RWARE environment")
        self.parse_args()
        self.arglist = self.parser.parse_args()

    def parse_default_args(self):
        """
        Parse default arguments for MARL training script
        """
        # algorithm
        self.parser.add_argument("--hidden_dim", default=128, type=int)
        self.parser.add_argument("--shared_experience", action="store_true", default=True)
        self.parser.add_argument("--shared_lambda", default=1.0, type=float)
        self.parser.add_argument("--targets", type=str, default="simple", help="target computation used for DQN")

        # training length
        # Plan it for 6000
        self.parser.add_argument("--num_episodes", type=int, default=20000, help="number of episodes")  # Originally default=120000, My previous exps=25000
        self.parser.add_argument("--max_episode_len", type=int, default=100, help="maximum episode length")  # Originally default=25, My exps=100

        # core training parameters
        self.parser.add_argument("--n_training_threads", default=1, type=int, help="number of training threads")
        self.parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        self.parser.add_argument("--tau", type=float, default=0.05, help="tau as stepsize for target network updates")
        # Originally "use 5e-5 for RWARE, 1e-4 for LBF"
        self.parser.add_argument("--lr", type=float, default=0.0005, help="learning rate for Adam optimizer")
        self.parser.add_argument("--seed", type=int, default=7, help="random seed used throughout training") # Originally default=None
        self.parser.add_argument("--steps_per_update", type=int, default=1, help="number of steps before updates")
        self.parser.add_argument("--buffer_capacity", type=int, default=int(1e6), help="Replay buffer capacity")
        self.parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
        # epsilon = 1.0 Originally
        self.parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon value")
        self.parser.add_argument("--goal_epsilon", type=float, default=0.01, help="epsilon target value")
        self.parser.add_argument("--epsilon_decay", type=float, default=10, help="epsilon decay value")
        self.parser.add_argument("--epsilon_anneal_slow", action="store_true", default=False, help="anneal epsilon slowly")
        self.parser.add_argument("--decay_factor", action="store_true", default=0.0, help="anneal epsilon slowly")

        # visualisation
        self.parser.add_argument("--render", action="store_true", default=False)
        self.parser.add_argument("--eval_frequency", default=100, type=int, help="frequency of evaluation episodes") # Originally == 50
        self.parser.add_argument("--eval_episodes", default=5, type=int, help="number of evaluation episodes") # Originally == 5
        self.parser.add_argument("--run", type=str, default="default", help="run name for stored paths")
        self.parser.add_argument("--training_returns_freq", default=100, type=int, help='log at every this many steps')  # Originally == 100
        self.parser.add_argument("--save_interval", default=100, type=int, help='Save into csv at every this many steps')  # Originally == 100

    def parse_args(self):
        """
        parse own arguments including default args and rware specific args
        """
        self.parse_default_args()
        # Tiny == (10x11), Small == (10x20), Medium == (16x20)
        #self.parser.add_argument("--env", type=str, default="rware-tiny-2ag-v1") # DONE
        #self.parser.add_argument("--env", type=str, default="rware-small-4ag-v1") # DONE
        #self.parser.add_argument("--env", type=str, default="rware-medium-8ag-v1")  # DONE

    def extract_sizes(self, spaces):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape)
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes

    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes
        """
        # load scenario from script
        env = gym.make(self.arglist.env)
        env = RecordEpisodeStatistics(env, deque_size=100) # Originally == 10
        task_name = self.arglist.env
        print("Rware env size with agents: ", task_name)
        n_agents = env.n_agents
        print("No. of agents: ", n_agents)
        #print("Observation spaces: ", [env.observation_space[i] for i in range(n_agents)])
        #print("Action spaces: ", [env.action_space[i] for i in range(n_agents)])
        # By me
        print("Installed / default Env Reward type:", env.reward_type)
        observation_sizes = self.extract_sizes(env.observation_space)  # Sizes == dimensions
        # print("Observation_sizes: ", observation_sizes)
        action_sizes = self.extract_sizes(env.action_space)  # Sizes == dimensions
        print(f"Obs sizes or dims: {observation_sizes}, \nAction sizes or dims: {action_sizes}")

        return (env, "rware", task_name, n_agents, env.observation_space, env.action_space, observation_sizes,
            action_sizes)

    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        obs = self.env.reset()
        obs = [np.expand_dims(o, axis=0) for o in obs]
        return obs

    # Originally explore=True
    def select_actions(self, obs, explore=True):
        """
        Select actions for agents
        :param obs: joint observations for agents
        :return: actions, onehot_actions
        """
        # get actions as torch Variables
        torch_agent_actions = self.alg.step(obs, explore)
        #print("torch agent actions: ", torch_agent_actions)  # By me
        # Output == [tensor([0., 0., 1., 0., 0.]), tensor([1., 0., 0., 0., 0.])]
        # convert actions to numpy arrays
        onehot_actions = [ac.data.numpy() for ac in torch_agent_actions]
        #print("onehot_actions: ", onehot_actions)  # By me
        # Output == [array([0., 1., 0., 0., 0.], dtype=float32), array([0., 0., 1., 0., 0.], dtype=float32)]

        # convert onehot to ints
        actions = np.argmax(onehot_actions, axis=-1)
        #print("select_actions(): ", actions)  # By me
        # Output == [1 2] if output for "onehot_actions" same as above

        #print("actions:", actions, "onehot_actions:", onehot_actions)  # By me
        return actions, onehot_actions

    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors), info
        """
        # environment step
        next_obs, reward, done, info = self.env.step(actions)
        next_obs = [np.expand_dims(o, axis=0) for o in next_obs]
        return reward, done, next_obs, info

    def environment_render(self):
        """
        Render visualisation of environment
        """
        self.env.render()
        #time.sleep(0.1)

    def fill_buffer(self, timesteps):
        """
        Randomly sample actions and store experience in buffer
        :param timesteps: number of timesteps
        """
        t = 0
        while t < timesteps:
            done = False
            obs = self.reset_environment()
            while not done and t < timesteps:
                actions = [space.sample() for space in self.action_spaces]
                rewards, dones, next_obs, _ = self.environment_step(actions)
                onehot_actions = np.zeros((len(actions), self.action_sizes[0]))
                onehot_actions[np.arange(len(actions)), actions] = 1
                self.memory.add(obs, onehot_actions, rewards, next_obs, dones)
                obs = next_obs
                t += 1
                done = all(dones)

    def eval(self, ep, n_agents):
        """
        Execute evaluation episode without exploration
        :param ep: episode number
        :param n_agents: number of agents in task
        :return: returns, episode_length, done
        """
        obs = self.reset_environment()
        self.alg.reset(ep)
        episode_returns = np.array([0.0] * n_agents)
        episode_length = 0
        done = False
        while not done and episode_length < self.arglist.max_episode_len:
            torch_obs = [
                Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)]
            actions, _ = self.select_actions(torch_obs, False)
            rewards, dones, next_obs, _ = self.environment_step(actions)
            episode_returns += rewards
            obs = next_obs
            episode_length += 1
            done = all(dones)
        return episode_returns, episode_length, done

    def set_seeds(self, seed):
        """
        Set random seeds before model creation
        :param seed (int): seed to use
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def learner(self):
        """
        Training flow
        """
        # set random seeds before model creation
        self.set_seeds(self.arglist.seed)

        # use number of threads if no GPUs are available
        if not USE_CUDA:
            torch.set_num_threads(self.arglist.n_training_threads)

        env, env_name, task_name, n_agents, observation_spaces, action_spaces, observation_sizes, action_sizes = (
            self.create_environment())
        self.env = env
        self.n_agents = n_agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes

        # steps being calculated to sort out epsilon decay
        if self.arglist.max_episode_len == 25:
            steps = self.arglist.num_episodes * 20  # e.g., if steps here == 20,000 x 20 == 400,000
        else:
            steps = self.arglist.num_episodes * self.arglist.max_episode_len # e.g., if steps here == 1000 x 500 == 500,000
        # steps-th root of goal epsilon
        # self.arglist.decay_factor == 0.0
        if self.arglist.epsilon_anneal_slow:
            decay_factor = self.arglist.epsilon_decay ** (1 / float(steps)) # "epsilon_decay" == 10
            self.arglist.decay_factor = decay_factor
            print(f"Epsilon is decaying with (({self.arglist.epsilon_decay} - {decay_factor}**t) "
                  f"/ {self.arglist.epsilon_decay}) to {self.arglist.goal_epsilon} over {steps} steps.")
        else:
            decay_epsilon = self.arglist.goal_epsilon ** (1 / float(steps))
            self.arglist.decay_factor = decay_epsilon
            print(
                "Epsilon is decaying with factor %.7f to %.3f over %d steps."
                % (decay_epsilon, self.arglist.goal_epsilon, steps))

        #print("Observation sizes: ", observation_sizes)
        #print("Action sizes: ", action_sizes)

        target_type = self.arglist.targets
        if not target_type in TARGET_TYPES:
            print(f"Invalid target type {target_type}!")
            return
        else:
            if target_type == "simple":
                print("Simple target computation used")
            elif target_type == "double":
                print("Double target computation used")
            elif target_type == "our-double":
                print("Agent-double target computation used")
            elif target_type == "our-clipped":
                print("Agent-clipped target computation used")

        # create algorithm trainer
        # observation_sizes = [71, 71], action_sizes = [5, 5] for 2x Agents
        self.alg = IQL(n_agents, observation_sizes, action_sizes, self.arglist)

        obs_size = observation_sizes[0]  # Returns 1st element in the list as a scalar variable
        for o_size in observation_sizes[1:]:  # Loops over the list starting from 2nd element
            assert obs_size == o_size
        act_size = action_sizes[0]
        for a_size in action_sizes[1:]:
            assert act_size == a_size

        self.memory = MARLReplayBuffer(self.arglist.buffer_capacity, n_agents)
        # set random seeds past model creation
        self.set_seeds(self.arglist.seed)

        #self.model_saver = ModelSaver("Main_5a_Models_SARQL_tiny2Ag_2e4x100_3Mar25", self.arglist.run)  # DONE
        #self.model_saver = ModelSaver("Main_5b_Models_SARQL_small4Ag_2e4x100_3Mar25", self.arglist.run)  # DONE
        #self.model_saver = ModelSaver("Main_5c_Models_SARQL_med8Ag_2e4x100_3Mar25", self.arglist.run)  # DONE

        self.logger = Logger(n_agents, task_name, self.arglist.run)

        self.fill_buffer(5000)  # Originally, self.fill_buffer(5000)

        print("Starting iterations...")

        start_time = time.process_time()
        # timer = time.process_time()
        # env_time = 0
        # step_time = 0
        # update_time = 0
        # after_ep_time = 0

        # Both positive integer-type scalar variables as counters
        t = 0
        training_returns_saved = 0

        episode_returns = []
        episode_agent_returns = []

        # Main loop to repeat for a no. of episodes
        # self.arglist.num_episodes == 1000 now --> plan for 6000 ???
        for ep in range(self.arglist.num_episodes):
            obs = self.reset_environment()
            self.alg.reset(ep)

            # episode_returns = np.array([0.0] * n_agents)
            # Episode counter
            episode_length = 0
            done = False

            # Inner loop to generate the trajectories of experience
            # self.arglist.max_episode_len == 100 now
            while not done and episode_length < self.arglist.max_episode_len:
                # Variable; a covering on the tensor item
                # requires_grad=False applied here to freeze the portion of the model
                torch_obs = [Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)]
                #print("torch_obs: ", torch_obs)  # By me

                # env_time += time.process_time() - timer
                # timer = time.process_time()
                actions, onehot_actions = self.select_actions(torch_obs)
                #print("actions: ", actions, "onehot_actions: ", onehot_actions)  # By me
                # actions:  [2 0] onehot_actions:  [array([0., 0., 1., 0., 0.], dtype=float32), array([1., 0., 0., 0., 0.], dtype=float32)]
                # step_time += time.process_time() - timer
                # timer = time.process_time()
                rewards, dones, next_obs, info = self.environment_step(actions)

                # episode_returns += rewards

                self.memory.add(obs, onehot_actions, rewards, next_obs, dones)

                t += 1

                # env_time += time.process_time() - timer
                # timer = time.process_time()

                """Log statements & Calling log functions-I"""

                # Losses and its logging
                # self.arglist.batch_size == 128
                # self.arglist.steps_per_update == 1
                if (len(self.memory) >= self.arglist.batch_size and (t % self.arglist.steps_per_update) == 0):
                    losses = self.alg.update(self.memory, USE_CUDA) # losses
                    self.logger.log_losses(ep, losses)
                    #self.logger.dump_losses(1)

                # update_time += time.process_time() - timer
                # timer = time.process_time()

                """For displaying learned policies"""
                if self.arglist.render:
                    self.environment_render()

                obs = next_obs
                episode_length += 1
                done = all(dones)

                # Updating of episode_returns, agent_returns & episode_agent_returns;
                # at the end of each episode (before breaking one inner loop)
                # 1x Episode == 1x epoch / iteration of "self.arglist.num_episodes" ==
                # 1x self.arglist.num_episodes x self.arglist.max_episode_len
                # self.arglist.max_episode_len == 100
                if done or episode_length == self.arglist.max_episode_len:
                    episode_returns.append(info["episode_reward"])
                    agent_returns = []
                    # agent_returns appended for no. of agents with episode_reward extracted from info(dict)
                    for i in range(n_agents):
                        agent_returns.append(info[f"agent{i}/episode_reward"])
                        # By me
                        #print(f"Installed Ver Agent-wise returns: {agent_returns}")

                    episode_agent_returns.append(agent_returns)
                    #print(f"Installed Ver Ep Agent-wise returns: {episode_agent_returns}")
                    #print(len(episode_agent_returns))

            # env_time += time.process_time() - timer
            # timer = time.process_time()

            """Log statements & Calling log functions-II"""

            # Logging of mean returns and agent-wise returns over timesteps from t=100 then with the interval of 25/500
            # self.arglist.training_returns_freq == 100
            if (training_returns_saved + 1) * t >= self.arglist.training_returns_freq:
                training_returns_saved += 1
                # episodes returns become Nparray returns
                # Convert and slice list to Nparrays (last 10 / 50 / 20 arrays only) simultaneously
                returns = np.array(episode_returns[-4:])  # Originally == [-10:], mine most commonly == [-5:]

                # returns: mean cumulative return over last 10 / 5 episodes
                #mean_return = returns.mean() # Originally

                # returns: summed cumulative return over last 10 episodes
                sum_return = returns.sum()  # By me

                # agent_returns: Nparrays conversion and slice returns over last 10 / 50 / 20 episodes for each agent
                agent_returns = np.array(episode_agent_returns[-4:])  # Originally  ==  [-10:]), mine most commonly == [-5:]
                # By me
                #print(f"Installed Ver Sliced Agent-wise returns: {agent_returns}")
                #print(len(agent_returns))

                # the axis parameter controls the axis to be collapsed;
                # mean(with parameter, axis=0) -> along the columns collapsing rows,
                # mean(with parameter, axis=1) -> along the rows collapsing columns
                # This line logs the mean reward by taking mean along columns/ for each agent for last 10 sliced arrays (steps).
                #mean_agent_return = agent_returns.mean(axis=0) # Originally
                sum_agent_return = agent_returns.sum(axis=0)  # By me
                # By me
                #print(f"Installed Ver Mean Agent-wise returns: {mean_agent_return}")

                #self.logger.log_training_returns(t, mean_return, mean_agent_return) # Originally
                # For every 100 timesteps
                self.logger.log_training_returns(t, sum_return, sum_agent_return) # By me

            # Did not make any difference/ improvement on commenting eval() at every 50th ep
            # Logging of eval ep info -> returns, variances, epsilon at every 100th ep
            if ep % self.arglist.eval_frequency == 0:
                # np.zeros() returns a new array of the specified shape with element's value as 0, e.g.,
                # Here eval_episodes = 5, n_agents = 2; the resulting Nparray will be,
                # with 5x rows and 2x columns of floating point 0s
                eval_returns = np.zeros((self.arglist.eval_episodes, n_agents))
                for i in range(self.arglist.eval_episodes):
                    ep_returns, _, _ = self.eval(ep, n_agents)  # episode_returns, episode_length, done
                    eval_returns[i, :] = ep_returns
                # Originally
                #self.logger.log_episode(ep, eval_returns.mean(0), eval_returns.var(0), self.alg.agents[0].epsilon)
                self.logger.log_episode(ep, eval_returns.sum(0), eval_returns.var(0), self.alg.agents[0].epsilon)
                # calling episode info function output
                self.logger.dump_episodes(1)

            # Calling training progress info output function
            if ep % 100 == 0 and ep > 0:
                duration = time.process_time() - start_time
                self.logger.dump_train_progress(ep, self.arglist.num_episodes, duration)

            """Calling Intermediate-logs & models saving funcs"""

            # For every 100 timesteps
            if ep % self.arglist.save_interval == 0 and ep > 0:
                # save models
                print("Remove previous models")
                self.model_saver.clear_models()
                print("Saving intermediate models")
                #self.model_saver.save_models(self.alg, str(ep))
                # save intermediate logs
                print("Remove previous logs")
                self.logger.clear_logs()
                print("Saving intermediate logs")
                self.logger.save_training_returns(extension=str(ep)) # As in original repo; # For saving training returns for __ x no. of timesteps
                self.logger.save_episodes(extension=str(ep)) # For saving evaluated episodes for __ x no. of eps
                self.logger.save_losses(extension=str(ep)) # For saving losses __ x no. of eps
                # save parameter log
                # For saving parameters
                self.logger.save_parameters(env_name, task_name, n_agents, observation_sizes, action_sizes, self.arglist)

            # after_ep_time += time.process_time() - timer
            # timer = time.process_time()
            # print(f"Episode {ep} times:")
            # print(f"\tEnv time: {env_time}s")
            # print(f"\tStep time: {step_time}s")
            # print(f"\tUpdate time: {update_time}s")
            # print(f"\tAfter Ep time: {after_ep_time}s")
            # env_time = 0
            # step_time = 0
            # update_time = 0
            # after_ep_time = 0

        duration = time.process_time() - start_time
        print("Overall duration: %.2fs" % duration)

        """Calling Final-logs & models saving funcs"""

        # save models
        print("Remove previous models")
        self.model_saver.clear_models()
        print("Saving final models")
        self.model_saver.save_models(self.alg, "final")

        # save final logs
        print("Remove previous logs")
        self.logger.clear_logs()
        print("Saving final logs")
        # save_training_returns() not in original repo
        # But if not called here; the returns over 10000 eps get saved for 247525 timesteps only during intermediate logs
        self.logger.save_training_returns(extension="final")
        self.logger.save_episodes(extension="final")
        self.logger.save_losses(extension="final")
        self.logger.save_duration_cuda(duration, torch.cuda.is_available())

        # save parameter log
        self.logger.save_parameters(
            env_name,
            task_name,
            n_agents,
            observation_sizes,
            action_sizes,
            self.arglist)

        env.close()

# By me
# This is functional
    def test(self):

        # path = "BL_1a_Models_IQL_tiny2Ag_2.5e4x100_25Dec24/default"
        # Length of learning policies testing
        TOTAL_EPS = 50  # 50
        MAX_EPISODE_STEPS = 100  # 100

        (env, env_name, task_name, n_agents, observation_spaces, action_spaces,
         observation_sizes, action_sizes) = self.create_environment()

        # env = gym.make(env_name)
        # env = RecordEpisodeStatistics(env)
        env = TimeLimit(env, MAX_EPISODE_STEPS)
        self.env = env

        self.alg = IQL(n_agents, observation_sizes, action_sizes, self.arglist)
        self.alg.load_model_networks(path, extension="final")

        test_episode_returns = []
        test_episode_agent_returns = []

        for ep in range(TOTAL_EPS):
            print(f"--- Episode {ep} finished ---")
            obs = self.reset_environment()
            self.environment_render()
            # 0.0001 is for no. of seconds to halt the simulator before it moves forward
            time.sleep(0.0001)

            test_episode_length = 0
            done = False

            while not done and test_episode_length < MAX_EPISODE_STEPS:
                torch_obs = [Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)]
                actions, onehot_actions = self.select_actions(torch_obs)
                rewards, dones, next_obs, info = self.environment_step(actions)
                obs = next_obs
                test_episode_length += 1
                done = all(dones)
                if done or test_episode_length == MAX_EPISODE_STEPS:
                    print(" #------------------------# ")
                    print(f"--- Episode {ep} finished ---")
                    print("Info: ", info)
                    test_episode_returns.append(info["episode_reward"])
                    print("Test ep returns: ", test_episode_returns)
                    test_agent_returns = []
                    for i in range(n_agents):
                        test_agent_returns.append(info[f"agent{i}/episode_reward"])
                    test_episode_agent_returns.append(test_agent_returns)
                    print("Test ep agent-wise returns: ", test_episode_agent_returns)
                    print(f"This episode total reward: {sum(info['episode_reward'])}")
                    print(" #------------------------# ")
        self.env.close()

if __name__ == "__main__":
    # For training
    Train = RwareTrain()
    Train.learner()

    # For testing learned / trained policies (By me)
     #Test = RwareTrain()
     #Test.test()