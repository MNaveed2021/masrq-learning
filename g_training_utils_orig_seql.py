# https://github.com/uoe-agents/seac/blob/master/seql/utilities/logger.py
# https://github.com/uoe-agents/seac/blob/master/seql/utilities/model_saver.py
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
# https://stackoverflow.com/questions/59095824/what-is-the-difference-between-pt-pth-and-pwf-extentions-in-pytorch

import os
import sys
from collections import namedtuple
import numpy as np
import torch

class Logger:
    """
    Class to log training information
    """

    def __init__(
        self,
        n_agents,
        task_name="mape",
        run_name="default",
        #log_path = "BL_1a_Logs_IQL_tiny2Ag_2e4x100_3Mar254", # DONE
        #log_path = "BL_1b_Logs_IQL_small4Ag_2e4x100_3Mar25", # DONE
        #log_path = "BL_1c_Logs_IQL_med8Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_2a_Logs_SEQL_tiny2Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_2b_Logs_SEQL_small4Ag_2e4x100_3Mar25", # DONE
        #log_path = "BL_2c_Logs_SEQL_med8Ag_2e4x100_3Mar25", # DONE
        #log_path = "BL_3a_Logs_SAQL_tiny2Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_3b_Logs_SAQL_small4Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_3c_Logs_SAQL_med8Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_4a_Logs_RQL_tiny2Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_4b_Logs_RQL_small4Ag_2e4x100_3Mar25",  # DONE
        #log_path = "BL_4c_Logs_RQL_med8Ag_2e4x100_3Mar25",  # DONE
        #log_path = "Main_5a_Logs_SARQL_tiny2Ag_2e4x100_3Mar25",  # DONE
        #log_path = "Main_5b_Logs_SARQL_small4Ag_2e4x100_3Mar25",  # DONE
        #log_path = "Main_5c_Logs_SARQL_med8Ag_2e4x100_3Mar25",  # DONE
    ):

        """
        Create Logger instance
        :param n_agents: number of agents
        :param task_name: name of task
        :param run_name: name of run iteration
        :param log_path: path where logs should be saved
        """
        self.n_agents = n_agents
        self.task_name = task_name
        self.run_name = run_name
        self.log_path = log_path

        # episode info
        self.episode = namedtuple("Ep", "number returns variances epsilon")
        self.current_episode = 0
        self.episodes = []

        # loss info
        self.loss = namedtuple("Loss", "name episode mean variance")

        # training returns
        self.training_returns = []
        self.training_agent_returns = []

        # parameters in arrays (for efficiency)
        self.returns_means = []
        self.returns_vars = []
        self.epsilons = []
        self.alg_losses_list = [[] for i in range(n_agents)]
        # store current episode
        self.current_alg_losses_list = [[] for i in range(n_agents)]

        # alg losses
        self.alg_losses = []
        for _ in range(n_agents):
            losses = {}
            losses["qnetwork"] = []
            self.alg_losses.append(losses)

    # Logging for eval ep at every 100th ep
    def log_episode(self, ep, returns_means, returns_vars, epsilon):
        """
        Save episode information
        :param ep: episode number
        :param returns_means: average returns during episode (for each agent)
        :param returns_vars: variance of returns during episode (for each agent)
        :param epsilon: value for exploration
        """
        ep = self.episode(ep, returns_means, returns_vars, epsilon)
        self.episodes.append(ep)
        self.returns_means.append(returns_means)
        self.returns_vars.append(returns_vars)
        self.epsilons.append(epsilon)

        self.current_episode = ep

        n_losses = 0
        for l in self.current_alg_losses_list:
            n_losses += l.__len__()
        if n_losses == 0:
            return

        for i in range(self.n_agents):
            q_losses = np.array(self.current_alg_losses_list[i])
            q_loss_mean = q_losses.mean()
            q_loss = self.loss("qnetwork", ep.number, q_loss_mean, q_losses.var())
            self.alg_losses[i]["qnetwork"].append(q_loss)
            self.alg_losses_list[i].append(q_loss_mean)

        # empty current episode lists
        self.current_alg_losses_list = [[] for i in range(self.n_agents)]

    def log_training_returns(self, timestep, ret, rets):
        """
        Save mean return over last x episodes
        :param timestep (int): timestep of returns
        :param ret (float): mean cumulative return over last 10 episodes
        :param rets (List[float]): mean returns over last 10 episodes for each agent
        """
        self.training_returns.append((timestep, ret))
        self.training_agent_returns.append(rets)

    def log_losses(self, ep, losses):
        """
        Save loss information
        :param ep: episode number
        :param losses: losses of algorithm
        """
        qnet_loss = losses
        if len(qnet_loss) > 0:
            for i in range(self.n_agents):
                q_loss = qnet_loss[i].item()
                self.current_alg_losses_list[i].append(q_loss)

    def dump_episodes(self, num=None):
        """
        Output episode info
        :param num: number of last episodes to output info for (or all if None)
        """
        if num is None:
            start_idx = 0
        else:
            start_idx = -num
        print("\n\nEpisode\t\t\treturns\t\t\tvariances\t\t\t\texploration")
        for ep in self.episodes[start_idx:]:
            line = str(ep.number) + "\t\t\t"
            for ret in ep.returns:
                line += "%.3f " % ret
            line = line[:-1] + "\t\t"
            for var in ep.variances:
                line += "%.3f " % var
            line = line[:-1] + "\t\t\t"
            line += "%.3f" % ep.epsilon
            print(line)
        print()

    def __format_time(self, time):
        """
        format time from seconds to string
        :param time: time in seconds (float)
        :return: time_string
        """
        hours = time // 3600
        time -= hours * 3600
        minutes = time // 60
        time -= minutes * 60
        time_string = "%d:%d:%.2f" % (hours, minutes, time)
        return time_string

    def dump_train_progress(self, ep, num_episodes, duration):
        """
        Output training progress info
        :param ep: current episode number
        :param num_episodes: number of episodes to complete
        :param duration: training duration so far (in seconds)
        """
        print(
            "Training progress:\tepisodes: %d/%d\t\t\t\tduration: %s"
            % (ep + 1, num_episodes, self.__format_time(duration))
        )
        progress_percent = (ep + 1) / num_episodes
        remaining_duration = duration * (1 - progress_percent) / progress_percent

        arrow_len = 50
        arrow_progress = int(progress_percent * arrow_len)
        arrow_string = "|" + arrow_progress * "=" + ">" + (arrow_len - arrow_progress) * " " + "|"
        print(
            "%.2f%%\t%s\tremaining duration: %s\n"
            % (progress_percent * 100, arrow_string, self.__format_time(remaining_duration))
        )

    def dump_losses(self, num=None):
        """
        Output loss info
        :param num: number of last loss entries to output (or all if None)
        """
        num_entries = len(self.alg_losses[0]["qnetwork"])
        start_idx = 0
        if num is not None:
            start_idx = num_entries - num

        if num_entries == 0:
            #print("No loss values stored yet!")
            return

        # build header
        header = "Episode index\t\tagent_id:\t\t"
        header += "q_loss "
        print(header)

        for i in range(start_idx, num_entries):
            for j in range(self.n_agents):
                alg_loss = self.alg_losses[j]
                line = ""
                q_loss = alg_loss["qnetwork"][i]
                line += str(q_loss.episode) + "\t\t\t" + str(j + 1) + ":\t\t\t"
                line += "%.5f\t\t" % q_loss.mean  # %.5f -> 5 decimal places while floating point decimal formatting
                print(line)

    def clear_logs(self):
        """
        Remove log files in log dir
        """
        if not os.path.isdir(self.log_path):
            return
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            return
        for f in os.listdir(log_dir):
            f_path = os.path.join(log_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    """clear_logs_2() added by me; its functional
    # A sample method to remove files from a directory except one
    # For ref:
    # https://stackoverflow.com/questions/56556815/how-to-delete-everything-in-a-folder-except-one-or-two-folders-that-i-want-to-re
    
    def clear_logs_2(self):
    
        # Remove log files in log dir except 'training_returns'
        
        working_directory = r'/home/naveed2022/Desktop/My PhD Contributions/Implementation/4_I&SEQL_RWare_trial/logs_seql/default'
        retain = ["iql_rware-tiny-2ag-v1_training_returns_intermediate.csv"]
        # Change directory to your folder
        os.chdir(working_directory)
        # Loop through everything in folder in current working directory
        for item in os.listdir(working_directory):
            if item not in retain:  # If it isn't in the list for retaining
                os.remove(item)
        working_directory2 = r'/home/naveed2022/Desktop/My PhD Contributions/Implementation/4_I&SEQL_RWare_trial'
        os.chdir(working_directory2)
        """

    # Save evaluated ep at 100th ep
    def save_episodes(self, num=None, extension="final"):
        """
        Save episode information in CSV file
        :param num: number of last episodes to save (or all if None)
        :param extension: extension name of csv file
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        #csv_name = "iql_" + self.task_name + "_evalepinfo_" + extension + ".csv"  ######################################
        #csv_name = "seql_" + self.task_name + "_evalepinfo_" + extension + ".csv" ######################################
        #csv_name = "saql_" + self.task_name + "_evalepinfo_" + extension + ".csv"  ######################################
        #csv_name = "rql_" + self.task_name + "_evalepinfo_" + extension + ".csv"  ######################################
        #csv_name = "sarql_" + self.task_name + "_evalepinfo_" + extension + ".csv"  ######################################
        csv_path = os.path.join(log_dir, csv_name)

        with open(csv_path, "w") as csv_file:
            # write header line
            h = "number,returns,variances,epsilon\n"
            csv_file.write(h)

            if num is None:
                start_idx = 0
            else:
                start_idx = -num
            for ep in self.episodes[start_idx:]:
                line = ""
                line += str(ep.number) + ","
                if len(ep.returns) > 1:
                    line += "["
                    for r in ep.returns:
                        line += "%.5f " % r  # %.5f -> 5 decimal places while floating point decimal formatting
                    line = line[:-1] + "],"
                else:
                    line += str(ep.returns) + ","
                if len(ep.variances) > 1:
                    line += "["
                    for v in ep.variances:
                        line += "%.5f " % v
                    line = line[:-1] + "],"
                else:
                    line += str(ep.variances) + ","
                line += str(ep.epsilon) + "\n"
                csv_file.write(line)

    def save_training_returns(self, extension="final"):
        """
        Save training returns so far in file
        :param extension: extension name of csv file
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        # Filename and path
        #file_name = "iql_" + self.task_name + "_training_returns_" + extension + ".csv"  ###############################
        #file_name = "seql_" + self.task_name + "_training_returns_" + extension + ".csv" ###############################
        #file_name = "saql_" + self.task_name + "_training_returns_" + extension + ".csv"  ###############################
        #file_name = "rql_" + self.task_name + "_training_returns_" + extension + ".csv"  ###############################
        #file_name = "sarql_" + self.task_name + "_training_returns_" + extension + ".csv"  ###############################
        csv_path = os.path.join(log_dir, file_name)

        # Header-setter
        with open(csv_path, "w") as csv_file:
            # write header line
            h = "timestep,return,"
            for i in range(self.n_agents):
                h += f"ag{i + 1}_return,"
            h = h[:-1] + "\n"
            csv_file.write(h)

            # Data-filler loop
            for i in range(len(self.training_returns)):
                timestep, ret = self.training_returns[i]
                rets = self.training_agent_returns[i]
                line = f"{timestep},{ret},"
                for ret in rets:
                    line += "%.5f," % ret  # %.5f -> 5 decimal places while floating point decimal formatting
                line = line[:-1] + "\n"
                csv_file.write(line)

    # Save evaluated ep at 100th ep
    def save_losses(self, num=None, extension="final"):
        """
        Save loss information in CSV file
        :param num: number of last episodes to save (or all if None)
        :param extension: extension name of csv file
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        #csv_name = "iql_" + self.task_name + "_lossinfo_" + extension + ".csv"  ########################################
        #csv_name = "seql_" + self.task_name + "_lossinfo_" + extension + ".csv" ########################################
        #csv_name = "saql_" + self.task_name + "_lossinfo_" + extension + ".csv"  ########################################
        #csv_name = "rql_" + self.task_name + "_lossinfo_" + extension + ".csv"  ########################################
        #csv_name = "sarql_" + self.task_name + "_lossinfo_" + extension + ".csv"  ########################################
        csv_path = os.path.join(log_dir, csv_name)

        with open(csv_path, "w") as csv_file:
            # write header line
            h = "iteration,episode,"
            for i in range(self.n_agents):
                h += f"ag{i + 1}_iql_loss,"
            h = h[:-1] + "\n"
            csv_file.write(h)

            num_entries = len(self.alg_losses[0]["qnetwork"])
            start_idx = 0
            if num is not None:
                start_idx = num_entries - num

            for i in range(start_idx, num_entries):
                line = str(i) + ","
                for j in range(self.n_agents):
                    alg_loss = self.alg_losses[j]
                    q_loss = alg_loss["qnetwork"][i]
                    if j == 0:
                        line += str(q_loss.episode) + ","
                    line += "%.5f," % q_loss.mean
                line = line[:-1] + "\n"
                csv_file.write(line)

    def save_duration_cuda(self, duration, cuda):
        """
        Store mini log file with duration and if cuda was used
        :param duration: duration of run in seconds
        :param cuda: flag if cuda was used
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        #log_name = "iql_" + self.task_name + ".log"  ###################################################################
        #log_name = "seql_" + self.task_name + ".log" ###################################################################
        #log_name = "saql_" + self.task_name + ".log"  ###################################################################
        #log_name = "rql_" + self.task_name + ".log"  ###################################################################
        #log_name = "sarql_" + self.task_name + ".log"  ###################################################################
        log_path = os.path.join(log_dir, log_name)

        with open(log_path, "w") as log_file:
            log_file.write("duration: %.2fs\n" % duration)
            log_file.write("cuda: %s\n" % str(cuda))

    def save_parameters(
        self, env, task, n_agents, observation_sizes, action_sizes, arglist
    ):
        """
        Store mini csv file with used parameters
        :param env: environment name
        :param task: task name
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param arglist: parsed arglist of parameters
        """
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        log_dir = os.path.join(self.log_path, self.run_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        #log_name = "iql_" + self.task_name + "_parameters.csv"  ########################################################
        #log_name = "seql_" + self.task_name + "_parameters.csv" ########################################################
        #log_name = "saql_" + self.task_name + "_parameters.csv"  ########################################################
        #log_name = "rql_" + self.task_name + "_parameters.csv"  ########################################################
        #log_name = "sarql_" + self.task_name + "_parameters.csv"  ########################################################
        log_path = os.path.join(log_dir, log_name)

        with open(log_path, "w") as log_file:
            log_file.write("param,value\n")
            log_file.write("env,%s\n" % env)
            log_file.write("task,%s\n" % task)
            log_file.write("n_agents,%d\n" % n_agents)
            log_file.write("observation_sizes,%s\n" % observation_sizes)
            log_file.write("action_sizes,%s\n" % action_sizes)
            for arg in vars(arglist):
                log_file.write(arg + ",%s\n" % str(getattr(arglist, arg)))


class ModelSaver:
    """
    Class to save model parameters
    """

    def __init__(self, models_dir="models", run_name="default"):
        self.models_dir = models_dir
        self.run_name = run_name

    def clear_models(self):
        """
        Remove model files in model dir
        """
        if not os.path.isdir(self.models_dir):
            return
        model_dir = os.path.join(self.models_dir, self.run_name)
        if not os.path.isdir(model_dir):
            return
        for f in os.listdir(model_dir):
            f_path = os.path.join(model_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    def save_models(self, alg, extension):
        """
        generate and save networks
        :param model_dir_path: path of model directory
        :param run_name: name of run
        :param alg_name: name of used algorithm
        :param alg: training object of trained algorithm
        :param extension: name extension
        """
        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)
        model_dir = os.path.join(self.models_dir, self.run_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        for i, agent in enumerate(alg.agents):
            #name = "iql_agent%d_params_" % i  #########################################################################
            #name = "seql_agent%d_params_" % i #########################################################################
            #name = "saql_agent%d_params_" % i  #########################################################################
            #name = "rql_agent%d_params_" % i  #########################################################################
            #name = "sarql_agent%d_params_" % i  #########################################################################
            name += extension
            torch.save(agent.model.state_dict(), os.path.join(model_dir, name))