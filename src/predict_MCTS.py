# standard libraries
import numpy as np
import random
import time
from collections import namedtuple, Counter
import operator
import os
from copy import deepcopy
import heapq
import time
# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
# import from other files
from .toric_model import Toric_code
from .toric_model import Action
from .toric_model import Perspective
from .Replay_memory import Replay_memory_uniform, Replay_memory_prioritized
# import networks 
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .util import incremental_mean, convert_from_np_to_tensor, Transition
from .MCTS import MCTS


class predict_MCTS():
    def __init__(self, Network, Network_name, system_size=int, p_error_start=0.1, p_error_step=0.01, p_error_end=0.1, increase_p_error_win_rate=0.7,
                replay_memory_capacity=int, learning_rate=0.00025,
                max_nbr_actions_per_episode=50, device='cpu', replay_memory='uniform',
                num_simulations=10, discount_factor=0.95, epsilon=0.1, target_update=400):
        # device
        self.device = device
        # Toric code
        if system_size%2 > 0:
            self.toric = Toric_code(system_size)
        else:
            raise ValueError('Invalid system_size, please use only odd system sizes.')
        self.grid_shift = int(system_size/2)
        self.max_nbr_actions_per_episode = max_nbr_actions_per_episode
        self.system_size = system_size
        self.p_error_step = p_error_step

        # Network
        self.network_name = Network_name
        self.network = Network
        if Network == ResNet18 or Network == ResNet34 or Network == ResNet50 or Network == ResNet101 or Network == ResNet152:
            self.model = self.network()
        else:
            self.model = self.network(system_size, 3, device)
        self.model = self.model.to(self.device)
        self.learning_rate = learning_rate
        # hyperparameters RL
        self.num_simulations = num_simulations
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.solve_time = []
        self.avarage_nr_steps = []


    def load_network(self, PATH):
        self.model = torch.load(PATH, map_location='cpu')
        self.model = self.model.to(self.device)


    def prediction(self, num_of_predictions=1, num_of_steps=50, PATH=None, plot_one_episode=False, 
        show_network=False, show_plot=False, prediction_list_p_error=float, print_Q_values=False, save_prediction=True):
        # load network for prediction and set eval mode 
        if PATH != None:
            self.load_network(PATH)
        # init matrices 
        ground_state_list = np.zeros(len(prediction_list_p_error))
        error_corrected_list = np.zeros(len(prediction_list_p_error))
        average_number_of_steps_list = np.zeros(len(prediction_list_p_error))
        failed_syndroms = []
        # loop through different p_error
        for i, p_error in enumerate(prediction_list_p_error):
            ground_state = np.ones(num_of_predictions, dtype=bool)
            error_corrected = np.zeros(num_of_predictions)
            mean_steps_per_p_error = 0

            
            for j in range(num_of_predictions):
                num_of_steps_per_episode = 0
                prev_action = 0
                terminal_state = 0
                runtime = []
                #print('prediction nr ',j)
                # generate random syndrom
                self.toric = Toric_code(self.system_size)
                self.toric.generate_random_error(p_error)
                terminal_state = self.toric.terminal_state(self.toric.current_state)
                # plot one episode
                if plot_one_episode == True and j == 0 and i == 0:
                    self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')
                
                init_qubit_state = deepcopy(self.toric.qubit_matrix)

                # define mcts object
                mcts = MCTS(deepcopy(self.model), self.device, self.num_simulations, self.epsilon, self.discount_factor, self.grid_shift)
                old_tree = None
                loop_check = set()
                start_solve = time.time()
                # solve one episode
                while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                    num_of_steps_per_episode += 1
    
                    # tree search
                    tree = mcts.get_tree(old_tree, self.toric, loop_check)
    
                    # select best action among visited nodes
                    qvals_visited = list(list(zip(*tree.visited_PQ.values()))[1])
                    action = None
                    while action is None:
                        row, col = np.where(tree.Q.cpu().numpy() == max(qvals_visited))
                        perspective_index = row[0]
                        action_index = col[0] + 1
                        a = Action(tree.perspectives[1][perspective_index], action_index)
                        self.toric.step(a)
                        if np.array_str(self.toric.next_state) not in loop_check:
                            action = a
                        else:
                            qvals_visited[qvals_visited.index(max(qvals_visited))] = -1e6
                        self.toric.step(a)
    
                    loop_check.add(np.array_str(self.toric.current_state))
    
                    # take step
                    self.toric.step(action)
                    self.toric.current_state = self.toric.next_state
                    terminal_state = self.toric.terminal_state(self.toric.current_state)
    
                    #self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))
    
                    # reuse tree   
                    old_tree = tree.child_nodes.get(np.array_str(self.toric.current_state))
    
                    if plot_one_episode == True and j == 0 and i == 0:
                        self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))

                end_solve = time.time()
                solve_time = end_solve-start_solve

                # compute mean steps 
                mean_steps_per_p_error = incremental_mean(num_of_steps_per_episode, mean_steps_per_p_error, j+1)
                # save error corrected 
                error_corrected[j] = self.toric.terminal_state(self.toric.current_state)
                # update groundstate
                self.toric.eval_ground_state()                                                          
                ground_state[j] = self.toric.ground_state # False non trivial loops

                if terminal_state == 1 or self.toric.ground_state == False:
                    failed_syndroms.append(init_qubit_state)
                    failed_syndroms.append(self.toric.qubit_matrix)                
                elif terminal_state == 0 and self.toric.ground_state == True:
                    if num_of_steps_per_episode != 0 and j > 20:
                        #print('time per move: ', solve_time/num_of_steps_per_episode)
                        self.solve_time.append(solve_time)
                        self.avarage_nr_steps.append(num_of_steps_per_episode)


            success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
            error_corrected_list[i] = success_rate
            ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
            ground_state_list[i] =  1 - ground_state_change

            avarage_solve_time = np.sum(self.solve_time)/len(self.solve_time)
            avarage_steps = np.sum(self.avarage_nr_steps)/len(self.avarage_nr_steps)
            

        return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, avarage_solve_time, avarage_steps
