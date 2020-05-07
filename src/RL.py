# standard libraries
import numpy as np
import random
import time
from collections import namedtuple, Counter
import operator
import os
from copy import deepcopy
import heapq
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


class RL():
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
        self.p_error = p_error_start
        self.p_error_end = p_error_end
        self.p_error_step = p_error_step
        self.increase_p_error_win_rate = increase_p_error_win_rate
        # Replay Memory
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_memory = replay_memory
        if self.replay_memory == 'proportional':
            self.memory = Replay_memory_prioritized(replay_memory_capacity, 0.6) # alpha
        elif self.replay_memory == 'uniform':
            self.memory = Replay_memory_uniform(replay_memory_capacity)
        else:
            raise ValueError('Invalid memory type, please use only proportional or uniform.')
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
        self.target_update = target_update


    def save_network(self, PATH):
        torch.save(self.model, PATH)


    def load_network(self, PATH):
        self.model = torch.load(PATH, map_location='cpu')
        self.model = self.model.to(self.device)


    def experience_replay(self, optimizer, criterion, batch_size):
        self.model.train()
        # get transitions and unpack them
        transitions, weights, indices = self.memory.sample(batch_size, 0.4) # beta parameter 
        action_batch, perspective_batch, q_target_batch = zip(*transitions)
        action_batch = torch.tensor([a.action - 1 for a in action_batch], device=self.device)
        q_target_batch = torch.tensor(q_target_batch, device=self.device)
        perspective_batch_tensor = torch.zeros((batch_size, *perspective_batch[0].shape), device=self.device)

        for i in range(batch_size):
            perspective_batch_tensor[i] = perspective_batch[i]

        optimizer.zero_grad()
        q_batch = self.model(perspective_batch_tensor)
        q_batch = q_batch.gather(1, action_batch.view(-1, 1)).squeeze(1)    
        loss = criterion(q_target_batch, q_batch)

        # for prioritized experience replay
        if self.replay_memory == 'proportional':
            loss = convert_from_np_to_tensor(np.array(weights)) * loss.cpu()
            priorities = loss
            priorities = np.absolute(priorities.detach().numpy())
            self.memory.priority_update(indices, priorities)

        loss = loss.mean()
        # backpropagate loss
        loss.backward()
        optimizer.step()


    def train(self, epochs, training_steps=int, optimizer=str,
        batch_size=int):
        # define criterion and optimizer
        criterion = nn.MSELoss(reduction='none')
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        elif optimizer == 'Adam':    
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        # init counters
        samples_in_memory = 0
        iteration = 0
        generated_errors = 0

        # main loop over training steps 
        while iteration < training_steps:
            num_of_steps_per_episode = 0

            # initialize syndrom
            self.toric = Toric_code(self.system_size)
            terminal_state = 0

            # generate syndroms
            self.toric.generate_random_error(self.p_error)
            generated_errors += 1
            terminal_state = self.toric.terminal_state(self.toric.current_state)

            #self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')

            # define mcts object
            mcts = MCTS(deepcopy(self.model), self.device, self.num_simulations, self.epsilon, self.discount_factor, self.grid_shift)

            old_tree = None
            loop_check = set()
            # solve one episode
            while terminal_state == 1 and num_of_steps_per_episode < self.max_nbr_actions_per_episode and iteration < training_steps:
                num_of_steps_per_episode += 1
                iteration += 1

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

                # save transitions in memory
                for (s, a), (perspective, Q) in tree.visited_PQ.items():
                    self.memory.save((a, perspective, Q), 10000)  # max priority
                    samples_in_memory += 1

                # reuse tree   
                old_tree = tree.child_nodes.get(np.array_str(self.toric.current_state))

                #print('training steps:', iteration) 
            
            # experience replay
            if samples_in_memory > self.target_update:                
                for _ in range(samples_in_memory):
                    self.experience_replay(optimizer, criterion, batch_size)

                # reset memory
                if self.replay_memory == 'proportional':
                    self.memory = Replay_memory_prioritized(self.replay_memory_capacity, 0.6)
                elif self.replay_memory == 'uniform':
                    self.memory = Replay_memory_uniform(self.replay_memory_capacity)
                
                samples_in_memory = 0


    def select_action_prediction(self):
        # set network in evluation mode 
        self.model.eval()
        # generate perspectives 
        perspectives = self.toric.generate_perspective(self.grid_shift, self.toric.current_state)
        number_of_perspectives = len(perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = convert_from_np_to_tensor(batch_perspectives)
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        # choose action 
        with torch.no_grad():
            qvals = self.model(batch_perspectives)
            qvals = np.array(qvals.cpu())
            row, col = np.where(qvals == np.max(qvals))
            perspective = row[0]
            max_q_action = col[0] + 1
            best_action = Action(batch_position_actions[perspective], max_q_action)
        return best_action


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
                # generate random syndrom
                self.toric = Toric_code(self.system_size)
                self.toric.generate_random_error(p_error)
                terminal_state = self.toric.terminal_state(self.toric.current_state)
                # plot one episode
                if plot_one_episode == True and j == 0 and i == 0:
                    self.toric.plot_toric_code(self.toric.current_state, 'initial_syndrom')
                
                init_qubit_state = deepcopy(self.toric.qubit_matrix)
                # solve syndrome
                while terminal_state == 1 and num_of_steps_per_episode < num_of_steps:
                    num_of_steps_per_episode += 1
                    action = self.select_action_prediction()
                    self.toric.step(action)
                    self.toric.current_state = self.toric.next_state
                    terminal_state = self.toric.terminal_state(self.toric.current_state)
                    
                    if plot_one_episode == True and j == 0 and i == 0:
                        self.toric.plot_toric_code(self.toric.current_state, 'step_'+str(num_of_steps_per_episode))

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

            success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
            error_corrected_list[i] = success_rate
            ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
            ground_state_list[i] =  1 - ground_state_change
            average_number_of_steps_list[i] = np.round(mean_steps_per_p_error, 1)

        return error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error


    def train_for_n_epochs(self, training_steps=int, epochs=int, num_of_predictions=100, num_of_steps_prediction=50, 
        optimizer=str, save=True, directory_path='network', prediction_list_p_error=[0.1],
        batch_size=32):

        best_win_rate = 0
        
        data_all = np.zeros((1, 17))

        training_time = 0
        prediction_time = 0

        for i in range(epochs):
            t0 = time.time()
            self.train(training_steps=training_steps,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    epochs=epochs)

            training_time += time.time() - t0
            print('training, epoch: ', i+1, 'training time:', training_time, 's')
            

            t0 = time.time()
            # evaluate network
            error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = self.prediction(num_of_predictions=num_of_predictions, 
                                                                                                                                                                        prediction_list_p_error=[self.p_error],                                                                                                                                                                        save_prediction=True,
                                                                                                                                                                        num_of_steps=num_of_steps_prediction)
            
            prediction_time += time.time() - t0

            win_rate = (num_of_predictions - len(failed_syndroms)/2) / num_of_predictions

            print('prediction, epoch: ', i+1, 'prediction time:', prediction_time, 's', 'win rate:', win_rate)

            data_all = np.append(data_all, np.array([[self.system_size, self.network_name, i+1, self.replay_memory, self.device, self.learning_rate, optimizer,
            training_steps * (i+1), prediction_list_p_error[0], num_of_predictions, len(failed_syndroms)/2, error_corrected_list[0], ground_state_list[0], average_number_of_steps_list[0], self.p_error, win_rate, training_time]]), axis=0)


            if win_rate > self.increase_p_error_win_rate and self.p_error < self.p_error_end:
                self.p_error = min(round(self.p_error + self.p_error_step, 2), self.p_error_end)

            # save training settings in txt file 
            np.savetxt(directory_path + '/data_all.txt', data_all, 
                header='system_size, network_name, epoch, replay_memory, device, learning_rate, optimizer, total_training_steps, prediction_list_p_error, number_of_predictions, number_of_failed_syndroms, error_corrected_list, ground_state_list, average_number_of_steps_list, p_error_train, win_rate, training_time', delimiter=',', fmt="%s")
            # save network
            step = (i + 1) * training_steps
            PATH = directory_path + '/network_epoch/size_{2}_{1}_epoch_{0}.pt'.format(
                i+1, self.network_name, self.system_size)
                
            # if win_rate > best_win_rate:
            #     best_win_rate = win_rate
            #     self.save_network(PATH)

            self.save_network(PATH)
            
        return error_corrected_list
