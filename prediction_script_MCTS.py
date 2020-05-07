import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.predict_MCTS import predict_MCTS

from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


##########################################################################
def predict_MCTS_func(syst_size, Network_File_Name, Network_type, num_predict, p_err):
    start = time.time()
    device = 'cuda'
    
    # common system sizes are 3,5,7 and 9, grid size must be odd! 
    system_size = syst_size
    
    network = Network_type
    
    # this file is stored in the network folder and contains the trained agent that will guide the MCTS  
    NETWORK_FILE_NAME = Network_File_Name
    
    # Generate folder structure, all results are stored in the data folder 
    timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
    PATH = 'data/prediction_Time_' +str(NETWORK_FILE_NAME) +'__'+  timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    # Path for the network to use for the prediction
    PATH2 = 'network/Fitzek et al/'+str(NETWORK_FILE_NAME)+'.pt'
    
    num_of_predictions = num_predict
    p_e = p_err
    
    data_all = np.zeros((1, 8))
    data_result = np.zeros((1, 2))
    data_steps = np.zeros((1, 3))
    for i in p_e:
        # initialize RL class
        rl = predict_MCTS(Network=network,
                Network_name=NETWORK_FILE_NAME,
                system_size=system_size,
                device=device,
                num_simulations=30)
        
        # generate syndrome with error probability 0.1 
        prediction_list_p_error = [i]
    
        error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error, Avarage_solve_time, avarage_nr_steps = rl.prediction(
            num_of_predictions=num_of_predictions, 
            num_of_steps=200, 
            PATH=PATH2, 
            prediction_list_p_error=prediction_list_p_error,
            plot_one_episode=False)
        
        win_rate = (num_of_predictions - len(failed_syndroms)/2) / num_of_predictions
        
        # runtime of prediction
        runtime = time.time()-start
        runtime = runtime
        
        print(win_rate, 'win_rate')
        print(error_corrected_list, 'error corrected')
        print(ground_state_list, 'ground state conserved')
        print(avarage_nr_steps, 'average number of steps')
        print(Avarage_solve_time, 's Avarage_solve_time')
        
          
        # save training settings in txt file 
        data_all = np.append(data_all, np.array([[NETWORK_FILE_NAME, num_of_predictions, error_corrected_list, ground_state_list,average_number_of_steps_list, len(failed_syndroms)/2, win_rate, runtime]]), axis=0)
        np.savetxt(PATH + '/data_all.txt', data_all, header='network, error corrected, ground state conserved, average number of steps, number of failed syndroms, win_rate, runtime (h)', delimiter=',', fmt="%s")
        
         # save training results in txt file 
        data_result = np.append(data_result, np.array([[prediction_list_p_error[0], win_rate]]), axis=0)
        np.savetxt(PATH + '/data_result.txt', data_result, delimiter=',', fmt="%f")
        
        # save training results in txt file 
        data_steps = np.append(data_steps, np.array([[prediction_list_p_error[0], Avarage_solve_time, avarage_nr_steps]]), axis=0)
        np.savetxt(PATH + '/data_steps.txt', data_steps, delimiter=',', fmt="%f")

############################################################################################

num_of_predictions = 100
p_e = [0.1]
system_size = 5
network = NN_11
# this file is stored in the network folder and contains the trained agent that will guide the MCTS  
NETWORK_FILE_NAME = 'Size_5_NN_11'
predict_MCTS_func(system_size, NETWORK_FILE_NAME, network, num_of_predictions, p_e)



