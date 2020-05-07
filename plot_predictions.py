import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

t0 = time.time()

def get_results(system_size, NETWORK_FILE_NAME, network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_of_predictions = 3000

    # initialize RL class
    rl = RL(Network=network,
            Network_name=NETWORK_FILE_NAME,
            system_size=system_size,
            device=device)

    # Generate folder structure, all results are stored in the data folder 
    timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
    PATH = f'data/d_{system_size}/results__' + str(NETWORK_FILE_NAME) + '__' +  timestamp
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Path for the network to use for the prediction
    PATH2 = 'network/' + str(NETWORK_FILE_NAME) + '.pt'

    win_rates = []

    with open(PATH + '/data_all', 'w+') as f:

        f.write('network, p_error, num_of_predictions, error corrected, ground state conserved, average number of steps, number of failed syndroms, win_rate')

        p_errors = np.arange(0.05, 0.17, 0.02)

        for error in p_errors:

            f.write('\n')
            
            error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = rl.prediction(
                num_of_predictions=num_of_predictions, 
                num_of_steps=50, 
                PATH=PATH2, 
                prediction_list_p_error=[error],
                plot_one_episode=False)

            win_rate = (num_of_predictions - len(failed_syndroms)/2) / num_of_predictions
            win_rates.append(win_rate)

            for result in [NETWORK_FILE_NAME, error, num_of_predictions, error_corrected_list[0], ground_state_list[0],average_number_of_steps_list[0], len(failed_syndroms)/2, win_rate]:
                f.write(str(result) + ', ')

    return p_errors, win_rates

nets = [(5, 'size_5_size_5_NN_11_epoch_79', NN_11), (7, 'size_7_size_7_size_7_NN_11_epoch_178_epoch_21', NN_11), (9, 'size_9_size_9_NN_11_epoch_279', NN_11), (11, 'size_11_size_11_NN_11_epoch_207', NN_11)]

results_nets = [get_results(*net) for net in nets]

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(results_nets[0][0], results_nets[0][1], s=100, label='d = 5', color='steelblue', marker='o')
ax.scatter(results_nets[1][0], results_nets[1][1], s=100, label='d = 7', color='green', marker='D')
ax.scatter(results_nets[2][0], results_nets[2][1], s=100, label='d = 9', color='orange', marker='X')
ax.scatter(results_nets[3][0], results_nets[3][1], s=100, label='d = 11', color='firebrick', marker='^')

# ax.scatter(P_error11, P_success11,s=100, label='d = '+str(system_size11), color='firebrick', marker='^')
# ax.scatter(P_error13, P_success13,s=100, label='d = '+str(system_size13), color='saddlebrown', marker='s')
ax.legend(fontsize=14)
ax.plot(results_nets[0][0], results_nets[0][1], color='steelblue')
ax.plot(results_nets[1][0], results_nets[1][1], color='green')
ax.plot(results_nets[2][0], results_nets[2][1], color='orange')
ax.plot(results_nets[3][0], results_nets[3][1], color='firebrick')
# ax.plot(P_error11,P_success11, color='firebrick')
# ax.plot(P_error13,P_success13, color='saddlebrown')
#ax.set_xlim(0.005, 0.205)
plt.xlabel('$P_e$', fontsize=20)
plt.ylabel('$P_s$', fontsize=20)
plt.title('Prestanda för tränade agenter', fontsize=20)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('plots/results_' + time.strftime("%y_%m_%d__%H_%M_%S__") + '.png')
plt.show()

print('tid:', time.time() - t0, 's')