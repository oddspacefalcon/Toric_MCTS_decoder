from src.toric_model import Toric_code
from ResNet import ResNet18
from src.MCTS import MCTS
from src.util import Perspective, Action
import torch
import torch.nn as nn
import numpy as np
from src.util import convert_from_np_to_tensor
import time
import copy
import random
import matplotlib.pyplot as plt


# plot_range = 10
# P_error =[0.05, 0.06]
# P_success = [0.5, 0.6]


# fig, ax = plt.subplots()
# ax.scatter(P_error, P_success, label='d=5', color='blue', marker='o')
# ax.legend(fontsize = 13)
# ax.plot(P_error, P_success, color='blue')
# ax.set_xlim(0.005, plot_range*0.01 + 0.005)
# plt.xlabel('$P_e$', fontsize=16)
# plt.ylabel('$P_s$', fontsize=16)

# plt.tick_params(axis='both', labelsize=15)
# #fig.set_figwidth(10)
# plt.savefig('plots/plt1' '.png')
# plt.show()



# with open('bajs', '+w') as f:

#     l = [1, 2, 3]
#     l2 = [1, 2, 3]

#     f.write(str(l) + ',' + str(l2))
#     f.write(str(l2))



