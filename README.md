# toric-RL-decoder and MCTS-decoder

Deep Q-learning decoder and MCTS decoder for depolarizing noise on the toric code

Simon Sundelin, Marcus Remgård, Christian Nilsson, Mikkel Opperud, Joel Erikanders, Joel Harf Abili

![](docs/visual/toric_code_gif.gif)




Deep reinforcement learning decoder for the toric code in pytorch. 

## Prerequisites 
- Python 3

## Getting started 
### Installation 
- The required libraries are matplotlib, numpy and pytorch

```bash
pip install -r requirements.txt
```

- Clone this repo:
```bash
git clone https://github.com/mats-granath/toric-RL-decoder.git
```

## How to use the simulator
There are three example scripts
- train_script.py
- prediction_script_network.py
- prediction_script_MCTS.py

The train script trains an agent to solve syndromes. All the hyperparameters related to the training are specified in the script. Moreover, an evaluation of the training run is stored in the data folder with a timestamp.

The predict script uses a trained network and predicts given a specified amount of syndromes. The trained network can be loaded from the network folder.

The predict MCTS script uses a trained network to guide a tree search and predicts given a specified amount of syndromes. The guiding network can be loaded from the network folder


## Structure of this repo

File | Description
----- | -----
`├── data` | A directory that contains the evaluation data for each training or prediction run.
`├── network` | Pretrained models for the grid sizes 5,7, and 9.
`├── plots` | Some of the plots of the toric code generated during prediction are saved in that folder.
`├── src` | Source files for the agent and toric code.
`·   ├── RL.py` | Training and predicition of network occurs here
`·   ├── predict_MCTS.py` | Prediction using MCTS decoder
`·   ├── MCTS.py` | MCTS decoder
`·   ├── Replay_memory.py` | Contains classes for a replay memory with uniform and proportional sampling. 
`·   ├── Sum_tree.py` | A binary tree data structure where the parent’s value is the sum of its children.
`·   └── Toric_model.py` | Contains the class toric_code and functions that are relevant to manipulate the grid.
`├── NN.py` | Contains different network architectures
`├── README.md` | About the project.
`├── ResNet.py` | Contains different ResNet architectures
`├── requirements.txt` | The Python libraries that will be installed. Only the libraries in this official repo will be available.
`├── train_script` | A script training an agent to solve the toric code.
`└── predict_script_network` | The trained agent solves syndromes.
`└── prediction_script_MCTS` | MCTS decoder solves syndromes.

