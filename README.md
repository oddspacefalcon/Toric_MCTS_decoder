# MCTS-avkodare och RL-avkodare på torisk kod

Djup Q-inlärnings avkodare och MCTS avkodare på torisk kod

Simon Sundelin, Marcus Remgård, Christian Nilsson, Mikkel Opperud, Joel Erikanders, Joel Harf Abili

![](docs/visual/toric_code_gif.gif)


## Krav 
- Python 3
 
### Installation 
- Biblioteken som användes var matplotlib, numpy and pytorch

```bash
pip install -r requirements.txt
```

- Klona repot:
```bash
git clone https://github.com/mats-granath/toric-RL-decoder.git
```

## Hur används simmulatorn?
Där finns tre exempel scripts
- train_script.py
- prediction_script_network.py
- prediction_script_MCTS.py

train_script tränar en agent för att lösa syndrom. Alla hyperparametrar relaterade till träningen anges i skriptet. Dessutom lagras en utvärdering av träningskörningen i data mappen.

prediction_script_network använder ett utbildat nätverk och förutsäger en viss mängd syndrom. Det utbildade nätverket kan laddas från nätverksmappen.

prediction_script_MCTS använder ett utbildat nätverk för att guida en trädsökning och förutsäger en viss mängd syndrom. Det styrande nätverket kan laddas från nätverksmappen


## Repots struktur

Fil | Beskrivning
----- | -----
`├── data` | En mapp som innehåller utvärderingsdata för varje tränings- eller förutsägelseskörning.
`├── network` | Förtränade modeller för strolek 5,7, 9 och 11.
`├── plots` | Några of graferna som genererats under förutsägelseskörning.
`├── src` | Source filer för MCTS, nätverken och den toriska koden.
`·   ├── RL.py` | Träning och förutsägelse av nätverk
`·   ├── predict_MCTS.py` | förutsägelseskript som använder MCTS som avkodare
`·   ├── MCTS.py` | MCTS avkodare
`·   ├── Replay_memory.py` | Innehåller klasser för ett reprisminne och proportionell sampling.
`·   ├── Sum_tree.py` | En binär träddatastruktur där "förälderns" värde är summan av sina barn
`·   └── Toric_model.py` | Innehåller klassen toric_code och relevanta funktioner.
`├── NN.py` | Innehåller olika nätverksarkitekturer
`├── README.md` | Om projektet
`├── ResNet.py` | Innehåller olika ResNet-arkitekturer
`├── train_script` | ett skript som tränar en agent för att lösa den toriska koden.
`└── predict_script_network` | Den tränade agenten löser syndromen
`└── prediction_script_MCTS` | MCTS löser syndromen

