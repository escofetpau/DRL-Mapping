# DRL-Mapping


## Project structure
```
DRL-Mapping/
│
├── src/                        # Source code
│   ├── circuit_generator/                  # Scripts to generate circuits
│   │   ├── random_circuit_generator.py     # Script to generate random circuits
│   │   └── synthetic_circuit_generator.py  # TODO 
│   │
│   ├── environment/            # 
│   │   ├── base_env.py         # Basic environment functionalities
│   │   ├── env_place_pair.py   # Environment where the pair of a qubit is placed together
│   │   └── utils.py            # Helper functions                      
│   │
│   ├── models/
│   │   ├── feature_extractor.py    # Graph Neural Network
│   │   └── ppo_policy.py           # Deep Reinforcement Learning model
│   │
│   └── utils/
│       ├── callback.py    # Metric logging and early stopping logic
│       └── trainer.py     # Training logic
│ 
├── tests/                          # Automated tests
│   ├── environment/                   
│   │   ├── test_base_env.py        # Tests
│   │   └── test_utils.py           # Tests
│   └── models/
│       └── test_feature_extractor.py
│
├── main.py                         # CLI run                
│
├── config.yaml                     # Experiment configuration             
│
├── .gitignore                      # Untracked files to ignore                
│
├── README.md                       # Guide and instructions for the project             
│
├── poetry.lock                     # Dependencies
│
└── pyproject.toml                  # Project configuration
```

## Usage

In a venv/conda environment with poetry installed, run the following command to install project dependencies:

```bash
poetry install
```

Do a single run using the gates_per_slice stored in level1:

```bash
poetry run python main.py
```

Automatically run curriculum learning:
```bash
poetry run python main.py --curriculum
```

Display tensorboard plots:
```bash
poetry run tensorboard --logdir=runs
```