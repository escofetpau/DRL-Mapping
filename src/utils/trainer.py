import os
import yaml
import torch
import re

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from src.models.ppo_policy import CustomPPOPolicy
from src.environment.env_place_pair import GraphSeriesEnvPlacePair
from src.utils.callback import CustomTensorboardCallback


class Trainer:
    def __init__(self, config_path):
        self.config = self.get_config(config_path)
        self.env = self.get_env()
        self.initialize_model()

    def get_callback(self, level: str = '') -> CustomTensorboardCallback:
        config = self.config["callback"]
        return CustomTensorboardCallback(
            experiment_name=self.config["experiment_name"],
            save_path=config["tensorboard_path"],
            save_freq=config["save_freq"],
            early_stopping=config["early_stopping"],
            delta=config["delta"],
            patience=config["patience"],
            verbose=1,
            level=level,
        )

    def get_config(self, config_path: str) -> dict:

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if config['experiment_name'] != 'prova' and self.find_largest_matching_file('config_files', f'{config['experiment_name']}_config') is not None:
            raise ValueError(f'An experiment with name {config['experiment_name']} already exists. Please change the experiment name in the config file.')

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        config["policy_kwargs"]["features_extractor_kwargs"]["n_cores"] = config[
            "n_cores"
        ]
        config["policy_kwargs"]["features_extractor_kwargs"]["device"] = config[
            "device"
        ]
        config["policy_kwargs"]["features_extractor_kwargs"]["action_type"] = config[
            "action_type"
        ]
        config["policy_kwargs"]["features_extractor_kwargs"]["n_qubits"] = config["n_qubits"]
        return config

    def get_env(self) -> DummyVecEnv:
        self.config['circuit']['gates_per_slice'] = self.config['levels']['level1']

        def make_env():
            env = GraphSeriesEnvPlacePair(
                circuit_config=self.config["circuit"],
                action_type=self.config["action_type"],
                n_qubits=self.config["n_qubits"],
                n_cores=self.config["n_cores"],
                weights_reward=self.config["weights_reward"],
            )
            env = ActionMasker(env, lambda e: e.env_mask())
            # env = Monitor(env)
            return env

        return DummyVecEnv([make_env])

    def initialize_model(self) -> None:
        '''Initialize the model and save the configuration to a YAML file'''
        self.model = MaskablePPO(
            policy=CustomPPOPolicy,
            policy_kwargs=self.config["policy_kwargs"],
            env=self.env,
            device=self.config["device"],
            verbose=1,
            seed=42,
            tensorboard_log="runs",
            **self.config["ppo"],
        )
        save_path = f'config_files/{self.config['experiment_name']}_config.yaml'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def load_model(self, model_path) -> None:
        self.model = MaskablePPO.load(model_path, self.env, device=self.config["device"])
        print(f"Model loaded from {model_path}")

    def fit(self) -> None:
        run_id = "prova"
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.get_callback(),
            reset_num_timesteps=False,
            tb_log_name=run_id,
        )


    def find_largest_matching_file(self, directory: str, prefix: str) -> str | None:
        matching_files = []
        
        os.makedirs(directory, exist_ok=True)

        for f in os.listdir(directory):
            if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f)):
                # Extraer la parte después del prefix + "_" y antes de la extensión
                try:
                    suffix = f[len(prefix) + 1:]  # skip prefix and underscore
                    number_str = suffix.split('_')[0]  # toma solo el número
                    steps = int(number_str)
                    matching_files.append((steps, f))
                except (IndexError, ValueError):
                    continue  # ignora archivos que no sigan la convención

        if not matching_files:
            return None

        # Elige el archivo con mayor número de steps
        _, filename = max(matching_files, key=lambda x: x[0])
        return os.path.join(directory, filename)
    

    def curriculum_learn(self) -> None:
        # level 1
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.get_callback(level='level1'),
            reset_num_timesteps=False,
            tb_log_name = f'{self.config['experiment_name']}_level1',
        )

        for level, gates_per_slice in self.config['levels'].items():
            if level == 'level1':
                continue
            
            prefix = f"{self.config['experiment_name']}_level{str(int(level.removeprefix('level'))-1)}_model"
            model_path = self.find_largest_matching_file(self.config['callback']['tensorboard_path'], prefix)
            
            if model_path is None:
                raise FileNotFoundError(f"No matching files found for prefix {prefix} in directory {self.config['callback']['tensorboard_path']}")
            
            self.load_model(model_path)
            # change parameter env
            self.model.env.envs[0].env.gates_per_slice = gates_per_slice
            self.model.env.envs[0].env.reset()

            self.model.learn(
                total_timesteps=self.config["total_timesteps"],
                callback=self.get_callback(level=level),
                reset_num_timesteps=False,
                tb_log_name = f'{self.config['experiment_name']}_{level}',
            )
