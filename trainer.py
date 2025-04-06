import yaml
import torch
import wandb
import time
import os

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from src.models.ppo_policy import CustomPPOPolicy
from src.environment.env_place_pair import GraphSeriesEnvPlacePair
from src.utils.callback import CustomTensorboardCallback
from src.utils.constants import N_CORES


class Trainer:
    def __init__(self, config_path):
        self.config = self.get_config(config_path)
        self.env = self.get_env()
        self.initialize_model()

    def get_callback(self, level = ''):
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

    def get_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        config["policy_kwargs"]["features_extractor_kwargs"]["device"] = config[
            "device"
        ]
        config["policy_kwargs"]["features_extractor_kwargs"]["action_type"] = config[
            "action_type"
        ]
        config["policy_kwargs"]["features_extractor_kwargs"]["n_qbits"] = config[
            "circuit"
        ]["n_qbits"]
        return config

    def get_env(self):
        self.config['circuit']['gates_per_slice'] = self.config['levels']['level1']

        def make_env():
            env = GraphSeriesEnvPlacePair(
                circuit_config=self.config["circuit"],
                action_type=self.config["action_type"],
                weights_reward=self.config["weights_reward"],
            )
            env = ActionMasker(env, lambda e: e.env_mask())
            # env = Monitor(env)
            return env

        return DummyVecEnv([make_env])

    def initialize_model(self):

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

    def load_model(self, model_path):
        self.model = MaskablePPO.load(model_path, self.env, device=self.config["device"])
        print(f"Model loaded from {model_path}")

    def fit(self):
        run_id = "prova"
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.get_callback(),
            reset_num_timesteps=False,
            tb_log_name=run_id,
        )


    def find_largest_matching_file(self, model_dir, experiment_name, level):
        prefix = f"{experiment_name}_{level}"
        matching_files = [
            f for f in os.listdir(model_dir)
            if f.startswith(prefix) and os.path.isfile(os.path.join(model_dir, f))
        ]

        if not matching_files:
            raise FileNotFoundError(f"No matching files found for prefix: {prefix}")

        return os.path.join(model_dir, max(matching_files))


    def curriculum_learn(self):
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

            model_path = self.find_largest_matching_file(self.config['callback']['tensorboard_path'], self.config['experiment_name'], f'level{str(int(level[-1])-1)}')
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

