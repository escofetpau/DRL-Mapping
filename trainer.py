import yaml
import torch
import wandb
import time

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from src.models.ppo_policy import CustomPPOPolicy
from src.environment.env_place_pair import GraphSeriesEnvPlacePair
from src.utils.callback import CustomTensorboardCallback



class Trainer():
    def __init__(self, config_path):
        self.config = self.get_config(config_path)
        self.env = self.get_env()
        self.initialize_model()
        self.callback = CustomTensorboardCallback(save_path=self.config['tensorboard_path'], verbose=1)


    def get_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if config['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available')

        config['policy_kwargs']['device'] = config['device']
        config['policy_kwargs']['features_extractor_kwargs']['device'] = config['device']

        return config


    def get_env(self):
        def make_env():
            env = GraphSeriesEnvPlacePair(circuit_config=self.config['circuit'], action_type=self.config['action_type'], weights_reward=self.config['weights_reward'])
            env = ActionMasker(env, lambda e: e.env_mask())
            #env = Monitor(env)
            return env

        return DummyVecEnv([make_env])


    def initialize_model(self):
            
        self.model = MaskablePPO(policy=CustomPPOPolicy,
                policy_kwargs=self.config['policy_kwargs'],
                env = self.env,
                device=self.config['device'],
                verbose = 1,
                seed = 42,
                tensorboard_log = 'runs',
                **self.config['ppo']
            )
        
    def load_model(self, model_path):
        self.model = MaskablePPO.load(model_path)

    def fit(self):
        run_id = 'prova'
        self.model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=self.callback,
            reset_num_timesteps = False,
            tb_log_name=run_id
        )

