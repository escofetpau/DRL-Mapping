import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, save_path, save_freq=100000, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.verbose = verbose
        self.episode_rewards = []
        self.nl_comm_sums = 0
        self.intervention_sums = 0
        self.direct_capacity_violation_sums = 0
        self.missing_space_for_interaction_violation_sums = 0
        self.no_space_for_future_gates_violation_sums = 0


    def _on_step(self) -> bool:
        # Retrieve the reward for the current step
        reward = self.locals.get("rewards")
        if reward is not None:
            self.episode_rewards.append(reward)

        # Access the current environment
        #env = self.training_env.envs[0]  # Assuming a single environment is used
        env = self.training_env
        # Accumulate the relevant environment variables
        self.nl_comm_sums += env.get_attr('nl_com')[0]
        self.intervention_sums += env.get_attr('intervention')[0]
        self.direct_capacity_violation_sums += env.get_attr('direct_capacity_violation')[0]
        self.missing_space_for_interaction_violation_sums += env.get_attr('missing_space_for_interaction_violation')[0]
        self.no_space_for_future_gates_violation_sums += env.get_attr('no_space_for_future_gates_violation')[0]

        # Check if the episode is done
        dones = self.locals.get("dones")
        if dones is not None and any(dones):
            # Compute episode statistics
            episode_reward = np.sum(self.episode_rewards)

            # Log statistics to TensorBoard
            self.logger.record("episode/final_reward", episode_reward)
            self.logger.record("episode/nl_comm_sum", self.nl_comm_sums)
            self.logger.record("episode/intervention_sum", self.intervention_sums)
            self.logger.record("episode/direct_capacity_violation_sum", self.direct_capacity_violation_sums)
            self.logger.record("episode/missing_space_for_interaction_violation_sum", self.missing_space_for_interaction_violation_sums)
            self.logger.record("episode/no_space_for_future_gates_violation", self.no_space_for_future_gates_violation_sums)

            self.logger.dump(self.num_timesteps)

            # Reset buffers for the next episode
            self.episode_rewards = []
            self.nl_comm_sums = 0
            self.intervention_sums = 0
            self.direct_capacity_violation_sums = 0
    
        # Save the model periodically
        # Save the model periodically
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/model_{self.n_calls}_steps.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at {model_path}")
        return True
    
