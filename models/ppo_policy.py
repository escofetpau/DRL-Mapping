import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
#from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomPPOPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_class, net_arch = [64, 64, 64], features_extractor_kwargs={}, **kwargs):
        
        print(kwargs)
        super().__init__(observation_space, action_space, lr_schedule, 
                         net_arch = net_arch,
                         features_extractor_class=features_extractor_class, 
                         features_extractor_kwargs=features_extractor_kwargs)
        '''
        self.actor_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n),
        ).to(device)

        self.critic_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)



    def forward(self, obs):
        with torch.enable_grad():

            features = self.extract_features(obs)
            
        
            #Output actor
            logits = self.actor_net(features)
            #Normalització del output
            action_probs = torch.softmax(logits, dim=-1)
            print(action_probs)
            #Output critic
            values = self.critic_net(features)

            #Sampleja una acció
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
            return actions, values.detach(), log_probs.detach()
'''
    
    def _predict(self, obs, deterministic = True):
        #És el que es crida durant el procés de recopilació d'observacions (quant els batchs son 1), en comptes de forward.
        #És el mateix però pot ser deterministic i només entra en joc l'actor
        observation = observation.to(device)
        features = self.extract_features(observation)
        logits = self.actor_net(features)
        action_probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action
    

    
