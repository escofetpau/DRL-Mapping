from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from src.models.feature_extractor import GNNFeatureExtractor


class CustomPPOPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_kwargs, net_arch):

        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=net_arch,
                         features_extractor_class=GNNFeatureExtractor,
                         features_extractor_kwargs=features_extractor_kwargs)
