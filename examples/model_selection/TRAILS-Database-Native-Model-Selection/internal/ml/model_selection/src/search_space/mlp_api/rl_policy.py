from src.search_space.core.rl_policy import RLPolicyBase


class RLMlpSPaceTopology(RLPolicyBase):
    def __init__(self, search_space, rl_learning_rate, max_nodes=4):
        super().__init__()

    def generate_arch(self, config):
        pass

    def select_action(self):
        pass

    def _sample_new_cfg(self):
        pass

    def update_policy(self, reward, baseline_values, log_prob):
        pass
