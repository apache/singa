from src.controller.core.sample import Sampler
from src.search_space.core.space import SpaceWrapper
from src.search_space.core.model_params import ModelMicroCfg


class RandomSampler(Sampler):

    def __init__(self, space: SpaceWrapper):
        super().__init__(space)
        self.visited = []

    def sample_next_arch(self, sorted_model: list = None) -> (str, ModelMicroCfg):
        while True:
            arch_id, model_micro = self.space.random_architecture_id()

            if arch_id not in self.visited:
                self.visited.append(arch_id)
                return str(arch_id), model_micro

    def fit_sampler(self, score: float):
        # random sampler can skip this.
        pass
