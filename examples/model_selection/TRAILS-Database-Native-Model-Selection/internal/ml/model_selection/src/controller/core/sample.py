from abc import abstractmethod

from src.search_space.core.model_params import ModelMicroCfg
from src.search_space.core.space import SpaceWrapper


class Sampler:

    def __init__(self, space: SpaceWrapper):
        self.space = space

    @abstractmethod
    def sample_next_arch(self, sorted_model: list) -> (str, ModelMicroCfg):
        """
        Sample next architecture,
        :param sorted_model: the scoted model,
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def fit_sampler(self, score: float):
        """
        Fit the sampler with architecture's score.
        :param score:
        :return:
        """
        raise NotImplementedError
