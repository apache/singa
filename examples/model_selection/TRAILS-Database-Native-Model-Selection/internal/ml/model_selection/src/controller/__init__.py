

from src.common.constant import CommonVars
from src.controller.sampler_ea.regularized_ea import RegularizedEASampler
from src.controller.sampler_all.seq_sampler import SequenceSampler
from src.controller.sampler_rl.reinforcement_learning import RLSampler
from src.controller.sampler_rand.random_sample import RandomSampler
from src.controller.sampler_all.seq_sampler import SequenceSampler

sampler_register = {
    CommonVars.TEST_SAMPLER: SequenceSampler,
    # CommonVars.RANDOM_SAMPLER: RandomSampler,
    CommonVars.RANDOM_SAMPLER: SequenceSampler,
    CommonVars.RL_SAMPLER: RLSampler,
    CommonVars.EA_SAMPLER: RegularizedEASampler,
}

