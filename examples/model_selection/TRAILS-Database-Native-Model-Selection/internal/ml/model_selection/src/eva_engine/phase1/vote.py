from src.eva_engine.phase1.algo.alg_base import Evaluator
from .utils.autograd_hacks import *
from src.common.constant import Config

class VoteEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is simply sum over all weigth's norm to calculate models performance
        :param arch:
        :param device: CPU or GPU
        :param batch_data:
        :param batch_labels:
        :return:
        """

        pass


def vote_between_two_arch(arch1_info: dict, arch2_info: dict, metric: list, space: str):
    """
    Return which architecture is better,
    :param arch1_info:
    :param arch2_info:
    :param metric:
    :param space:
    :return:
    """
    left_vote = 0
    right_vote = 0
    for m_name in metric:
        # if this metrics vote to left
        if vote_to_left[space](m_name,
                               float(arch1_info["scores"][m_name]["score"]),
                               float(arch2_info["scores"][m_name]["score"])):
            left_vote += 1
        else:
            right_vote += 1

    if left_vote > right_vote:
        return arch1_info["architecture_id"]
    else:
        return arch2_info["architecture_id"]


def compare_score_201(m_name: str, s1: float, s2: float) -> bool:
    """
    Return if s1 is better than s2,
    :param m_name:
    :param s1:
    :param s2:
    :return: if s1 is better than s2
    """
    if m_name == "grad_norm":
        return s1 > s2
    if m_name == "grad_plain":
        return s1 < s2
    if m_name == "ntk_cond_num":
        return s1 < s2
    if m_name == "ntk_trace":
        return s1 > s2
    if m_name == "ntk_trace_approx":
        return s1 > s2
    if m_name == "fisher":
        return s1 > s2
    if m_name == "grasp":
        return s1 > s2
    if m_name == "snip":
        return s1 > s2
    if m_name == "synflow":
        return s1 > s2
    if m_name == "weight_norm":
        return s1 > s2
    if m_name == "nas_wot":
        return s1 > s2


def compare_score_101(m_name: str, s1: float, s2: float) -> bool:
    """
    Return if s1 is better than s2,
    :param m_name:
    :param s1:
    :param s2:
    :return: if s1 is better than s2
    """
    if m_name == "grad_norm":
        return s1 < s2
    if m_name == "grad_plain":
        return s1 < s2
    if m_name == "ntk_cond_num":
        return s1 < s2
    if m_name == "ntk_trace":
        return s1 < s2
    if m_name == "ntk_trace_approx":
        return s1 < s2
    if m_name == "fisher":
        return s1 < s2
    if m_name == "grasp":
        return s1 > s2
    if m_name == "snip":
        return s1 < s2
    if m_name == "synflow":
        return s1 > s2
    if m_name == "weight_norm":
        return s1 > s2
    if m_name == "nas_wot":
        return s1 > s2


vote_to_left = {}
vote_to_left["101"] = compare_score_101
vote_to_left["201"] = compare_score_201
