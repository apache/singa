import os
from src.common.constant import Config
from src.search_space.core.space import SpaceWrapper
from src.query_api.query_api_img import ImgScoreQueryApi

def init_search_space(args) -> SpaceWrapper:
    """
    :param args:
    :param loapi: Local score API, records all scored arch, 101 use it to detect which arch is scored.
    :return:
    """
    # elif args.search_space == Config.MLPSP:
    if args.search_space == Config.MLPSP:
        from .mlp_api.space import MlpSpace
        from .mlp_api.model_params import MlpMacroCfg
        from .mlp_api.space import DEFAULT_LAYER_CHOICES_20, DEFAULT_LAYER_CHOICES_10
        print ("src/search_space/init_search_space.py config.MLPSP")
        if args.hidden_choice_len == 10:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_10)
        else:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_20)

        return MlpSpace(model_cfg)
    else:
        raise Exception
