from src.common.constant import *
from src.eva_engine.phase1.algo.prune_synflow import SynFlowEvaluator

# evaluator mapper to register many existing evaluation algorithms
evaluator_register = {

    # prune based
    CommonVars.PRUNE_SYNFLOW: SynFlowEvaluator(),

}
