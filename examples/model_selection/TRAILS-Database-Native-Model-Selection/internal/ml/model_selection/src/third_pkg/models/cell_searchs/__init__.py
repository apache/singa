##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts import TinyNetworkDarts
from .search_model_gdas import TinyNetworkGDAS
from .search_model_setn import TinyNetworkSETN
from .search_model_enas import TinyNetworkENAS
from .search_model_random import TinyNetworkRANDOM
from .generic_model import GenericNAS201Model
from .genotypes import Structure as CellStructure, architectures as CellArchitectures

# NASNet-based macro structure
from .search_model_gdas_nasnet import NASNetworkGDAS
from .search_model_gdas_frc_nasnet import NASNetworkGDAS_FRC
from .search_model_darts_nasnet import NASNetworkDARTS


nas201_super_nets = {
    "DARTS-V1": TinyNetworkDarts,
    "DARTS-V2": TinyNetworkDarts,
    "GDAS": TinyNetworkGDAS,
    "SETN": TinyNetworkSETN,
    "ENAS": TinyNetworkENAS,
    "RANDOM": TinyNetworkRANDOM,
    "generic": GenericNAS201Model,
}

nasnet_super_nets = {
    "GDAS": NASNetworkGDAS,
    "GDAS_FRC": NASNetworkGDAS_FRC,
    "DARTS": NASNetworkDARTS,
}
