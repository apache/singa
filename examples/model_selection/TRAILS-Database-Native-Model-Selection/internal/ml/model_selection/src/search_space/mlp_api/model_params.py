from src.search_space.core.model_params import ModelMacroCfg


class MlpMacroCfg(ModelMacroCfg):

    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 num_layers: int,
                 num_labels: int,
                 layer_choices: list):
        super().__init__(num_labels)

        self.nfield = nfield
        self.nfeat = nfeat
        self.nemb = nemb
        self.layer_choices = layer_choices
        self.num_layers = num_layers
