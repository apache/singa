class ModelMacroCfg:
    """
    Macro search space config
    Search Space basic init,  use bn or not, input features, output labels, etc. 
    """

    def __init__(self, num_labels):
        """
        Args:
            num_labels: output labels.
        """
        self.num_labels = num_labels


class ModelMicroCfg:
    """
    Micro space cfg
    Identifier for each model, connection patter, operations etc.
    encoding = serialized(ModelMicroCfg)
    """

    def __init__(self):
        pass
