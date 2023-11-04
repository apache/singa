##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
from search_model_enas_utils import Controller


def main():
    controller = Controller(6, 4)
    predictions = controller()


if __name__ == "__main__":
    main()
