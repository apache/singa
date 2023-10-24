from enum import Enum, auto


class Metric(Enum):
    RAW = auto()
    ALL = auto()

    TRAIN_ACCURACY = auto()
    VAL_ACCURACY = auto()
    TEST_ACCURACY = auto()

    TRAIN_LOSS = auto()
    VAL_LOSS = auto()
    TEST_LOSS = auto()

    TRAIN_TIME = auto()
    VAL_TIME = auto()
    TEST_TIME = auto()

    FLOPS = auto()
    LATENCY = auto()
    PARAMETERS = auto()
    EPOCH = auto()
    HP = auto()




