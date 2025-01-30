import torch
import numpy as np
from benchmarks.Exampler_V import Example


class CegisConfig():
    N_HIDDEN_NEURONS = [10]
    EXAMPLE = None
    ACTIVATION = ['SKIP']
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01
    LOSS_WEIGHT = (1, 1)
    BIAS = False
    SPLIT_D = False
    MARGIN = 0.
    DEG = [2, 2, 2]
    OPT = torch.optim.AdamW
    C_b = 0.2
    R_b = 0.5
    eps = 0.05
    counter_nums = 20
    learn_loops = 100
    max_iter = 10
    beta = 10**6
    ellipsoid = False
    loss_optimization = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
