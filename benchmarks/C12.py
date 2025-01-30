from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
import random
from benchmarks.Exampler_V import get_example_by_name
from matplotlib import pyplot as plt
import pickle
# import sys, os
# file_name = os.path.basename(__file__).split(".")[0]
# path = './results/'
# full_path = path + file_name
# sys.stdout = open(full_path, 'w')

def main():
    activations = ['SKIP']
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('C12')
    example.D_zones.r = pow(1, 2)
    start = timeit.default_timer()

    # example.D_zones.r = pow(1.2, 2)

    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "BATCH_SIZE": 200,
        "LEARNING_RATE": 0.001,
        "LOSS_WEIGHT": (1.0, 1.0),
        "SPLIT_D": False,
        'BIAS': False,
        'DEG': [4, 2, 0],
        'max_iter': 20,
        'counter_nums': 50,
        'ellipsoid': True,
        'x0': [5] * example.n,
        'loss_optimization': False,
    }
    Config = CegisConfig(**opts)
    c = Cegis(Config)
    c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))
    if example.n == 2:
        from plots.plot import Draw
        draw = Draw(c.ex, c.Learner.net.get_lyapunov())
        # draw.plot_benchmark_2d()

        draw.plot_benchmark_3d(points=None)
        plt.show()

if __name__ == '__main__':

    torch.manual_seed(2024)
    np.random.seed(2024)

    # a = random.sample(range(1000, 10000), 1)
    # # print(a)
    # torch.manual_seed(a[0])
    # np.random.seed(a[0])



    main()
