from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
from benchmarks.Exampler_V import get_example_by_name

# import sys, os
# file_name = os.path.basename(__file__).split(".")[0]
# path = './results/'
# full_path = path + file_name
# sys.stdout = open(full_path, 'w')

def main():
    activations = ['SKIP']
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('C14')
    start = timeit.default_timer()
    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "BATCH_SIZE": 1000,
        "LEARNING_RATE": 0.5,
        "LOSS_WEIGHT": (1.0, 1),
        "SPLIT_D": False,
        'BIAS': False,
        'DEG': [2, 2, 0],
        'max_iter': 20,
        # 'C_b': 0.2,
        'counter_nums': 100,
        'ellipsoid': False,
        'x0': [10] * example.n,
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
        draw.plot_benchmark_2d()
        # draw.plot_benchmark_3d()


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
