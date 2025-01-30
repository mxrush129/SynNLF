from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
from benchmarks.Exampler_V import get_example_by_name
from matplotlib import pyplot as plt
from plots.plot import Draw
from verify.CounterExampleFind_V import CounterExampleFinder


def main():
    activations = ['MUL']
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('C1')
    example.D_zones.r = pow(0.3, 2)

    start = timeit.default_timer()
    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "BATCH_SIZE": 500,
        "LEARNING_RATE": 0.01,
        "LOSS_WEIGHT": (1.0, 1.0),
        "SPLIT_D": False,
        'BIAS': False,
        'DEG': [2, 2, 0],
        'max_iter': 50,
        # 'C_b' : 0.2,
        'counter_nums': 100,
        'ellipsoid': True,
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
        # draw.plot_benchmark_3d(points=None)
        levels = np.linspace(0, 0.001, 10)  # 设置等高线的范围和步长
        points = [[0, 0, 0]]
        draw.plot_benchmark_3d(points=points, levels=None)
        # draw.plot_benchmark_2d(levels=[0],color='r',show=True)
        # draw.plot_point((-0.01042682, 0.0994549),color='b')

        plt.show()




if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
