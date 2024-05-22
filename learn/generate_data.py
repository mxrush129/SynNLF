import numpy as np
import torch
from utils.Config_V import CegisConfig


def x2dotx(X, f):  # todo Changing to map mapping will be faster
    f_x = []
    n = len(f)
    for x in X:
        f_x.append([f[i](x) for i in range(n)])
    return torch.Tensor(f_x)


class Sampling_data:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.inv = config.EXAMPLE.D_zones
        self.n = config.EXAMPLE.n
        self.batch_size = config.BATCH_SIZE
        self.f = config.EXAMPLE.f

    def generate_data(self):
        global s
        zone = self.inv
        if zone.shape == 'box':
            times = 1 / (1 - self.config.R_b)
            s = np.clip((np.random.rand(self.batch_size, self.n) - 0.5) * times, -0.5, 0.5)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center
        elif zone.shape == 'ball':
            s = np.random.randn(self.batch_size, self.n)
            s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) for e in s])
            s = np.array(
                [e * np.random.random() ** (1 / self.n) if np.random.random() > self.config.C_b else e for e in s])
            s = s + zone.center

        s = torch.Tensor(s)
        inv_dot = x2dotx(s, self.f)
        return s, inv_dot
