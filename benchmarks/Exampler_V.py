import numpy as np


class Zone:
    def __init__(self, shape: str, low=None, up=None, center=None, r=None):
        self.shape = shape
        if shape == 'ball':
            self.center = np.array(center, dtype=np.float32)
            self.r = r  # radius squared
        elif shape == 'box':
            self.low = np.array(low, dtype=np.float32)
            self.up = np.array(up, dtype=np.float32)
            self.center = (self.low + self.up) / 2
        else:
            raise ValueError(f'There is no area of such shape!')


class Example:
    def __init__(self, n, D_zones, f, name):
        self.n = n  # number of variables
        self.D_zones = D_zones  # local condition
        self.f = f  # differential equation
        self.name = name  # name or identifier


examples = {
    1: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1 ** 2),
        f=[
            lambda x: -x[0],
            lambda x: -x[1]
        ],
        name='C1'
    ),
    2: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + x[0] * x[1],
            lambda x: -x[1]
        ],
        name='C2'
    ),
    3: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
        f=[
            lambda x: -2 * x[0] + x[0] * x[1],
            lambda x: -x[1] + x[0] * x[1],
        ],
        name='C3'
    ),
    4: Example(
        n=2,
        D_zones=Zone('ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + 2 * x[0]**2 * x[1],
            lambda x: -x[1]
        ],
        name='C4'
    ),
    5: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1300 ** 2),
        f=[
            lambda x: -x[0] ** 3 + x[1],
            lambda x: -x[0] - x[1],
        ],
        name='C5'
    ),
    6: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
        ],
        name='C6'
    ),
    7: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + x[1],
            lambda x: 0.1 * x[0] - 2 * x[1] - x[0] ** 2 + 0.1 * x[0] ** 3,
        ],
        name='C7'
    ),
    8: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=pow(1., 2)),
        # D_zones=Zone(shape='box', low=[-2]*2,up=[2]*2),
        f=[
            lambda x: -x[1],
            lambda x: x[0] - x[1] * (1 - x[0] ** 2),
        ],
        name='C8'
    ),

    9: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=pow(10, 2)),
        f=[
            lambda x: -x[1] - 3 / 2 * x[0] * x[1] - x[0] ** 3,
            lambda x: x[0] + 3 * x[0] ** 2 + 9 / 4 * x[0] ** 3 - x[1] ** 3
        ],
        name='C9'
    ),
    10: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=0.01** 2),
        f=[
            lambda x: -x[1],
            lambda x: -x[0] - 4 * x[1] + 0.25 * (x[1] - 0.5 * x[0]) * (x[1] - 2 * x[0]) * (x[1] + 2 * x[0]) * (
                        x[0] + x[1]),
        ],
        name='C10'
    ),
    11: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1 ** 2),
        f=[
            lambda x: -x[0] - 1.5 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 0.5 * x[0] ** 3 * x[1] ** 2
        ],
        name='C11'
    ),
    12: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=0.4 ** 2),
        f=[
            lambda x: -x[0] - 3 / 2 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 1 / 2 * (x[0] * x[1]) ** 2
        ],
        name='C12'
    ),
    13: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=0.5 ** 2),
        f=[
            lambda x: x[1] - x[0] ** 3 + x[0] * x[1] ** 4,
            lambda x: -x[0] ** 3 - x[1] ** 5
        ],
        name='C13'
    ),
    14: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=11 ** 2),
        f=[
            lambda x: -x[0],
            lambda x: -2 * x[1] + 0.1 * x[0] * x[1] ** 2 + x[2],
            lambda x: -x[2] - 1.5 * x[1]
        ],
        name='C14'
    ),
    15: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=0.5 ** 2),
        f=[
            lambda x: -x[0] ** 3 - x[0] * x[2] ** 2,
            lambda x: -x[1] - x[0] ** 2 * x[1],
            lambda x: -x[2] + 3 * x[0] ** 2 * x[2] - 3 * x[2]
        ],
        name='C15'
    ),
    16: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=1 ** 2),
        f=[
            lambda x: -3 * x[0] - 0.1 * x[0] * x[1] ** 3,
            lambda x: -x[1] + x[2],
            lambda x: -x[2]
        ],
        name='C16'
    ),
    17: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=0.01 ** 2),
        f=[
            lambda x: x[1] - x[0] ** 2 * x[2] ** 2 - x[0] ** 3,
            lambda x: -x[0] + x[2] ** 2 - x[1] ** 3,
            lambda x: -x[2] - x[1] * x[2]
        ],
        name='C17'
    ),
    18: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=1 ** 2),
        f=[
            lambda x: -x[0] + x[1] - x[0] * x[2] ** 2,
            lambda x: (x[1] * x[2]) ** 2 - x[0] ** 3 - x[1] ** 3,
            lambda x: -x[2] - x[1] ** 3 * x[2]
        ],
        name='C18'
    ),
    19: Example(
        n=4,
        D_zones=Zone(shape='ball', center=[0] * 4, r=0.5 ** 2),
        f=[
            lambda x: -x[0] ** 3 + x[0] ** 2 * x[2] * x[3],
            lambda x: -x[1] - 3 * x[2] + 2 * x[3] + (x[2] * x[3]) ** 2,
            lambda x: 3 * x[1] - x[2] - x[3],
            lambda x: -2 * x[1] + x[2] - x[3]
        ],
        name='C19'
    ),
    20: Example(
        n=6,
        D_zones=Zone(shape='ball', center=[0] * 6, r=0.01 ** 2),
        f=[
            lambda x: x[1] * x[3] - x[0] ** 3,
            lambda x: -3 * x[0] * x[3] - x[1] ** 3,
            lambda x: -x[2] - 3 * x[0] * x[3] ** 3,
            lambda x: -x[3] + x[0] * x[2],
            lambda x: -x[4] + x[5] ** 3,
            lambda x: -x[4] - x[5] + x[2] ** 4
        ],
        name='C20'
    ),
    # 21: Example(
    #     n=2,
    #     D_zones=Zone(),
    #     f=[
    #         lambda x:-x[0] + x[0]*x[1],
    #         lambda x:-x[1]
    #     ],
    #     name='D1'
    # ),
    22: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=0.3**2),
        f=[
            lambda x:-10*(x[0] - x[1]),
            lambda x:28*x[0] - x[1] - x[0]*x[2],
            lambda x:x[0]*x[1] - 8/3*x[2]
        ],
        name='D2'
    ),
    23: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=0.3**2),
        f=[
            lambda x:-2*x[1],
            lambda x:0.8*x[0] + 10 * (x[0]**2 - 0.21) * x[1]
        ],
        name='D3'
    ),
    24: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=0.5**2),
        f=[
            lambda x:x[0] * (x[0]**2 + x[1]**2 - 1) - x[1] * (x[2]**2 + 1),
            lambda x:x[1] * (x[0]**2 + x[1]**2 - 1) + x[0] * (x[2]**2 + 1),
            lambda x:10 * x[2] * (x[2]**2 - 1)
        ],
        name='D4'
    ),
    25:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=0.01**2),
        f=[
            lambda x:-0.42 * x[0] - 1.05 * x[1] - 2.3 * x[0] ** 2 - 0.5 * x[0] * x[1] - x[0] ** 3,
            lambda x:1.98 * x[0] + x[0] * x[1],
        ],
        name='D5'
    ),
    26:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1**2),
        f=[
            lambda x:-x[0] * (1 - x[0] * x[1]),
            lambda x:-x[1]
        ],
        name='D6'
    ),
    27:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=.01**2),
        f=[
            lambda x:-5 * x[1] - 4 * x[0] + x[0] * x[1] ** 2 - 6 * x[0] ** 3,
            lambda x:-20 * x[0] - 4 * x[1] + 4 * x[0] ** 2 * x[1] + x[1] ** 3
        ],
        name='D7'
    ),
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError(f'The example {name} was not found.')


if __name__ == '__main__':
    for idx, ex in examples.items():
        print(ex.n)
