import numpy as np
import math
import sympy as sp


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
        D_zones=Zone(shape='ball', center=[0] * 2, r=250 ** 2),
        f=[
            lambda x: -x[0],
            lambda x: -x[1]
        ],
        name='C5'
    ),
    2: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + x[0] * x[1],
            lambda x: -x[1]
        ],
        name='C6'
    ),
    3: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
        f=[
            lambda x: -2 * x[0] + x[0] * x[1],
            lambda x: -x[1] + x[0] * x[1],
        ],
        name='C7'
    ),
    4: Example(
        n=2,
        D_zones=Zone('ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + 2 * x[0]**2 * x[1],
            lambda x: -x[1]
        ],
        name='C8'
    ),
    5: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1300 ** 2),
        f=[
            lambda x: -x[0] ** 3 + x[1],
            lambda x: -x[0] - x[1],
        ],
        name='C9'
    ),
    6: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
        f=[
            lambda x: -x[0] ** 3 - x[1] ** 2,
            lambda x: x[0] * x[1] - x[1] ** 3,
        ],
        name='C10'
    ),
    7: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1000 ** 2),
        f=[
            lambda x: -x[0] + x[1],
            lambda x: 0.1 * x[0] - 2 * x[1] - x[0] ** 2 + 0.1 * x[0] ** 3,
        ],
        name='C11'
    ),
    8: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=pow(10, 2)),
        # D_zones=Zone(shape='box', low=[-2]*2,up=[2]*2),
        f=[
            lambda x: -x[1],
            lambda x: x[0] - x[1] * (1 - x[0] ** 2),
        ],
        name='C12'
    ),


    11: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=1 ** 2),
        f=[
            lambda x: -x[0] - 1.5 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 0.5 * x[0] ** 3 * x[1] ** 2
        ],
        name='C13'
    ),
    12: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=50 ** 2),
        f=[
            lambda x: -x[0] - 3 / 2 * x[0] ** 2 * x[1] ** 3,
            lambda x: -x[1] ** 3 + 1 / 2 * (x[0] * x[1]) ** 2
        ],
        name='C14'
    ),

    14: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=11 ** 2),
        f=[
            lambda x: -x[0],
            lambda x: -2 * x[1] + 0.1 * x[0] * x[1] ** 2 + x[2],
            lambda x: -x[2] - 1.5 * x[1]
        ],
        name='C16'
    ),

    16: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=1 ** 2),
        f=[
            lambda x: -3 * x[0] - 0.1 * x[0] * x[1] ** 3,
            lambda x: -x[1] + x[2],
            lambda x: -x[2]
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
    23: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=0.3**2),
        f=[
            lambda x:-2*x[1],
            lambda x:0.8*x[0] + 10 * (x[0]**2 - 0.21) * x[1]
        ],
        name='C1'
    ),
    24: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0]*3, r=0.5**2),
        f=[
            lambda x:x[0] * (x[0]**2 + x[1]**2 - 1) - x[1] * (x[2]**2 + 1),
            lambda x:x[1] * (x[0]**2 + x[1]**2 - 1) + x[0] * (x[2]**2 + 1),
            lambda x:10 * x[2] * (x[2]**2 - 1)
        ],
        name='C15'
    ),
    25:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=0.01**2),
        f=[
            lambda x:-0.42 * x[0] - 1.05 * x[1] - 2.3 * x[0] ** 2 - 0.5 * x[0] * x[1] - x[0] ** 3,
            lambda x:1.98 * x[0] + x[0] * x[1],
        ],
        name='C2'
    ),
    26:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=1**2),
        f=[
            lambda x:-x[0] * (1 - x[0] * x[1]),
            lambda x:-x[1]
        ],
        name='C3'
    ),
    27:Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=0.01**2),
        f=[
            lambda x:-5 * x[1] - 4 * x[0] + x[0] * x[1] ** 2 - 6 * x[0] ** 3,
            lambda x:-20 * x[0] - 4 * x[1] + 4 * x[0] ** 2 * x[1] + x[1] ** 3
        ],
        name='C4'
    ),
    ######NLC##########
    28: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=500 ** 2),
        f=[
            lambda x: -x[0]+x[0]*x[1],
            lambda x: -x[1]
        ],
        name='C_(5)'
    ),
    29: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0] * 2, r=100 ** 2),
        f=[
            lambda x: -x[0] + 2*x[0]*x[0] * x[1],
            lambda x: -x[1]
        ],
        name='C_(13)'
    ),
    30: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=1000 ** 2),
        f=[
            lambda x: -x[0],
            lambda x: -2*x[1]+0.1*x[0]*x[1]*x[1]+x[2],
            lambda x: -x[2]-1.5*x[1]
        ],
        name='C_(14)'
    ),
    31: Example(
        n=3,
        D_zones=Zone(shape='ball', center=[0] * 3, r=1000 ** 2),
        f=[
            lambda x: -3*x[0]-0.1*x[0]*x[1]**3,
            lambda x: -x[1]+x[2],
            lambda x: -x[2]
        ],
        name='C_(15)'
    ),
    32: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2, r=math.pi**2),
        f=[lambda x, u: x[1],
           lambda x, u: -10 * (9.87855464e-01*x[0]-1.55267355e-01*x[0]**3+5.64266597e-03*x[0]**5) - 0.1 * x[1] + -0.72054 - 2.66857 * x[0] - 10.6991 * x[1] - 0.13387 * x[
               0] ** 2 + 1.31456 * x[0] * x[1] + 1.04989 * x[1] + + 1.04989 * x[1] ** 2 + u[0]
           ],
        name='pendulum'
    ),
# pendulum
    33:Example(
            n=2,
            D_zones=Zone('ball',center=[0]*2,r=7*math.pi/10),
            f=[lambda x:(9.87855464e-01*x[1]-1.55267355e-01*x[1]**3+5.64266597e-03*x[1]**5),
               lambda x: 0.01177-3.01604*x[0]-19.59416*x[1]+2.96065*x[0]**2+27.86854*x[0]*x[1]+48.41103*x[1]**2
              ],#0.01177-3.01604*x1-19.59416*x2+2.96065*x1^2+27.86854*x1*x2+48.41103*x2^2
        name='dubin_car'
        ),#Dubins' Car
    34: Example(
        n=2,
        D_zones=Zone('ball', center=[0]*2,r=2),

        f=[lambda x: x[1],
           lambda x: (1 - x[0] ** 2) * x[1] - x[0] - 0.013343803937261811 + 1.2056734488868661 * x[
               0] + 1.2298618854459398 * x[1] + 13.532273236315778 * x[0] ** 2 + 81.16108339720827 * x[0] * x[
                            1] + 57.96265493455242 * x[1] ** 2 + 0.02511 - 4.22171 * x[0] - 20.82402 * x[1] - 10.57162 *
                        x[0] ** 2 - 53.29254 * x[0] * x[1] - 9.55162 * x[1] ** 2
           ],  # 0.01177-3.01604*x1-19.59416*x2+2.96065*x1^2+27.86854*x1*x2+48.41103*x2^2
name='Oscillator'
    ),  # Oscillator

    35: Example(
        n=2,
        D_zones=Zone(shape='ball', center=[0]*2,r=0.8),
        f=[lambda x: 6 * (9.87855464e-01*x[1]-1.55267355e-01*x[1]**3+5.64266597e-03*x[1]**5),
           lambda x: 6 * (-0.286*x[0]-1.3352*x[1]+0.2347*x[0]**2+0.3372*x[0]*x[1]-0.1027*x[1]**2) - ((-4.99998744e-01*x[1]**2+4.16558586e-02*x[1]**4-1.35953076e-03*x[1]**6+0.99999998) *(1+x[0]+x[0]**2+x[0]**3+x[0]**4+x[0]**5))
           ],
name='Vehicle'
    ),  # Vehicle path tracking
    36: Example(
            n=3,
    D_zones=Zone(shape='ball',center=[0]*3,r=5),
             f=[lambda x: x[2] + 8 * x[1],
               lambda x: -x[1] + x[2],
               lambda x: -x[2] - x[0] ** 2 + 0.1247-3.3332*x[0]-5.726*x[1]-10.6688*x[2]+1.9106*x[0]**2+1.2121*x[0]*x[1]+2.1376*x[0]*x[2]-1.3317*x[1]**2-10.0699*x[1]*x[2]-12.9515*x[2]**2,
                          ##--+
               ],



name='Academic 3D'
        ),#Academic 3D
    37: Example(
        n=3,
        D_zones=Zone('ball', center=[0]*3,r=2.2),
        f=[lambda x: x[1],
           lambda x: 30 * (9.87855464e-01*x[0]-1.55267355e-01*x[0]**3+5.64266597e-03*x[0]**5) + 15 * (-4.99998744e-01*x[0]**2+4.16558586e-02*x[0]**4-1.35953076e-03*x[0]**6+0.99999998) *(- 7.15341*x[0] - 4.73806*x[1] - 15.26188*x[2] + 1.4062*x[0]**2 - 6.96049*x[0]*x[1]+ 0.41919*x[0]*x[2] - 4.14752*x[1]**2 + 6.70594*x[1]*x[2] - 6.14518*x[2]**2),
           lambda x:-20*(9.87855464e-01*x[2]-1.55267355e-01*x[2]**3+5.64266597e-03*x[2]**5)*(-4.99998744e-01*x[2]**2+4.16558586e-02*x[2]**4-1.35953076e-03*x[2]**6+0.99999998)+(-4.99998744e-01*x[2]**2+4.16558586e-02*x[2]**4-1.35953076e-03*x[2]**6+0.99999998)**2*(- 7.15341*x[0] - 4.73806*x[1] - 15.26188*x[2] + 1.4062*x[0]**2 - 6.96049*x[0]*x[1]
+ 0.41919*x[0]*x[2] - 4.14752*x[1]**2 + 6.70594*x[1]*x[2] - 6.14518*x[2]**2)],

        name='Bicycle'
    ),  # Bicycle Steering
    38: Example(
        n=4,
        D_zones=Zone(shape='ball', center=[0]*4,r=1.3),
        f=[lambda x: x[2],
           lambda x: x[3],
           lambda x: (     -1.1237*x[0]-0.1558*x[1]-2.1002*x[2]-1.0579*x[3]-0.3102*x[0]**2-0.121*x[0]*x[1]-1.1061*x[0]*x[2]-0.2505*x[0]*x[3]+0.2049*x[1]**2-0.112*x[1]*x[2]-0.4603*x[1]*x[3]-0.5411*x[2]**2-0.4735*x[2]*x[3]-0.2273*x[3]**2) * (-9.49845082e-01*x[1]**2+9.19717026e-01*x[1]**4-4.06137871e-01*x[1]**6+0.99899106) + x[3] ** 2 * (9.78842244e-01*x[1]-8.87441593e-01*x[1]**3+4.35351792e-01*x[1]**5) - (9.70088125e-01 * x[1] - 1.27188818 * x[1] ** 3 + 6.16181488e-01 * x[1] ** 5),
           lambda x: (     -1.1237*x[0]-0.1558*x[1]-2.1002*x[2]-1.0579*x[3]-0.3102*x[0]**2-0.121*x[0]*x[1]-1.1061*x[0]*x[2]-0.2505*x[0]*x[3]+0.2049*x[1]**2-0.112*x[1]*x[2]-0.4603*x[1]*x[3]-0.5411*x[2]**2-0.4735*x[2]*x[3]-0.2273*x[3]**2) * (-1.42907660e+00 * x[1] ** 2 + 1.29010139e+00 * x[1] ** 4 - 5.75414531e-01 * x[1] ** 6 + 0.99857329) + x[3] ** 2 * (9.70088125e-01 * x[1] - 1.27188818 * x[1] ** 3 + 6.16181488e-01 * x[1] ** 5) - 2 * (9.78842244e-01*x[1]-8.87441593e-01*x[1]**3+4.35351792e-01*x[1]**5)
           ],
name='Cartpole'
    ),  # Cartpole linear
    13: Example(
        n=7,
        D_zones=Zone('ball',center=[0]*7,r=5),

        f=[lambda x: 1.4 * x[2] - 0.9 * x[0],
           lambda x: 2.5 * x[4] - 1.5 * x[1] + 0.62175*x[0] + 0.72265*x[1]-0.08394*x[2] + 0.32562*x[3]-1.12218*x[4] + 0.07453*x[5]
- 0.41217*x[6] - 0.02857*x[0]**2 - 0.48429*x[0]*x[1] - 0.24386*x[0]*x[2] - 0.08897*x[0]*x[3]
+ 0.47079*x[0]*x[4] - 0.37944*x[0]*x[5] + 0.6159*x[0]*x[6] + 0.07929*x[1]**2+ 0.318*x[1]*x[2]
- 0.15821*x[1]*x[3] - 0.31944*x[1]*x[4] + 0.58502*x[1]*x[5] - 0.21606*x[1]*x[6] + 0.09618*x[2]**2
+ 0.02111*x[2]*x[3] - 0.15265*x[2]*x[4] + 0.10732*x[2]*x[5] - 0.33442*x[2]*x[6] - 0.03321*x[3]**2
+ 0.24253*x[3]*x[4] - 0.64637*x[3]*x[5] - 0.01769*x[3]*x[6] + 0.28253*x[4]**2 + 0.98225*x[4]*x[5]
- 0.11167*x[4]*x[6] - 0.63756*x[5]**2 + 0.56567*x[5]*x[6] + 0.42151*x[6]**2,
           lambda x: 0.6 * x[6] - 0.8 * x[1] * x[2],
           lambda x: 2 - 1.3 * x[2] * x[3],
           lambda x: 0.7 * x[0] - x[3] * x[4],
           lambda x: 0.3 * x[0] - 3.1 * x[5],
           lambda x: 1.8 * x[5] - 1.5 * x[1] * x[6],
           ],
name='LALO20'
    ),  # Laub-Loomis





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
