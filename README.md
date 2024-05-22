# 1.Introduction

Stability analysis forms the core of control system design, ensuring the stability of dynamical
systems by preventing deviations from a region of attraction around an equilibrium point. Moreover,
there is a consideration for potential asymptotic convergence towards the equilibrium point, all
within the theory of Lyapunov function. This tool introduces an novel approach to synthesizing
Lyapunov functions for nonlinear continuous systems, integrating learning and verification. We
efficiently leverage machine learning to train the Lyapunov neural network in a data-driven manner, automatically
generating candidate functions that may serve as Lyapunov certificate. For formal verification, we encode the Lyapunov
conditions for asymptotic stability into Linear Matrix Inequalities (LMIs) and solve LMI feasibility testing problems
for identifying a real one as formal certificate. In the case of verification failure, counterexamples violating the
requirements are computed through polynomial optimization technique. The constructed counterexample set is then
added to the dataset for refining neural network training. Comparative experiments on a set of benchmarks demonstrate
the efficiency and scalability of our tool compared to traditional numerical method and state-of-the-art neural network
based approach, enabling effective verification of high-dimensional systems over wider
region of attractions.

The directory in which you install SynNLF contains five subdirectories:

* `/benchmarks`: the source code and some examples;
* `/learn`: the code of learners;
* `/verify`: the code of verifiers;
* `/plots`: the code of plots;
* `/utils`: the configuration of CEGIS.

# 2.Configuration

## 2.1 System requirements

To install and run SynNLF, you need:

* Windows Platform: `Python 3.9`;
* Linux Platform: `Python 3.9`;
* Mac OS X Platform: `Python 3.9`.

## 2.2 Installation instruction

You need install required software packages listed below and setting up a MOSEK license .

1. Download SynHbc.zip, and unpack it;
2. Install the required software packages for using SynNLF:

```
matplotlib==3.5.3
numpy==1.23.2
scipy==1.9.0
SumOfSquares==1.2.1
sympy==1.11
torch==1.12.1
gurobipy~=11.0.0
mosek==10.0.30
picos==2.4.11
scikit-learn==1.2.2
cvxpy~=1.4.1
```

3. Obtain a fully featured Trial License if you are from a private or public company, or Academic License if you are a
   student/professor at a university.

* Free licenses
    * To obtain a trial license go to <https://www.mosek.com/products/trial/>
    * To obtain a personal academic license go to <https://www.mosek.com/products/academic-licenses/>
    * To obtain an institutional academic license go to <https://www.mosek.com/products/academic-licenses/>
    * If you have a custom license go to <https://www.mosek.com/license/request/custom/> and enter the code you
      received.
* Commercial licenses
    * Assuming you purchased a product ( <https://www.mosek.com/sales/order/>) you will obtain a license file.

# 3.Automated Synthesis of Neural Lyapunov function

## 3.1 New examples

In SynNLF, if we want to synthesize a barrier certificate, at first we need create a new example in the examples
dictionary in `Exampler_V.py`. Then we should confirm its number. In an example, its number is the key and value is the
new example constructed by Example class.

## 3.2 Inputs for new examples

At first, we should confirm the dimension `n` ,basic domains: `local` and differential equations.Here we show a hybrid
system example to illustrate.

**Example 1** &emsp; Suppose we wish to input the following example:

```python
1: Example(
    n=2,
    D_zones=Zone(shape='ball', center=[0] * 2, r=1500 ** 2),
    f=[
        lambda x: -x[0],
        lambda x: -x[1]
    ],
    name='C1'
)
```

Then we should create a new python file named 'C1.py'. In this file we can adjust the hyperparameters for learning,
verification and counterexample generation.

For Example 1, the code example is as follows:

```python
activations = ['SKIP']
hidden_neurons = [10] * len(activations)
example = get_example_by_name('C1')
start = timeit.default_timer()
opts = {
    "ACTIVATION": activations,
    "EXAMPLE": example,
    "N_HIDDEN_NEURONS": hidden_neurons,
    "BATCH_SIZE": 100,
    "LEARNING_RATE": 0.1,
    "LOSS_WEIGHT": (1.0, 1.0),
    "SPLIT_D": False,
    'BIAS': False,
    'DEG': [0, 0],
    'max_iter': 20,
    'counter_nums': 30,
    'ellipsoid': True,
    'x0': [10] * example.n
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
```

At last, run the current file and we can get verified Lyapunov function. For Example 1, the result is as follows:

```python
V = 1.87377173676601 * x1 ** 2 - 0.840059772877422 * x1 * x2 + 1.07420838586117 * x2 ** 2
```
