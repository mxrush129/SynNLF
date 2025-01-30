import torch
import numpy as np
from utils.Config_V import CegisConfig
from learn.net_V import Learner
from learn.generate_data import Sampling_data, x2dotx
from verify.CounterExampleFind_V import CounterExampleFinder
from verify.SosVerify_V import SosValidator_V
import timeit


class Cegis:

    def __init__(self, config: CegisConfig):
        n = config.EXAMPLE.n
        self.ex = config.EXAMPLE

        self.n = n
        self.f = config.EXAMPLE.f
        self.batch_size = config.BATCH_SIZE
        self.learning_rate = config.LEARNING_RATE
        self.Learner = Learner(config)
        self.sample = Sampling_data(config)
        self.optimizer = config.OPT(self.Learner.net.parameters(), lr=self.learning_rate)
        self.CounterExampleFinder = CounterExampleFinder(config.EXAMPLE, config)

        self.max_cegis_iter = config.max_iter
        self.DEG = config.DEG

        self.beta = config.beta
    def solve(self):
        # print(list(self.Learner.net.parameters()))
        # import os
        # os.system('pause')
        S, Sdot = self.sample.generate_data()
        # the CEGIS loop
        deg = self.DEG
        t_learn = 0
        t_cex = 0
        t_sos = 0
        for i in range(self.max_cegis_iter):
            t1 = timeit.default_timer()
            self.Learner.learn(self.optimizer, S, Sdot)
            t2 = timeit.default_timer()
            t_learn += t2 - t1

            V = self.Learner.net.get_lyapunov()

            print(f'iter: {i + 1} \nV = {V}')

            t3 = timeit.default_timer()
            Sos_Validator = SosValidator_V(self.ex, V, self.beta)
            if Sos_Validator.SolveAll(deg=deg):
                print('SOS verification passed!')
                t4 = timeit.default_timer()
                t_sos += t4 - t3
                break

            t4 = timeit.default_timer()
            t_sos += t4 - t3

            t5 = timeit.default_timer()
            samples = self.CounterExampleFinder.find_counterexamples(V)
            t6 = timeit.default_timer()
            t_cex += t6 - t5
            if len(samples) == 0:
                print('No counterexamples were found!')
                # deg[1] += 2
            else:
                S, Sdot = self.add_ces_to_data(S, Sdot, torch.Tensor(np.array(samples)))
            print('-' * 200)
        print('Total learning time:{}'.format(t_learn))
        print('Total counter-examples generating time:{}'.format(t_cex))
        print('Total sos verifying time:{}'.format(t_sos))
        return t_learn,t_cex,t_sos

    def add_ces_to_data(self, S, Sdot, ces):

        print(f'Add {len(ces)} counterexamples!')
        S = torch.cat([S, ces], dim=0).detach()
        dot_ces = x2dotx(ces, self.f)
        Sdot = torch.cat([Sdot, dot_ces], dim=0).detach()

        return S, Sdot


if __name__ == '__main__':
    pass
