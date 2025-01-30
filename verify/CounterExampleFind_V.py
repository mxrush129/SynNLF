import torch
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import sympy as sp
from benchmarks.Exampler_V import Example, Zone, get_example_by_id
import cvxpy as cp


def split_bounds(bounds, n):
    """
    Divide an n-dimensional cuboid into 2^n small cuboids, and output the upper and lower bounds of each small cuboid.

    parameter: bounds: An array of shape (n, 2), representing the upper and lower bounds of each dimension of an
    n-dimensional cuboid.

    return:
        An array with a shape of (2^n, n, 2), representing the upper and lower bounds of the divided 2^n small cuboids.
    """

    if n == bounds.shape[0]:
        return bounds.reshape((-1, *bounds.shape))
    else:
        # Take the middle position of the upper and lower bounds of the current dimension as the split point,
        # and divide the cuboid into two small cuboids on the left and right.
        mid = (bounds[n, 0] + bounds[n, 1]) / 2
        left_bounds = bounds.copy()
        left_bounds[n, 1] = mid
        right_bounds = bounds.copy()
        right_bounds[n, 0] = mid
        # Recursively divide the left and right small cuboids.
        left_subbounds = split_bounds(left_bounds, n + 1)
        right_subbounds = split_bounds(right_bounds, n + 1)
        # Merge the upper and lower bounds of the left and right small cuboids into an array.
        subbounds = np.concatenate([left_subbounds, right_subbounds])
        return subbounds


class CounterExampleFinder:
    def __init__(self, example: Example, config):
        self.n = example.n
        self.inv = example.D_zones
        self.f = example.f
        self.eps = config.eps
        self.x = sp.symbols(['x{}'.format(i + 1) for i in range(self.n)])
        self.config = config
        self.nums = config.counter_nums
        self.x0 = np.array([11] * example.n)
        self.Global_Optimization = False
        self.ellipsoid = config.ellipsoid

    def find_counterexamples(self, V):
        res = []

        expr1 = V
        if self.Global_Optimization:
            vis1, x1 = self.get_extremum_cvxpy(self.inv, expr1)
        else:
            vis1, x1 = self.get_extremum_scipy(self.inv, expr1)
        if vis1:
            x1 = self.generate_sample(x1, expr1)
            if self.ellipsoid:
                x1 = self.get_counterexamples_by_ellipsoid(x1, self.nums)
            res.extend(x1)

        x = self.x
        expr2 = -sum([sp.diff(V, x[i]) * self.f[i](x) for i in range(self.n)])
        if self.config.SPLIT_D:
            bounds = self.split_zone(self.inv)
        else:
            bounds = [self.inv]

        for bound in bounds:
            if self.Global_Optimization:
                vis2, x2 = self.get_extremum_cvxpy(bound, expr2)
            else:
                vis2, x2 = self.get_extremum_scipy(bound, expr2)
            if vis2:
                x2 = self.generate_sample(x2, expr2)
                if self.ellipsoid:
                    x2 = self.get_counterexamples_by_ellipsoid(x2, self.nums)
                res.extend(x2)

        return res

    def generate_sample(self, x, expr):
        eps = self.eps
        nums = self.nums
        result = [x]
        for i in range(nums - 1):
            rd = (np.random.random(self.n) - 0.5) * eps
            rd = rd + x
            result.append(rd)
        fun = sp.lambdify(self.x, expr)
        result = [e for e in result if fun(*e) < 0]
        result = [e for e in result if self.check(e)]
        return result

    def check(self, x):
        zone = self.inv

        if zone.shape == 'ball':
            return sum((zone.center - x) ** 2) <= zone.r
        else:
            vis = True
            low, up = zone.low, zone.up
            for i in range(self.n):
                vis = vis and (low[i] <= x[i] <= up[i])
            return vis

    def get_extremum_scipy(self, zone: Zone, expr):
        x_ = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        opt = sp.lambdify(x_, expr)
        result = None
        if zone.shape == 'box':
            bound = tuple(zip(zone.low, zone.up))
            res = minimize(lambda x: opt(*x), self.x0, bounds=bound)
            if res.fun < 0 and res.success:
                print(f'Counterexample found:{res.x}')
                result = res.x
        elif zone.shape == 'ball':
            poly = zone.r
            for i in range(self.n):
                poly = poly - (x_[i] - zone.center[i]) ** 2
            poly_fun = sp.lambdify(x_, poly)
            con = {'type': 'ineq', 'fun': lambda x: poly_fun(*x)}
            res = minimize(lambda x: opt(*x), self.x0, constraints=con)
            if res.fun < 0 and res.success:
                print(f'Counterexample found:{res.x}')
                result = res.x
        if result is None:
            return False, []
        else:
            return True, result

    def get_extremum_cvxpy(self, zone: Zone, expr):
        x_ = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        opt = sp.lambdify(x_, expr)

        x = [cp.Variable(name=f'x{i + 1}') for i in range(self.n)]
        con = []

        if zone.shape == 'box':
            for i in range(self.n):
                con.append(x[i] >= zone.low[i])
                con.append(x[i] <= zone.up[i])
        elif zone.shape == 'ball':
            poly = zone.r
            for i in range(self.n):
                poly = poly - (x[i] - zone.center[i]) ** 2
            con.append(poly >= 0)

        obj = cp.Minimize(opt(*x))
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI)
        if prob.value < 0 and prob.status == 'optimal':
            ans = [e.value for e in x]
            print(f'Counterexample found:{ans}')
            return True, np.array(ans)
        else:
            return False, []

    def get_ellipsoid(self, data):
        n = self.n
        A = cp.Variable((n, n), PSD=True)
        B = cp.Variable((n, 1))
        con = []
        for e in data:
            con.append(cp.sum_squares(A @ np.array([e]).T + B) <= 1)

        obj = cp.Minimize(-cp.log_det(A))
        prob = cp.Problem(obj, con)
        try:
            prob.solve(solver=cp.MOSEK)
            if prob.status == 'optimal':
                P = np.linalg.inv(A.value)
                center = -P @ B.value
                return True, P, center
            else:
                return False, None, None
        except:
            return False, None, None

    def get_counterexamples_by_ellipsoid(self, data, nums):
        state, P, center = self.get_ellipsoid(data)
        if not state:
            return data
        else:
            ans = np.random.randn(nums, self.n)
            ans = np.array([e / np.sqrt(sum(e ** 2)) * np.random.random() ** (1 / self.n) for e in ans]).T
            ellip = P @ ans + center
            return list(ellip.T)

    def split_zone(self, zone: Zone):
        bound = list(zip(zone.low, zone.up))
        bounds = split_bounds(np.array(bound), 0)
        ans = [Zone(shape='box', low=e.T[0], up=e.T[1]) for e in bounds]
        return ans


if __name__ == '__main__':
    """

    test code!!

    """
    ex = get_example_by_id(2)
