from benchmarks.Exampler_V import Example, Zone, get_example_by_name
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import sympy as sp
from benchmarks import C8

class Draw:
    def __init__(self, ex: Example, V):
        self.ex = ex
        self.V = V

    def set_V(self, V): self.V = V
    def plot_benchmark_2d(self, levels,color, show=True):
        ex = self.ex
        V = self.V

        ax = plt.gca()
        zone = self.draw_zone(ex.D_zones, 'black', 'ROA')

        r = np.sqrt(ex.D_zones.r)
        self.plot_contour(V, r, levels, color)
        self.plot_vector_field(r, ex.f)
        ax.add_patch(zone)
        ax.set_xlim(-1 * r, 1 * r)
        ax.set_ylim(-1 * r, 1 * r)
        ax.set_aspect(1)
        # plt.savefig(f'img/{self.ex.name}_2d.pdf', dpi=1000, bbox_inches='tight')

        return ax

    def plot_benchmark_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        r = np.sqrt(self.ex.D_zones.r)
        self.plot_barrier_3d(ax, r, self.V)

        domain = self.draw_zone(self.ex.D_zones, color='g', label='domain')
        ax.add_patch(domain)
        art3d.pathpatch_2d_to_3d(domain, z=0, zdir="z")
        plt.savefig(f'img/{self.ex.name}_3d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_barrier_3d(self, ax, r, v):
        r = 2 * r
        x = np.linspace(-r, r, 1000)
        y = np.linspace(-r, r, 1000)
        X, Y = np.meshgrid(x, y)
        s_x = sp.symbols(['x1', 'x2'])
        lambda_b = sp.lambdify(s_x, v, 'numpy')
        plot_b = lambda_b(X, Y)
        ax.plot_surface(X, Y, plot_b, rstride=5, cstride=5, alpha=0.5, cmap='cool')

    def draw_zone(self, zone: Zone, color, label, fill=False):
        if zone.shape == 'ball':
            circle = Circle(zone.center, np.sqrt(zone.r), color=color, label=label, fill=fill, linewidth=1.5)
            return circle
        else:
            w = zone.up[0] - zone.low[0]
            h = zone.up[1] - zone.low[1]
            box = Rectangle(zone.low, w, h, color=color, label=label, fill=fill, linewidth=1.5)
            return box

    def plot_contour(self, hx, r, levels, color):
        r = 2 * r
        x = np.linspace(-r, r, 1000)
        y = np.linspace(-r, r, 1000)

        X, Y = np.meshgrid(x, y)

        s_x = sp.symbols(['x1', 'x2'])
        fun_hx = sp.lambdify(s_x, hx, 'numpy')
        value = fun_hx(X, Y)
        CS = plt.contour(X, Y, value,levels=levels, alpha=1, linestyles="dashed", colors=color,
                         linewidths=3)
        # plt.clabel(CS, inline=True, fontsize=10)
    def plot_vector_field(self, r, f, color='grey'):
        r = 2 * r
        xv = np.linspace(-r, r, 100)
        yv = np.linspace(-r, r, 100)
        Xd, Yd = np.meshgrid(xv, yv)

        DX, DY = f[0]([Xd, Yd]), f[1]([Xd, Yd])
        DX = DX / np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
        DY = DY / np.linalg.norm(DY, ord=2, axis=1, keepdims=True)

        plt.streamplot(Xd, Yd, DX, DY, linewidth=0.3,
                       density=1.7, arrowstyle='-|>', arrowsize=1, color=color)


import pickle

if __name__ == "__main__":

    import sympy as sp
    from numpy import tanh
    from matplotlib import cm
    r = 1.
    X = np.linspace(-1., 1., 100)
    Y = np.linspace(-1., 1., 100)
    x1, x2 = np.meshgrid(X, Y)
    V = tanh((0.51641756296157837 + 0.75732171535491943 * tanh(
        (-1.6187947988510132 + 2.0125248432159424 * x1 - 0.86828583478927612 * x2)) - 1.6154271364212036 * tanh(
        (-1.0764049291610718 + 0.26035198569297791 * x1 - 0.058430317789316177 * x2)) + 1.2375599145889282 * tanh(
        (-0.96464759111404419 - 0.50644028186798096 * x1 + 1.4162489175796509 * x2)) + 0.41873458027839661 * tanh(
        (-0.82901746034622192 + 2.5682404041290283 * x1 - 1.2206004858016968 * x2)) - 0.89795422554016113 * tanh(
        (0.98988056182861328 + 0.83175277709960938 * x1 + 1.0546237230300903 * x2)) + 1.0879759788513184 * tanh(
        (1.1398535966873169 - 0.2350536435842514 * x1 + 0.075554989278316498 * x2))))
    V1 = sp.sympify('0.0957386811580085*x1**2 - 0.0207406181048899*x1*x2 + 0.116861324292346*x2**2')
    ex = get_example_by_name('C8')
    draw = Draw(ex, V1)
    ax1 = draw.plot_benchmark_2d(levels=[0.09], show=False, color='r')
    ax1.contour(x1, x2, V, 8, linewidths=0.2, alpha=0.2, colors='k')
    ax1.contourf(x1, x2, V, 8, alpha=0.4, cmap=cm.coolwarm)
    V2 = sp.sympify('-2.99992911136e-19*x1+1.12722834258e-20*x2+2.66354849158*x1^2+2.26094962287*x2^2-1.62204762326*x1*x2')
    draw.set_V(V2)
    ax2 = draw.plot_benchmark_2d(levels=[1.57], show=False, color='b')
    V3 = sp.sympify('( - 2.339 * x1 + 2.099 * x2)**2 + (1.861 * x1 + 1.677 * x2)**2 + (2.188 * x1 - 1.831 * x2)**2 + (2.404 * x1 - 2.75 * x2)**2 + (3.4 * x1 + 1.627 * x2)**2')
    draw.set_V(V3)
    ax3 = draw.plot_benchmark_2d(levels=[17], show=False, color='g')
    plt.title('Region of Attraction')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(f'img_2d.pdf', dpi=1000, bbox_inches='tight')
    plt.show()
