import parallel_Gradient_Descent
from KevinBumperClass_adjustable import Bumper
import numpy as np

r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
len_norm = 0.25
d_norm = 0.25
L_norm = 0.1
phi_norm = 0.2
st_norm = 0.02


def func(args):
    l1, d1, L, phi, l2, d2, st = args
    testBumper = Bumper(0.95233173 * 0.0254, l1 * len_norm, d1 * d_norm, L * L_norm, phi * phi_norm,
                        0.71907843 * 0.0254, l2 * len_norm, d2 * d_norm, 500, start=st * st_norm, long_off=0.0,
                        focus_off=-0.03, real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 3 / 8 * 0.0254))
    return testBumper.cost(size_cost=True)


def main():
    xi = np.array([0.8382, 0.912, -0.265, 0.5, 0.4064, 0.34, -0.6])

    test_opt = parallel_Gradient_Descent.gradient_Descent(func, xi, 0.01, 20, Plot=True, disp=True)
    print(test_opt)


if __name__ == '__main__':
    main()

