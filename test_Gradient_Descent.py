import parallel_Gradient_Descent
from KevinBumperClass_adjustable import Bumper
import numpy as np

r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
l_norm = 0.25
d_norm = 0.25
L_norm = 0.05
phi_norm = 0.2
st_norm = 0.02


def func(args):
    d1, L, phi, d2, st = args
    testBumper = Bumper(0.0271, (9 + 7 / 8) * 0.0254, d1 * d_norm, L * L_norm, phi * phi_norm,
                        0.0208, (4 + 3 / 8) * 0.0254, d2 * d_norm, 500, start=st * st_norm, long_off=0.0,
                        focus_off=-0.03, real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 3 / 8 * 0.0254))
    return testBumper.cost(size_cost=True)


def main():
    xi = np.array([1.23265383, -0.49133712,  0.42441995, 0.68121338, -0.63251733])

    test_opt = parallel_Gradient_Descent.gradient_Descent(func, xi, 0.005, 20, Plot=True, disp=True)
    print(test_opt)


if __name__ == '__main__':
    main()

