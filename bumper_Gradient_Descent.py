import parallel_Gradient_Descent
from KevinBumperClass_grades import Bumper
import numpy as np

r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
len_norm = 0.25
d_norm = 0.25
L_norm = 0.1
phi_norm = 0.2
st_norm = 0.02

norms = np.array([0.25, 0.25, 0.05, 0.2, 0.25, 0.25, 0.02])
mag_unc = 0.01 / np.sin(5 * np.pi / 12) * 0.0254
width_to_r = np.tan(np.pi / 12) * 2


def func(args):
    l1, d1, L, phi, l2, d2, st = args * norms
    testBumper = Bumper((0.5 * 0.0254 + mag_unc) / width_to_r, l1, d1, L, phi,
                        (0.5 * 0.0254 + mag_unc) / width_to_r, l2, d2, 1500, start=st, long_off=0.0,
                        focus_off=-0.03, real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 1 / 2 * 0.0254), he_leeway=0.002)
    return testBumper.cost(size_cost=True, alignment_cost=False)


def main():
    xi_meters = np.array([0.1905, 0.418, -0.0405, 0.096, 0.10795, 0.191, -0.0116])
    xi = xi_meters / norms
    test_opt = parallel_Gradient_Descent.gradient_Descent(func, xi, 0.01, 20, Plot=True, disp=True)
    print(test_opt)


if __name__ == '__main__':
    main()

