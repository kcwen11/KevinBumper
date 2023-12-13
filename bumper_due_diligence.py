import numpy as npimport matplotlib.pyplot as pltimport multiprocess as mpimport randomfrom latticeElements.elements import HalbachLensSimfrom KevinBumperClass_grades import KevinBumper, recreate_bumper, trace_kevin_bumperassert HalbachLensSim.fringeFracOuter == 1.5n0 = 500def space_step(x_mult):    space_bumper = recreate_bumper(n0, field_mult=x_mult)  # default 1    print('x_mult', x_mult, space_bumper.cost(size_cost=False, area_cost=False))    return space_bumper.cost(size_cost=False, area_cost=False)def run_space():    with mp.Pool() as pool:        space_test = np.asarray(pool.map(space_step, np.arange(1, 2.5, 0.125)))    print(space_test)    plt.plot(np.arange(1, 2.5, 0.125), space_test, 'o-', color='black')    plt.title('Image Cost vs Relative Field Resolution')    plt.show()def time_step(dt):    KevinBumper.swarm = KevinBumper.trace_simulated_focus(n0, time_step=dt)  # default 5e-6    KevinBumper.obj_q, KevinBumper.obj_p, KevinBumper.im_q, KevinBumper.im_p = KevinBumper.get_phase()    print('dt', dt, KevinBumper.cost(size_cost=False, area_cost=False))    return KevinBumper.cost(size_cost=False, area_cost=False)def run_time():    with mp.Pool() as pool:        time_test = np.asarray(pool.map(time_step, np.arange(5e-6, 1e-6, -5e-7)))    print(time_test)    plt.plot(np.arange(5e-6, 1e-6, -5e-7), time_test, 'o-', color='black')    plt.title('Image Cost vs Time Step')    plt.show()def magnet_error():    error_bumper = recreate_bumper(n0, mag_err=True)    print('image cost with and without magnet error considerations',          error_bumper.cost(size_cost=False, area_cost=False), KevinBumper.cost(size_cost=False, area_cost=False))def plot_random_energy(n):    for _ in range(n):        KevinBumper.swarm.particles[random.randint(0, len(KevinBumper.swarm.particles) - 1)].plot_Energies()if __name__ == '__main__':    # run_space()    # run_time()    trace_kevin_bumper(n=n0, fast=False)  # creates the swarm in case run_time() gets skipped earlier    magnet_error()    plot_random_energy(5)    # go to constants.py and change the N52 magnetization around from the lower to upper bound (see the    # kj magnetics website link). tested on 6/28/2022, everything still works