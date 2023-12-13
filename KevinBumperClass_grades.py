import numpy as np
import matplotlib.pyplot as plt
import random
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from helperTools import *
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.75
mpl.rcParams['ytick.major.width'] = 1.75
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams["errorbar.capsize"] = 4


#  made 6/17/2022, after four rounds of optimizing.
#  use KevinBumper.clone_bumper() to create an identical bumper without an end, so more elements can be added.
#  (or use KevinBumperOpen, the same thing but maybe more convenient)

#  if you want to add the KevinBumper to something else, you would need to consider the offset drift before it.
#  if you want the bumper to be point left and start at zero, call KevinBumper.clone_bumper(angle=np.pi, long_off=0)

#  the focus should be at the variable np.abs(KevinBumper.start) above the center of the entrance of the first lens.

#  be careful when tracing the swarm in the cloned bumper, you would need to flip the velocities and project the
#  particles backwards a bit. look at KevinBumper.trace_simulated_focus for the flip


mu = 9.274 * 10 ** -24  # J/T
m_li7 = 1.165 * 10 ** -26  # kg
v = 210  # m/s
good_field_1 = 0.8  # percent of the radius of the first magnet that is usable
good_field_2 = 0.9


def y0_max(r, b_max):
    numerator = good_field_1 ** 2 * mu * b_max - 0.5 * m_li7 * (v * 0.0521) ** 2
    return r * np.sqrt(numerator / (mu * b_max)) - 0.0075


class Bumper:
    def __init__(self, r1p1, l1, d1, L, phi, r2, l2, d2, n, start=None, leftwards=False, long_off=None, trace=True,
                 focus_off=0.0, he_leeway=None, real_mag=None, grade=(['N42', 'N42'], ['N52']), r1p2=None,
                 ap=(None, None), fdm=1, magnet_error=False, mesh_first=False, mesh_second=False, hk=(None, None)):
        self.r1p1 = r1p1
        self.magwidth1 = r1p1 * np.tan(np.pi / 12) * 2 if real_mag is None else real_mag[0]
        self.r1p2 = r1p1 + r1p1 * np.tan(np.pi / 12) * 2 if r1p2 is None else r1p2
        self.magwidth2 = self.r1p2 * np.tan(np.pi / 12) * 2 if real_mag is None else real_mag[1]
        self.grade = grade
        self.hk = hk
        self.ap = ap

        self.l1 = l1
        self.d1 = d1
        self.L = L
        self.phi = phi
        self.r2 = r2
        self.magwidth3 = r2 * np.tan(np.pi / 12) * 2 if real_mag is None else real_mag[2]
        self.l2 = l2
        self.d2 = d2

        self.start = -y0_max(r1p1, 0.9) if start is None else start
        self.start = -self.start if leftwards else self.start
        self.long_off = -self.r1p2 * 1.5 if long_off is None else long_off
        self.angle = np.pi if leftwards else 0.0
        self.leftwards = leftwards
        assert focus_off <= 0
        self.focus_off = focus_off

        self.do_I_mesh_this = mesh_first
        self.do_I_mesh_second = mesh_second
        self.PTL: ParticleTracerLattice = self.create_lattice(fdm, magnet_error)

        self.helium_tube = self.create_he_tube(l1 + self.r1p2 * 3 + d1 + l2 + r2 * 3 + d2, extra_width=he_leeway)
        self.helium_traj = self.create_he_tube(l1 + self.r1p2 * 3 + d1 + l2 + r2 * 3 + d2)
        self.he_tube_intersect = self.tube_intersection()
        if trace:
            self.swarm = self.trace_simulated_focus(n)
            self.obj_q, self.obj_p, self.im_q, self.im_p = self.get_phase()

    def create_lattice(self, fdm, mag_err):
        delta_x = self.r2 * 1.5 * np.cos(self.phi)
        delta_y = self.r2 * 1.5 * np.sin(self.phi)
        a1 = np.arctan((-self.L - delta_y) / (self.d1 - self.r1p2 * 1.5 - delta_x))
        a2 = a1 - self.phi
        d_fix = np.sqrt((-self.L - delta_y) ** 2 + (self.d1 - self.r1p2 * 1.5 - delta_x) ** 2)
        l1_plus_fringe = self.l1 + self.r1p2 * 3
        l2_plus_fringe = self.l2 + self.r2 * 3

        PTL = ParticleTracerLattice(latticeType='injector', initialAngle=self.angle,
                                    initialLocation=(self.long_off, self.start), magnetGrade=self.grade,
                                    fieldDensityMultiplier=fdm, standardMagnetErrors=mag_err, hk_list=self.hk)
        PTL.add_Halbach_Lens_Sim((self.r1p1, self.r1p2), l1_plus_fringe,
                                 magnetWidth=(self.magwidth1, self.magwidth2), ap=self.ap[0], mesh=self.do_I_mesh_this)
        PTL.add_Drift(d_fix, .04, inputTiltAngle=a1, outputTiltAngle=a2)
        PTL.add_Halbach_Lens_Sim(self.r2, l2_plus_fringe, magnetWidth=self.magwidth3, ap=self.ap[1],
                                 mesh=self.do_I_mesh_second)
        PTL.add_Drift(self.d2, .04)

        PTL.end_Lattice()
        return PTL

    def create_he_tube(self, tube_l, extra_width=None):
        height = 0.005 if extra_width is None else 0.005 + extra_width
        if not self.leftwards:
            return Polygon([(self.r1p2 * 1.5 + self.long_off + self.focus_off, -height),
                            (self.r1p2 * 1.5 + self.long_off + self.focus_off, height),
                            (tube_l, height + tube_l * 0.00264), (tube_l, -height - tube_l * 0.00264)])
        else:
            return Polygon([(-(self.r1p2 * 1.5 + self.long_off + self.focus_off), -height),
                            (-(self.r1p2 * 1.5 + self.long_off + self.focus_off), height),
                            (-tube_l, height + tube_l * 0.00264), (-tube_l, -height - tube_l * 0.00264)])

    def trace_simulated_focus(self, n, time_step=5e-6, fast=True):
        st = SwarmTracer(self.PTL)
        swarm = st.initialize_Simulated_Collector_Focus_Swarm(n)

        warp = 1.0
        for particle in swarm:
            particle.pi[1] *= warp
            particle.pi[2] *= warp
            particle.qi[1] /= warp
            particle.qi[2] /= warp

            # particle.qi[1] *= 10 ** -5
            # particle.qi[2] *= 10 ** -5


        # turn stuff around, project backwards
        if not self.leftwards:
            for particle in swarm:
                particle.pi[0] *= -1  # EDIT for different focus
                particle.obj_qi = particle.qi.copy()
                t = (-self.r1p2 * 1.5 - self.focus_off + particle.qi[0]) / particle.pi[0]
                particle.qi[0] = self.long_off + 10 ** -4
                particle.qi[1] = particle.qi[1] + t * particle.pi[1]
                particle.qi[2] = particle.qi[2] + t * particle.pi[2]
        elif self.leftwards:
            for particle in swarm:
                particle.obj_qi = particle.qi.copy()
                t = (self.r1p2 * 1.5 + self.focus_off - particle.qi[0]) / particle.pi[0]
                particle.qi[0] = self.long_off - 10 ** -4
                particle.qi[1] = particle.qi[1] + t * particle.pi[1]
                particle.qi[2] = particle.qi[2] + t * particle.pi[2]

        swarm = st.trace_Swarm_Through_Lattice(swarm, time_step, 1.0, fastMode=fast)
        return swarm

    def plot_trace(self, saveTitle=None):
        plt.tick_params(axis='both', direction='in', top='true', right='true')
        plt.plot(*self.helium_tube.exterior.xy, color='blue')
        plt.plot(*self.helium_traj.exterior.xy, linestyle=':', color='blue')
        if isinstance(self.he_tube_intersect, Polygon):
            plt.plot(*self.he_tube_intersect.exterior.xy, color='red')
        elif isinstance(self.he_tube_intersect, MultiPolygon):
            for geom in self.he_tube_intersect.geoms:
                plt.plot(*geom.exterior.xy, color='red')
        plt.rc('xtick', labelsize=24)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=24)  # fontsize of the tick labels
        self.PTL.show_Lattice(plotOuter=True, plotInner=True, swarm=self.swarm, showTraceLines=True, traceLineAlpha=.25,
                              saveTitle=saveTitle)

    def get_phase(self, coord=1, save=False, name='kevin_bumper_phase.txt'):
        assert coord == 1 or coord == 2 or coord == 0
        p_avg = 0
        q_avg = 0
        living_count = 0
        for particle in self.swarm:
            if not particle.clipped:
                living_count += 1
                p_avg += particle.pf[coord]
                q_avg += particle.qf[coord]
        if living_count > 1:
            p_avg /= living_count
            q_avg /= living_count
        elif living_count == 1:
            print('there were no survivors, i am the only one left')
        else:
            print('there were no survivors, there is no one left')

        obj_q, obj_p, im_q, im_p = [], [], [], []
        for particle in self.swarm:
            obj_q.append(particle.obj_qi[coord])
            obj_p.append(particle.pi[coord])
            im_q.append(particle.qf[coord] - q_avg)
            im_p.append(particle.pf[coord] - p_avg)

        if save:
            with open(name, 'w') as f:
                for particle in self.swarm:
                    f.write(' '.join(str(e) for e in particle.qf))
                    f.write('    ')
                    f.write(' '.join(str(e) for e in particle.pf))
                    f.write('\n')

        return obj_q, obj_p, im_q, im_p

    def plot_phase(self):
        plt.plot(self.obj_q, self.obj_p, 'bo', label='Object')
        plt.plot(self.im_q, self.im_p, 'go', label='Image')
        plt.legend()
        plt.show()

    def tube_intersection(self):
        return self.PTL.elList[2].SO_Outer.intersection(self.helium_tube)

    def image_quality(self):
        assert hasattr(self, 'obj_q'), 'be sure to trace the particles first, check if trace=False'
        q_range = max(self.obj_q) - min(self.obj_q)
        p_range = max(self.obj_p) - min(self.obj_p)
        im = np.concatenate((np.array([self.im_q]) / q_range, np.array([self.im_p]) / p_range), axis=0).T
        obj = np.concatenate((np.array([self.obj_q]) / q_range, np.array([self.obj_p]) / p_range), axis=0).T
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nbrs.fit(obj)
        return np.array(nbrs.kneighbors(im)[0])

    def size_penalty(self):  # the displacement L is negative, we want more
        return 5 * self.L - 8 * np.abs(self.start) + 3 * self.l1 + 3 * self.l2 + self.d1 + self.d2  # + 15 * self.r1p2

    def cost(self, area_cost=True, image_cost=True, size_cost=True, alignment_cost=False):
        area = self.he_tube_intersect.area if area_cost else 0
        im_qual = np.sum(self.image_quality() ** 4) * 50000 / len(self.im_q) if image_cost else 0
        size_p = self.size_penalty() if size_cost else 0
        alignment_p = self.alignment()[0] ** 2 * 25 + self.alignment()[1] ** 2 * 10 if alignment_cost else 0
        return im_qual + np.exp(area * 5000) - 1 + size_p + alignment_p

    def alignment(self):
        im_q, im_p = [], []
        for particle in self.swarm:
            im_q.append(particle.qf)
            im_p.append(particle.pf)
        avg_q = sum(np.array(im_q)) / len(im_q)
        avg_p = sum(np.array(im_p)) / len(im_p)
        el_q = self.PTL.elList[-1].transform_Lab_Coords_Into_Element_Frame(avg_q)
        el_p = self.PTL.elList[-1].transform_Lab_Coords_Into_Element_Frame(avg_p)
        angle = np.tan(el_p[1] / el_p[0])
        return el_q[1], angle

    def print_params(self, inches=False):
        ratio = 39.3701 if inches else 1
        print('#' * 100)
        if inches:
            print('in inches!!!!')
        print('params: r1, l1, d1, L, phi, r2, l2, d2')
        print(np.array([self.r1p1, self.l1, self.d1, self.L, self.phi, self.r2, self.l2, self.d2]) * ratio)
        print('he offset,', self.start * ratio)
        print('apertures (radius)', self.PTL.elList[0].ap * ratio, self.PTL.elList[2].ap * ratio)
        print('outer radius', self.PTL.elList[0].outerHalfWidth * ratio, self.PTL.elList[2].outerHalfWidth * ratio)
        print('first mag size', self.magwidth1 * ratio, self.magwidth2 * ratio)
        print('second mag size', self.magwidth3 * ratio)
        print('#' * 100)

    def clone_bumper(self, print_code=False):  # creates an identical bumper, but does not end the lattice
        delta_x = self.r2*1.5*np.cos(self.phi)
        delta_y = self.r2*1.5*np.sin(self.phi)
        a1 = np.arctan((-self.L-delta_y)/(self.d1-self.r1p2*1.5-delta_x))
        a2 = a1-self.phi
        d_fix = np.sqrt((-self.L-delta_y)**2+(self.d1-self.r1p2*1.5-delta_x)**2)
        l1_plus_fringe = self.l1+self.r1p2*3
        l2_plus_fringe = self.l2+self.r2*3

        PTL = ParticleTracerLattice(latticeType='injector', initialAngle=self.angle,
                                    initialLocation=(self.long_off, self.start))
        PTL.add_Halbach_Lens_Sim((self.r1p1, self.r1p2), l1_plus_fringe,
                                 magnetWidth=(self.magwidth1, self.magwidth2))
        PTL.add_Drift(d_fix, .04, inputTiltAngle=-a1, outputTiltAngle=-a2)
        PTL.add_Halbach_Lens_Sim(self.r2, l2_plus_fringe)
        PTL.add_Drift(self.d2, .04)
        if print_code:
            print('def add_Kevin_Bumper_Elements(PTL):  # creates an identical bumper, but does not end the lattice')
            print('assert PTL.initialLocation[0] == 0.0 and PTL.initialLocation[1] == 0.0')
            print('PTL.initialLocation =', (self.long_off, self.start))
            print('PTL.add_Halbach_Lens_Sim(', (self.r1p1, self.r1p2), ',', l1_plus_fringe, ', magnetWidth=',
                  (self.magwidth1, self.magwidth2), ')')
            print('PTL.add_Drift(', d_fix, ', .04, inputTiltAngle=', -a1, ', outputTiltAngle=', -a2, ')')
            print('PTL.add_Halbach_Lens_Sim(', self.r2, ',', l2_plus_fringe, ')')
            print('PTL.add_Drift(', self.d2, ', .04', ')')
            print('return PTL')
        return PTL

# r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
# l_norm = 0.25
# d_norm = 0.25
# L_norm = 0.1
# phi_norm = 0.2
# st_norm = 0.02

# norms = np.array([0.5, 0.5, 0.1, 0.2, 0.25, 0.25, 0.02])
# opt_norm = np.array([0.37656675, 0.83548451, -0.42736662, 0.46024255, 0.43031401, 0.76274515, -0.58230727])
# opt_p = opt_norm * norms

# mag_unc = 0.01 / np.sin(5 * np.pi / 12) * 0.0254
# width_to_r = np.tan(np.pi / 12) * 2
# norms = np.array([0.25, 0.25, 0.1, 0.2, 0.25, 0.25, 0.02])
# opt_norm = np.array([0.77465948,  1.68598647, -0.41340418,  0.48585299,  0.45232657, 0.76736094, -0.57018616])
# opt_p = opt_norm * norms
#
# KevinBumper = Bumper((0.5 * 0.0254 + mag_unc) / width_to_r, opt_p[0], opt_p[1], opt_p[2] + 0.001, opt_p[3] - 0.004,
#                      (0.5 * 0.0254 + mag_unc) / width_to_r, opt_p[4], opt_p[5] - 0.01, 500, focus_off=-0.03,
#                      start=opt_p[6], he_leeway=0.00127, real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 1 / 2 * 0.0254),
#                      long_off=0, leftwards=False, ap=(0.825 * 0.0254, 0.825 * 0.0254), trace=False)

# test_bumper = Bumper(opt_p[0], opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 500,
#                      focus_off=-0.03, start=opt_p[8], actual_he=True)


mag_unc = 0.01 * 0.0254
width_to_r = np.tan(np.pi / 12) * 2
KevinBumper = Bumper((0.51 * 0.0254 + mag_unc) / width_to_r, (7 + 1/2) * 0.0254, 0.421, -0.038, 0.093,
                     (0.51 * 0.0254 + mag_unc) / width_to_r, (4 + 1/2) * 0.0254, 0.182, 500, focus_off=-0.03,
                     start=-0.0114, he_leeway=0.00127, real_mag=(1/2 * 0.0254, 3/4 * 0.0254, 1/2 * 0.0254),
                     long_off=0, leftwards=False, ap=(0.805 * 0.0254, 0.805 * 0.0254), trace=False, hk=(14.75, 10),
                     r1p2=(1.9 - 0.75 / 2) * 0.0254)
#  with mesh quality 0.39654434764912216
#  without 0.39869993081704314


def recreate_bumper(n, field_mult=1, mag_err=False):
    remake = Bumper((0.5 * 0.0254 + mag_unc) / width_to_r, (7 + 1/2) * 0.0254, 0.421, -0.04, 0.094,
                    (0.5 * 0.0254 + mag_unc) / width_to_r, (4 + 1/2) * 0.0254, 0.182, n, focus_off=-0.03,
                    start=-0.0114, he_leeway=0.00127, real_mag=(1/2 * 0.0254, 3/4 * 0.0254, 1/2 * 0.0254),
                    long_off=0, leftwards=False, ap=(0.825 * 0.0254, 0.825 * 0.0254), fdm=field_mult,
                    magnet_error=mag_err)
    return remake


def trace_kevin_bumper(n=500, save=False, name='kevin_bumper_phase.txt', fast=True):
    KevinBumper.swarm = KevinBumper.trace_simulated_focus(n, fast=fast)
    KevinBumper.obj_q, KevinBumper.obj_p, KevinBumper.im_q, KevinBumper.im_p = KevinBumper.get_phase(
        save=save, name=name)


if __name__ == '__main__':
    KevinBumper.print_params()
    KevinBumper.print_params(inches=True)

    trace_kevin_bumper(fast=False)
    print('position and angle alignment', KevinBumper.alignment())
    plt.title('Traced Trajectory of Lithium Beam')
    plt.xlabel('z position $(m)$')
    plt.ylabel('x position $(m)$')
    KevinBumper.plot_trace(saveTitle='trace.png')
    print('quality (low is better)', KevinBumper.cost(size_cost=False))
    plt.title('Object vs Image Phase Space Distribution')
    plt.xlabel('x position spread from mean $(m)$')
    plt.ylabel('x velocity spread from mean $(m)$')
    KevinBumper.plot_phase()
    KevinBumper.obj_q, KevinBumper.obj_p, KevinBumper.im_q, KevinBumper.im_p = KevinBumper.get_phase(coord=2)
    KevinBumper.plot_phase()
    KevinBumper.swarm.particles[random.randint(0, len(KevinBumper.swarm.particles) - 1)].plot_Energies()

