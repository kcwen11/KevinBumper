import numpy as np
import matplotlib.pyplot as plt
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from helperTools import *
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors

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
                 focus_off=0.0, he_leeway=None, real_mag=None, grade='N52', r1p2=None, ap=(None, None)):
        self.r1p1 = r1p1
        self.magwidth1 = r1p1 * np.tan(np.pi / 12) * 2 if real_mag is None else real_mag[0]
        self.r1p2 = r1p1 + r1p1 * np.tan(np.pi / 12) * 2 if r1p2 is None else r1p2
        self.magwidth2 = self.r1p2 * np.tan(np.pi / 12) * 2 if real_mag is None else real_mag[1]
        self.grade = grade
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

        self.PTL: ParticleTracerLattice = self.create_lattice()

        self.helium_tube = self.create_he_tube(l1 + self.r1p2 * 3 + d1 + l2 + r2 * 3 + d2, extra_width=he_leeway)
        self.helium_traj = self.create_he_tube(l1 + self.r1p2 * 3 + d1 + l2 + r2 * 3 + d2)
        self.he_tube_intersect = self.tube_intersection()
        if trace:
            self.swarm = self.trace_simulated_focus(n)
            self.obj_q, self.obj_p, self.im_q, self.im_p = self.get_phase()

    def create_lattice(self):
        delta_x = self.r2 * 1.5 * np.cos(self.phi)
        delta_y = self.r2 * 1.5 * np.sin(self.phi)
        a1 = np.arctan((-self.L - delta_y) / (self.d1 - self.r1p2 * 1.5 - delta_x))
        a2 = a1 - self.phi
        d_fix = np.sqrt((-self.L - delta_y) ** 2 + (self.d1 - self.r1p2 * 1.5 - delta_x) ** 2)
        l1_plus_fringe = self.l1 + self.r1p2 * 3
        l2_plus_fringe = self.l2 + self.r2 * 3

        PTL = ParticleTracerLattice(latticeType='injector', initialAngle=self.angle,
                                    initialLocation=(self.long_off, self.start), magnetGrade=self.grade)
        PTL.add_Halbach_Lens_Sim((self.r1p1, self.r1p2), l1_plus_fringe,
                                 magnetWidth=(self.magwidth1, self.magwidth2), ap=self.ap[0])
        PTL.add_Drift(d_fix, .04, inputTiltAngle=a1, outputTiltAngle=a2)
        PTL.add_Halbach_Lens_Sim(self.r2, l2_plus_fringe, magnetWidth=self.magwidth3, ap=self.ap[1])
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

    def trace_simulated_focus(self, n):
        st = SwarmTracer(self.PTL)
        swarm = st.initialize_Simulated_Collector_Focus_Swarm(n)

        # turn stuff around, project backwards
        if not self.leftwards:
            for particle in swarm:
                particle.pi[0] *= -1
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

        swarm = st.trace_Swarm_Through_Lattice(swarm, 5e-6, 1.0, fastMode=False)
        return swarm

    def plot_trace(self):
        plt.plot(*self.helium_tube.exterior.xy, color='blue')
        plt.plot(*self.helium_traj.exterior.xy, linestyle=':', color='blue')
        if isinstance(self.he_tube_intersect, Polygon):
            plt.plot(*self.he_tube_intersect.exterior.xy, color='red')
        elif isinstance(self.he_tube_intersect, MultiPolygon):
            for geom in self.he_tube_intersect.geoms:
                plt.plot(*geom.exterior.xy, color='red')
        self.PTL.show_Lattice(plotOuter=True, plotInner=True, swarm=self.swarm, showTraceLines=True, traceLineAlpha=.25)

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
        plt.plot(self.obj_q, self.obj_p, 'bo')
        plt.plot(self.im_q, self.im_p, 'go')
        plt.show()

    def tube_intersection(self):
        return self.PTL.elList[2].SO_Outer.intersection(self.helium_tube)

    def image_quality(self):
        q_range = max(self.obj_q) - min(self.obj_q)
        p_range = max(self.obj_p) - min(self.obj_p)

        im = np.concatenate((np.array([self.im_q]) / q_range, np.array([self.im_p]) / p_range), axis=0).T
        obj = np.concatenate((np.array([self.obj_q]) / q_range, np.array([self.obj_p]) / p_range), axis=0).T
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nbrs.fit(obj)
        return np.array(nbrs.kneighbors(im)[0])

    def size_penalty(self):
        return 15 * self.r1p2 + 3 * self.l1 + 3 * self.l2 + 3 * self.L  # the displacement L is negative, we want more

    def cost(self, area_cost=True, image_cost=True, size_cost=True, alignment_cost=False):
        area = self.he_tube_intersect.area if area_cost else 0
        im_qual = np.sum(self.image_quality() ** 2) if image_cost else 0
        size_p = self.size_penalty() if size_cost else 0
        alignment_p = self.alignment()[0] * 25 + self.alignment()[1] * 10 if alignment_cost else 0
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
        print('apertures (radius)', self.PTL.elList[0].ap * ratio, self.PTL.elList[2].ap * ratio)
        print('outer radius', self.PTL.elList[0].outerHalfWidth * ratio, self.PTL.elList[2].outerHalfWidth * ratio)
        print('first mag size', self.magwidth1 * ratio, self.magwidth2 * ratio)
        print('second mag size', self.magwidth3 * ratio)

    def clone_bumper(self, print_code=False):  # creates an identical bumper, but does not end the lattice
        delta_x = self.r2 * 1.5 * np.cos(self.phi)
        delta_y = self.r2 * 1.5 * np.sin(self.phi)
        a1 = np.tan((self.L - delta_y) / (self.d1 - self.r1p2 * 1.5 - delta_x))
        a2 = a1 + self.phi
        d_fix = np.sqrt((self.L - delta_y) ** 2 + (self.d1 - self.r1p2 * 1.5 - delta_x) ** 2)
        l1_plus_fringe = self.l1 + self.r1p2 * 3
        l2_plus_fringe = self.l2 + self.r2 * 3

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
# L_norm = 0.05
# phi_norm = 0.2
# st_norm = 0.02
#
# opt_norm = [0.54260155,  1.00106384,  1.23265383, -0.49133712,  0.42441995,
#             0.41541118,  0.44413424,  0.68121338, -0.63251733]
# opt_p = [opt_norm[0] * r_norm, opt_norm[1] * l_norm, opt_norm[2] * d_norm, opt_norm[3] * L_norm - 0.001,
#          opt_norm[4] * phi_norm, opt_norm[5] * r_norm, opt_norm[6] * l_norm, opt_norm[7] * d_norm - 0.01,
#          opt_norm[8] * st_norm]


# KevinBumper = Bumper(0.0271, (9 + 7 / 8) * 0.0254, 0.3184, -0.0259, 0.088, 0.0208, (4 + 3 / 8) * 0.0254, 0.178, 500,
#                      focus_off=-0.03, start=-0.0125, actual_he=True,
#                      real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 3 / 8 * 0.0254))


mag_unc = 0.01 / np.sin(5 * np.pi / 12) * 0.0254
width_to_r = np.tan(np.pi / 12) * 2
KevinBumper = Bumper((0.5 * 0.0254 + mag_unc) / width_to_r, (8 + 1/4) * 0.0254, 0.228, -0.0265, 0.10,
                     (0.375 * 0.0254 + mag_unc) / width_to_r, 4 * 0.0254, 0.085, 500, focus_off=-0.03,
                     start=-0.012, he_leeway=0.00127, real_mag=(1/2 * 0.0254, 3/4 * 0.0254, 3/8 * 0.0254),
                     long_off=0, leftwards=False, ap=(0.825 * 0.0254, 0.575 * 0.0254))

# KevinBumper = Bumper((0.5 * 0.0254 + mag_unc) / width_to_r, (8 + 1 / 4) * 0.0254, 0.23, -0.02, 0.1,
#                      (0.375 * 0.0254 + mag_unc) / width_to_r, 4 * 0.0254, 0.085, 500, focus_off=-0.03,
#                      start=-0.012, actual_he=True, real_mag=(1 / 2 * 0.0254, 3 / 4 * 0.0254, 3 / 8 * 0.0254),
#                      long_off=0, leftwards=False)

# test_bumper = Bumper(opt_p[0], opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 500,
#                      focus_off=-0.03, start=opt_p[8], actual_he=True)


if __name__ == '__main__':
    # print(KevinBumper.PTL.elList[0].ap, KevinBumper.PTL.elList[2].ap)
    # print('position and angle alignment', KevinBumper.alignment())
    # KevinBumper.plot_trace()
    # print('quality (low is better)', KevinBumper.cost(size_cost=False))
    # KevinBumper.plot_phase()

    KevinBumper.print_params()
    KevinBumper.print_params(inches=True)
    print('position and angle alignment', KevinBumper.alignment())
    KevinBumper.plot_trace()
    print('quality (low is better)', KevinBumper.cost(size_cost=False, area_cost=False))
    KevinBumper.plot_phase()
    KevinBumper.obj_q, KevinBumper.obj_p, KevinBumper.im_q, KevinBumper.im_p = KevinBumper.get_phase(coord=2)
    KevinBumper.plot_phase()
