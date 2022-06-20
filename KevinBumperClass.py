import numpy as np
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from helperTools import *
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors


#  made 6/13/2022, after two rounds of optimizing.
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
    def __init__(self, r1p1, l1, d1, L, phi, r2, l2, d2, n, start=None, leftwards=False, long_off=None, trace=True):
        self.r1p1 = r1p1
        self.magwidth1 = r1p1 * np.tan(2 * np.pi / 24) * 2
        print(self.magwidth1)
        self.r1p2 = r1p1 + self.magwidth1
        self.magwidth2 = self.r1p2 * np.tan(2 * np.pi / 24) * 2
        print(self.magwidth2)

        self.l1 = l1
        self.d1 = d1
        self.L = L
        self.phi = phi
        self.r2 = r2
        self.l2 = l2
        self.d2 = d2

        self.start = -y0_max(r1p1, 0.9) if start is None else start
        self.start = -self.start if leftwards else self.start
        self.long_off = -self.r1p2 * 1.5 if long_off is None else long_off
        self.angle = np.pi if leftwards else 0.0
        self.leftwards = leftwards

        self.PTL: ParticleTracerLattice = self.create_lattice()
        self.helium_tube = self.create_he_tube(l1 + self.r1p2 * 3 + d1 + l2 + r2 * 3 + d2)
        self.he_tube_intersect = self.tube_intersection()
        if trace:
            self.swarm = self.trace_simulated_focus(n)
            self.obj_q, self.obj_p, self.im_q, self.im_p = self.get_phase()

    def create_lattice(self):
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
        PTL.end_Lattice()
        return PTL

    def create_he_tube(self, tube_l):
        if not self.leftwards:
            helium_tube = Polygon([(0, -0.006), (0, 0.006), (tube_l, 0.006 + tube_l * 0.00338),
                                   (tube_l, -0.006 - tube_l * 0.00338)])
        else:
            helium_tube = Polygon([(0, -0.006), (0, 0.006), (-tube_l, 0.006 + tube_l * 0.00338),
                                   (-tube_l, -0.006 - tube_l * 0.00338)])
        return helium_tube

    def trace_simulated_focus(self, n):
        st = SwarmTracer(self.PTL)
        swarm = st.initialize_Simulated_Collector_Focus_Swarm(n)

        # turn stuff around, project backwards
        if not self.leftwards:
            for particle in swarm:
                particle.pi[0] *= -1
                particle.obj_qi = particle.qi.copy()
                t = (-self.r1p2 * 1.5 + particle.qi[0]) / particle.pi[0]
                particle.qi[0] = self.long_off + 10 ** -4
                particle.qi[1] = particle.qi[1] + t * particle.pi[1]
                particle.qi[2] = particle.qi[2] + t * particle.pi[2]
        elif self.leftwards:
            for particle in swarm:
                particle.obj_qi = particle.qi.copy()
                t = (-self.r1p2 * 1.5 + particle.qi[0]) / particle.pi[0]
                particle.qi[0] = self.long_off - 10 ** -4
                particle.qi[1] = particle.qi[1] - t * particle.pi[1]
                particle.qi[2] = particle.qi[2] - t * particle.pi[2]

        swarm = st.trace_Swarm_Through_Lattice(swarm, 5e-6, 1.0, fastMode=False)
        return swarm

    def plot_trace(self):
        plt.plot(*self.helium_tube.exterior.xy)
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
        return 15 * self.r1p2 + 4 * self.l1 + 4 * self.l2 + self.d1 + self.d2

    def quality(self):
        area = self.he_tube_intersect.area
        im_qual = np.sum(self.image_quality() ** 2)
        size_p = self.size_penalty()
        return im_qual + np.exp(area * 5000) - 1 + size_p

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

    def clone_bumper(self):  # creates an identical bumper, but does not end the lattice
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
        return PTL


r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
l_norm = 0.5
d_norm = 0.5
L_norm = 0.05
phi_norm = 0.2

opt_norm = [0.53227244, 0.50382043, 0.78162133, -0.64865529, 0.34259073, 0.41882729, 0.17864541, 0.40304075]
opt_p = [opt_norm[0] * r_norm, opt_norm[1] * l_norm, opt_norm[2] * d_norm, opt_norm[3] * L_norm - 0.00045,
         opt_norm[4] * phi_norm + 0.020, opt_norm[5] * r_norm, opt_norm[6] * l_norm, opt_norm[7] * d_norm]

KevinBumper = Bumper(0.0266, 0.2519, 0.3908, -0.0329, 0.0885, 0.0209, 0.0893, 0.2015, 0,
                     leftwards=False, trace=False)
KevinBumperOpen = KevinBumper.clone_bumper()


if __name__ == '__main__':
    KevinBumperOpen.end_Lattice()
    KevinBumperOpen.show_Lattice()

    KevinBumper.swarm = KevinBumper.trace_simulated_focus(500)
    KevinBumper.obj_q, KevinBumper.obj_p, KevinBumper.im_q, KevinBumper.im_p = \
        KevinBumper.get_phase(save=False, name='6_15_2022_Kevin_Phase')
    print(KevinBumper.PTL.elList[0].ap, KevinBumper.PTL.elList[2].ap)
    print('position and angle alignment', KevinBumper.alignment())
    KevinBumper.plot_trace()
    print('quality', KevinBumper.quality())
    KevinBumper.plot_phase()

