from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer
from SwarmTracerClass import SwarmTracer
from ParticleClass import Particle
from helperTools import *
from shapely.geometry import Polygon

r1p1 = 0.0275
l1 = 0.295 - r1p1 / 4
d1 = 0.4
L = -0.037 + 0.003
phi = 0.08
r2 = 0.022
l2 = 0.0998 - r2 / 4
d2 = 0.25
start = -0.013

magwidth1 = r1p1 * np.tan(2 * np.pi / 24) * 2
r1p2 = r1p1+magwidth1
magwidth2 = r1p2 * np.tan(2 * np.pi / 24) * 2

delta_x = r2 * 1.5 * np.cos(phi)
delta_y = r2 * 1.5 * np.sin(phi)
a1 = np.tan((L - delta_y) / (d1 - r1p2 * 1.5 - delta_x))
a2 = a1 + phi
d_fix = np.sqrt((L - delta_y) ** 2 + (d1 - r1p2 * 1.5 - delta_x) ** 2)
l1_plus_fringe = l1 + r1p2 * 3
l2_plus_fringe = l2 + r2 * 3


PTL=ParticleTracerLattice(latticeType='injector',initialAngle=0.0,initialLocation=(-r1p2 * 1.5, start))
PTL.add_Halbach_Lens_Sim((r1p1,r1p2), l1_plus_fringe, magnetWidth=(magwidth1,magwidth2))
PTL.add_Drift(d_fix, .04, inputTiltAngle=-a1, outputTiltAngle=-a2)
PTL.add_Halbach_Lens_Sim(r2, l2_plus_fringe)
PTL.add_Drift(d2, .04)
PTL.end_Lattice()


st=SwarmTracer(PTL)
swarm=st.initialize_Simulated_Collector_Focus_Swarm(250)

object_swarm = swarm

#turn stuff around, project backwards
for particle in swarm:
    particle.pi[0] *= -1
    particle.obj_qi = particle.qi[1]
    t = (-r1p2 * 1.5 + particle.qi[0]) / particle.pi[0]
    particle.qi[0] = -r1p2 * 1.5 + 10 ** -4
    particle.qi[1] = particle.qi[1] + t * particle.pi[1]
    particle.qi[2] = particle.qi[2] + t * particle.pi[2]

swarm = st.trace_Swarm_Through_Lattice(swarm,5e-6,1.0,fastMode=False)

tube_l = 1.2
helium_tube = Polygon([(0, -0.0075), (0, 0.0075), (tube_l, 0.0075 + tube_l * 4.214e-5),
                       (tube_l, -0.0075 - tube_l * 4.214e-5)])

plt.plot(*helium_tube.exterior.xy)
PTL.show_Lattice(plotOuter=True,plotInner=True,swarm=swarm,showTraceLines=True,traceLineAlpha=.25)

p_avg = 0
q_avg = 0
living_count = 0
for particle in swarm:
    if not particle.clipped:
        living_count += 1
        p_avg += particle.pf[1]
        q_avg += particle.qf[1]
p_avg /= living_count
q_avg /= living_count

for particle in swarm:
    plt.plot(particle.obj_qi, particle.pi[1], 'bo')
    plt.plot(particle.qf[1] - q_avg, particle.pf[1] - p_avg, 'go')
plt.show()


