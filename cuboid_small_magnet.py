from HalbachLensClass import Cuboid, billyHalbachCollectionWrapper
from scipy.spatial.transform import Rotation
from demag_functions import mesh_cuboid, apply_demag
from KevinBumperLens_meshed import call_coords_return_mesh, make_line
from constants import MAGNETIC_PERMEABILITY
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

phi = 0 * np.pi / 12
params = [np.array([1310., 0., 0.]), np.array([6.35, 6.35, 25.4]), np.array([0., 0., 0.]),
          Rotation.from_rotvec([0.0, 0.0, phi]), 1.05]  # this is for N42
M, dim, pos, orientation, mur = params
box = Cuboid(magnetization=M, dimension=dim, position=pos, orientation=orientation, mur=mur)
box_object = billyHalbachCollectionWrapper(box)
box_object.show()

print(np.linalg.norm(box_object.M_Vec(np.array([0, 0, 0]) + 1e-4)),
      np.linalg.norm(box_object.M_Vec(np.array([0.004, 0.006, 0]) + 1e-4)))


def plot_labels():
    plt.title('Holy Shit Magnets')
    plt.xlabel('x position $(m)$')
    plt.ylabel('y position $(m)$')


x_range = [-0.02, 0.02]
y_range = [-0.02, 0.02]
z_eval = 1e-4
step = 3.61e-4
B_pos, B = call_coords_return_mesh(x_range, y_range, z_eval, step, box_object.B_Vec)
gradB_pos, gradB = call_coords_return_mesh(x_range, y_range, z_eval, step, box_object.BNorm_Gradient)

plt.pcolormesh(B_pos[0], B_pos[1], B[0], cmap=plt.cm.get_cmap('magma'), shading='auto',)
cbar = plt.colorbar()
plot_labels()
cbar.set_label('B Field $(T)$', rotation=90, labelpad=15)
plt.show()

cmesh = plt.pcolormesh(gradB_pos[0], gradB_pos[1], gradB[0], cmap=plt.cm.get_cmap('magma'), shading='auto')
cbar = plt.colorbar(cmesh, extend='max')
plt.clim(-120, 120)
plot_labels()
cbar.set_label('X component of the Gradient of the B Field $(T/m)$', rotation=90, labelpad=15)
plt.show()

probe = make_line(np.array([0.0032, 0]), np.array([0.05, 0]), z_eval, 500)
probe_H = box_object.H_Vec(probe, units='A/m')
probe_gradB = box_object.BNorm_Gradient(probe)
plt.plot(probe_H[:, 0], 'o', color='navy')
plt.title('X component of the H field $(A/m)$')
plt.show()
plt.plot(probe_gradB[:, 0], 'o', color='navy')
plt.title('X component of the Gradient of the B field')
plt.show()
