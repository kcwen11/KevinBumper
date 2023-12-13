from HalbachLensClass import Cuboid, billyHalbachCollectionWrapper
from scipy.spatial.transform import Rotation
from demag_functions import mesh_cuboid, apply_demag
from KevinBumperLens_meshed import call_coords_return_mesh, cull_demag
from constants import MAGNETIC_PERMEABILITY
import numpy as np
import matplotlib.pyplot as plt

phi = 0
params = [np.array([1310., 0., 0.]), np.array([5, 20, 20]), np.array([0., 0., 0.]),
          Rotation.from_rotvec([0.0, 0.0, phi]), 1.05]
M, dim, pos, orientation, mur = params
box = Cuboid(magnetization=M, dimension=dim, position=pos, orientation=orientation, mur=mur)
box_object = billyHalbachCollectionWrapper(box)
# box_object.show()

meshed = mesh_cuboid(box, (10, 10, 10))
meshed.show()
apply_demag(meshed)
print(np.linalg.norm(meshed.M_Vec(np.array([0, 0, 0]) + 1e-4)),
      np.linalg.norm(meshed.M_Vec(np.array([0.004, 0.006, 0]) + 1e-4)))


def plot_labels():
    plt.title('Magnetic Field Magnitude')
    plt.xlabel('x position $(m)$')
    plt.ylabel('y position $(m)$')


cull_demag(meshed, 12)

x_range = [-0.02, 0.02]
y_range = [-0.02, 0.02]
z_eval = 1e-4
step = 3.6e-4
M_pos, M = call_coords_return_mesh(x_range, y_range, z_eval, step, box_object.BNorm)
M_pos2, M2 = call_coords_return_mesh(x_range, y_range, z_eval, step, meshed.BNorm)

plt.pcolormesh(M_pos[0], M_pos[1], M, cmap=plt.cm.get_cmap('magma'), shading='auto',)
cbar = plt.colorbar()
# plt.clim(1e6, 1.05e6)
plot_labels()
cbar.set_label('B Field magnitude $(T)$', rotation=90, labelpad=15)
plt.show()

plt.pcolormesh(M_pos2[0], M_pos2[1], M2, cmap=plt.cm.get_cmap('magma'), shading='auto',)
cbar2 = plt.colorbar()
# plt.clim(1e6, 1.05e6)
plot_labels()
cbar2.set_label('B Field magnitude $(T)$', rotation=90, labelpad=15)
plt.show()
