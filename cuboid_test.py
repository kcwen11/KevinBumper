from HalbachLensClass import Cuboid, billyHalbachCollectionWrapper
from scipy.spatial.transform import Rotation
from demag_functions import mesh_cuboid, apply_demag
from KevinBumperLens import call_coords_return_mesh
from constants import MAGNETIC_PERMEABILITY
import numpy as np
import matplotlib.pyplot as plt

phi = np.pi / 12
params = [np.array([1310., 0., 0.]), np.array([12.7, 12.7, 148.60869239]), np.array([0., 0., 0.]),
          Rotation.from_rotvec([0.0, 0.0, phi]), 1.05]
M, dim, pos, orientation, mur = params
box = Cuboid(magnetization=M, dimension=dim, position=pos, orientation=orientation, mur=mur)
box_object = billyHalbachCollectionWrapper(box)
# box_object.show()

meshed = mesh_cuboid(box, (10, 10, 10))
# meshed.show()
for cuboid in meshed.sources_all:
    print(cuboid.position)
    cuboid.magnetization = np.array([0., 0., 0.])
# apply_demag(meshed)


def plot_labels():
    plt.title('Holy Shit Magnets')
    plt.xlabel('x position $(m)$')
    plt.ylabel('y position $(m)$')


x_range = [-0.02, 0.02]
y_range = [-0.02, 0.02]
z_eval = 1e-4
step = 3.6e-4
H_pos, H = call_coords_return_mesh(x_range, y_range, z_eval, step, box_object.M_Vec)
H_pos2, H2 = call_coords_return_mesh(x_range, y_range, z_eval, step, meshed.M_Vec)

plt.pcolormesh(H_pos[0], H_pos[1], H[0], cmap=plt.cm.get_cmap('magma'), shading='auto',)
cbar = plt.colorbar()
plt.clim(9e5, 1.01e6)
plot_labels()
cbar.set_label('H Field in x $(kOe)$', rotation=90, labelpad=15)
plt.show()

plt.pcolormesh(H_pos2[0], H_pos2[1], H2[0], cmap=plt.cm.get_cmap('magma'), shading='auto',)
cbar2 = plt.colorbar()
plt.clim(9e5, 1.01e6)
plot_labels()
cbar2.set_label('H Field in x $(kOe)$', rotation=90, labelpad=15)
plt.show()
