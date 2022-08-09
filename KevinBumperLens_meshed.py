from KevinBumperClass_grades import KevinBumper
from HalbachLensClass import HalbachLens
import matplotlib.pyplot as plt
import numpy as np


def make_domain(x, y, dx):
    x_list = np.arange(x[0], x[1], dx)
    y_list = np.arange(y[0], y[1], dx)
    return np.meshgrid(x_list, y_list)


def make_line(r1, r2, z_coord, n, perp_axis='z'):
    line = np.linspace(r1, r2, n)
    z = np.array([[z_coord] * n]).T
    if perp_axis == 'z':
        return np.concatenate((line, z), axis=1)
    elif perp_axis == 'x':
        return np.concatenate((z, line), axis=1)


def mesh_to_list_2d(mesh, z_coord):
    x, y = mesh
    stack = np.vstack([x.ravel(), y.ravel()])
    z = np.array([stack[0] * 0 + z_coord])
    return np.concatenate((stack, z), axis=0).T


def call_coords_return_mesh(x, y, z0, dx, func, perp_axis='z'):  # evaluates func in the z = z0 plane
    mesh = make_domain(x, y, dx)
    coords = mesh_to_list_2d(mesh, z0)
    s = slice(2, 0, -1)
    if perp_axis == 'x':
        coords[:, [0, 1, 2]] = coords[:, [2, 0, 1]]
        s = slice(1, None)
    output = func(coords)
    if len(np.shape(output)) == 1:
        return mesh, np.reshape(output, np.shape(mesh)[s])
    elif len(np.shape(output)) == 2 and np.shape(output)[1] == 3:
        out_x = np.reshape(output[:, 0:1], np.shape(mesh)[s])
        out_y = np.reshape(output[:, 1:2], np.shape(mesh)[s])
        out_z = np.reshape(output[:, 2:3], np.shape(mesh)[s])
        return mesh, np.array([out_x, out_y, out_z])
    else:
        print('what the fuck!!!!!!!!!')


if __name__ == '__main__':
    the_lens = KevinBumper.PTL.elList[0]
    # lens = KevinBumper.PTL.elList[0].lens
    # lens.show()

    kevin_mag_grade = KevinBumper.PTL.magnetGrade[0] if len(the_lens.rpLayers) == 2 else KevinBumper.PTL.magnetGrade[1]
    print(kevin_mag_grade)
    useStandardMagnetErrors = False
    numDisks = 1 if not useStandardMagnetErrors else the_lens.get_Num_Lens_Slices()
    lens = HalbachLens(the_lens.rpLayers, the_lens.magnetWidths, the_lens.Lm, kevin_mag_grade, mesh=False)
    lens.show()

    x_range = [-0.06, 0.06]
    y_range = [-0.06, 0.06]
    z_eval = 1e-4
    step = 6e-4  # 1.2e-4
    div = 52
    Hk = 13

    print('a')
    # B_pos, B = call_coords_return_mesh(x_range, y_range, z_eval, step, lens.B_Vec)
    H_pos, H = call_coords_return_mesh(x_range, y_range, z_eval, step, lens.H_Vec)
    # mag_pos, Bmag = call_coords_return_mesh(x_range, y_range, z_eval, step, lens.BNorm)

    print('b')
    probe = make_line(np.array([0, 0]), np.array([0, 0.06]), z_eval, div)
    probe2 = make_line(np.array([3.1e-3, 0]), np.array([3.1e-3, 0.06]), z_eval, div)
    probe3 = make_line(np.array([6.2e-3, 0]), np.array([6.2e-3, 0.06]), z_eval, div)
    probe4 = make_line(np.array([9.3e-3, 0]), np.array([9.3e-3, 0.06]), z_eval, div)
    probe_H, probe_H2, probe_H3, probe_H4 = lens.H_Vec(probe), lens.H_Vec(probe2), lens.H_Vec(probe3), lens.H_Vec(probe4)

    print('c')
    plt.pcolormesh(H_pos[0], H_pos[1], H[0], cmap=plt.cm.get_cmap('magma'), shading='auto',)
    plt.vlines([0, 3.1e-3, 6.2e-3, 9.3e-3], ymin=0, ymax=0.059, linestyles='dotted', color='black')
    cbar = plt.colorbar()
    plt.contour(H_pos[0], H_pos[1], H[0] > Hk, 1, colors='k')
    plt.title('Holy Shit Magnets')
    plt.xlabel('x position $(m)$')
    plt.ylabel('y position $(m)$')
    cbar.set_label('H Field in x $(kOe)$', rotation=90, labelpad=15)
    plt.show()

    plt.plot(probe_H[:, 0], 'o', color='crimson')
    plt.plot(probe_H2[:, 0], 'o', color='yellowgreen')
    plt.plot(probe_H3[:, 0], 'o', color='forestgreen')
    plt.plot(probe_H4[:, 0], 'o', color='navy')
    plt.title('H field in x along a transverse line')
    plt.show()

    x_eval = 0.0063
    y_range = [-0.06, 0.06]
    z_range = [-0.1, 0.1]
    H_pos2, H2 = call_coords_return_mesh(y_range, z_range, x_eval, step, lens.H_Vec, perp_axis='x')

    r = 0.0242
    div = 150
    z_max = 0.1
    probe = make_line(np.array([r, 0]), np.array([r, z_max]), x_eval, div, perp_axis='x')
    probe2 = make_line(np.array([r + 3.15e-3, 0]), np.array([r + 3.15e-3, z_max]), x_eval, div, perp_axis='x')
    probe3 = make_line(np.array([r + 6.3e-3, 0]), np.array([r + 6.3e-3, z_max]), x_eval, div, perp_axis='x')
    probe_H, probe_H2, probe_H3 = lens.H_Vec(probe), lens.H_Vec(probe2), lens.H_Vec(probe3)

    plt.pcolormesh(H_pos2[0], H_pos2[1], H2[0], cmap=plt.cm.get_cmap('magma'), shading='auto',)
    plt.vlines(np.array([0, 3.15e-3, 6.3e-3]) + r, ymin=0, ymax=0.099, linestyles='dotted', color='black')
    plt.hlines(np.array([3 * 0.0254, 3.75 * 0.0254]), xmin=r, xmax=r + 2 * 6.3e-3, linestyles='dotted', color='black')
    cbar = plt.colorbar()
    plt.contour(H_pos2[0], H_pos2[1], H2[0] > Hk, 1, colors='k')
    plt.title('Holy Shit Magnets')
    plt.xlabel('y position $(m)$')
    plt.ylabel('z position $(m)$')
    cbar.set_label('H Field in x $(kOe)$', rotation=90, labelpad=15)
    plt.show()

    plt.plot(probe_H[:, 0], 'o', color='crimson')
    plt.plot(probe_H2[:, 0], 'o', color='yellowgreen')
    plt.plot(probe_H3[:, 0], 'o', color='forestgreen')
    plt.vlines([3 * 0.0254 / z_max * div, 3.75 * 0.0254 / z_max * div], ymin=0, ymax=16, linestyles='dotted', color='black')
    plt.title('H field in x along a longitudinal line')
    plt.show()
