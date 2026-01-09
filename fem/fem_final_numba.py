import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from numba import njit


@njit(cache=True)
def _compute_element_forces_numba(p0, p1, p2, x_inv, volume, young_modulus, poisson_ratio, use_cauchy):
    """Return 3x2 element force matrix for a single triangle."""
    forces = np.zeros((3, 2), dtype=np.float64)
    if volume == 0.0:
        return forces

    col0x = p1[0] - p0[0]
    col0y = p1[1] - p0[1]
    col1x = p2[0] - p0[0]
    col1y = p2[1] - p0[1]

    p00 = col0x * x_inv[0, 0] + col1x * x_inv[1, 0]
    p01 = col0x * x_inv[0, 1] + col1x * x_inv[1, 1]
    p10 = col0y * x_inv[0, 0] + col1y * x_inv[1, 0]
    p11 = col0y * x_inv[0, 1] + col1y * x_inv[1, 1]

    grad_u00 = p00 - 1.0
    grad_u11 = p11 - 1.0
    grad_u01 = p01
    grad_u10 = p10

    if use_cauchy:
        strain00 = grad_u00
        strain11 = grad_u11
        strain01 = 0.5 * (grad_u01 + grad_u10)
    else:
        strain00 = grad_u00 + 0.5 * (grad_u00 * grad_u00 + grad_u10 * grad_u10)
        strain11 = grad_u11 + 0.5 * (grad_u01 * grad_u01 + grad_u11 * grad_u11)
        strain01 = 0.5 * (
            grad_u01 + grad_u10 + grad_u00 * grad_u01 + grad_u10 * grad_u11
        )

    coeff = young_modulus / (1.0 - poisson_ratio * poisson_ratio)
    shear_coeff = coeff * (1.0 - poisson_ratio) * 0.5

    s0 = coeff * strain00 + coeff * poisson_ratio * strain11
    s1 = coeff * poisson_ratio * strain00 + coeff * strain11
    s2 = shear_coeff * strain01

    grad1x = x_inv[0, 0]
    grad1y = x_inv[0, 1]
    grad2x = x_inv[1, 0]
    grad2y = x_inv[1, 1]
    grad0x = -(grad1x + grad2x)
    grad0y = -(grad1y + grad2y)

    grads = (
        (grad0x, grad0y),
        (grad1x, grad1y),
        (grad2x, grad2y),
    )

    for idx in range(3):
        gradx, grady = grads[idx]
        fx = -(volume) * (s0 * gradx + s2 * grady)
        fy = -(volume) * (s2 * gradx + s1 * grady)
        forces[idx, 0] = fx
        forces[idx, 1] = fy

    return forces


@njit(cache=True)
def _assemble_forces_numba(
    positions,
    tri_indices,
    tri_x_inv,
    tri_volumes,
    masses,
    fixed_mask,
    gravity,
    young_modulus,
    poisson_ratio,
    use_cauchy,
    apply_constraints,
):
    """Compute the global force vector for the current configuration."""
    n = positions.shape[0]
    f_global = np.zeros((n, 2), dtype=np.float64)

    # for i in range(n):
    #     f_global[i, 1] -= masses[i] * gravity

    tri_count = tri_indices.shape[0]
    for tri_idx in range(tri_count):
        volume = tri_volumes[tri_idx]
        if volume == 0.0:
            continue

        indices = tri_indices[tri_idx]
        i0 = indices[0]
        i1 = indices[1]
        i2 = indices[2]

        elem_forces = _compute_element_forces_numba(
            positions[i0],
            positions[i1],
            positions[i2],
            tri_x_inv[tri_idx],
            volume,
            young_modulus,
            poisson_ratio,
            use_cauchy,
        )

        f_global[i0, 0] += elem_forces[0, 0]
        f_global[i0, 1] += elem_forces[0, 1]
        f_global[i1, 0] += elem_forces[1, 0]
        f_global[i1, 1] += elem_forces[1, 1]
        f_global[i2, 0] += elem_forces[2, 0]
        f_global[i2, 1] += elem_forces[2, 1]

    if apply_constraints:
        for i in range(n):
            if fixed_mask[i]:
                f_global[i, 0] = 0.0
                f_global[i, 1] = 0.0

    return f_global


@njit(cache=True)
def _integrate_particles_numba(
    positions,
    velocities,
    masses,
    fixed_mask,
    forces,
    dt,
    damping,
    ground_y,
    collision_coeff,
):
    """Semi-implicit Euler integration with simple ground collisions."""
    n = positions.shape[0]
    for i in range(n):
        if fixed_mask[i]:
            velocities[i, 0] = 0.0
            velocities[i, 1] = 0.0
            continue

        inv_mass = 1.0 / masses[i]
        velocities[i, 0] = damping * (velocities[i, 0] + forces[i, 0] * inv_mass * dt)
        velocities[i, 1] = damping * (velocities[i, 1] + forces[i, 1] * inv_mass * dt)
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt

        if positions[i, 1] < ground_y:
            positions[i, 1] = ground_y
            if collision_coeff == 0.0:
                velocities[i, 1] = 0.0
            else:
                velocities[i, 1] *= -collision_coeff


class FEMSoftBody:

    def __init__(self, young_modulus=5000, poisson_ratio=0.3, gravity=100,
                 fix_top=True, strain_type='green'):
        self.E = young_modulus
        self.nu = poisson_ratio
        self.gravity = gravity
        self.fix_top = fix_top
        self.strain_type = strain_type  # 'cauchy' or 'green'

        self.particles = []
        self.triangles = []
        self.positions = None
        self.positions0 = None
        self.velocities = None
        self.masses = None
        self.fixed_mask = None
        self.tri_indices_array = None
        self.tri_x_inv_array = None
        self.tri_volume_array = None
        self.last_force_matrix = None
        self.time = 0.0

        self.create_mesh()
        self.precompute_rest_config()

    def create_mesh(self):
        """Create a square grid mesh of triangular elements."""
        grid_size = 15
        spacing = 0.1
        offset_x = 2.0
        offset_y = 1.0
        left_wall = 0.5
        right_wall = 5.5
        ground_y = 2.0

        self.particles = []
        for j in range(grid_size):
            for i in range(grid_size):
                x = (offset_x + i * spacing) / (spacing * grid_size)
                y = (offset_y + j * spacing) / (spacing * grid_size)
                pos = np.array([x, y], dtype=np.float64)
                self.particles.append({
                    'pos': pos.copy(),
                    'pos0': pos.copy(),
                    'vel': np.zeros(2, dtype=np.float64),
                    'mass': 1.0,
                    'fixed': (j == 0) and self.fix_top
                })

        self._initialize_particle_arrays()
        centroid = self.positions.mean(axis=0)
        self.positions -= centroid
        self.positions0 -= centroid
        self.positions[:, 1] *= -1
        self.positions0[:, 1] *= -1

        self.left_wall = left_wall - centroid[0]
        self.right_wall = right_wall - centroid[0]
        self.ground_y = -(ground_y - centroid[1])

        # Build triangle connectivity
        self.triangles = []
        for j in range(grid_size - 1):
            for i in range(grid_size - 1):
                idx = j * grid_size + i
                self.triangles.append({
                    'indices': [idx, idx + 1, idx + grid_size],
                    'X_inv': None,
                    'volume': 0
                })
                self.triangles.append({
                    'indices': [idx + 1, idx + grid_size + 1, idx + grid_size],
                    'X_inv': None,
                    'volume': 0
                })

    def precompute_rest_config(self):
        """Compute rest configuration matrices and areas."""
        x_inv_list = []
        volume_list = []
        for tri in self.triangles:
            i0, i1, i2 = tri['indices']
            x0 = self.particles[i0]['pos0']
            x1 = self.particles[i1]['pos0']
            x2 = self.particles[i2]['pos0']

            X = np.column_stack([x1 - x0, x2 - x0])
            volume = 0.5 * abs(np.linalg.det(X))

            if volume < 1e-12:
                tri['X_inv'] = np.eye(2)
                tri['volume'] = 0
            else:
                tri['X_inv'] = np.linalg.inv(X)
                tri['volume'] = volume
            x_inv_list.append(tri['X_inv'])
            volume_list.append(tri['volume'])

        self.tri_indices_array = np.ascontiguousarray(
            np.array([tri['indices'] for tri in self.triangles], dtype=np.int64)
        )
        self.tri_x_inv_array = np.ascontiguousarray(np.array(x_inv_list, dtype=np.float64))
        self.tri_volume_array = np.ascontiguousarray(np.array(volume_list, dtype=np.float64))

    def compute_deformation_gradient(self, tri):
        """Deformation gradient F = [p1 - p0, p2 - p0] * X_inv"""
        i0, i1, i2 = tri['indices']
        p0 = self.particles[i0]['pos']
        p1 = self.particles[i1]['pos']
        p2 = self.particles[i2]['pos']
        P = np.column_stack([p1 - p0, p2 - p0]) @ tri['X_inv']
        return P

    def compute_strain(self, P):
        I = np.eye(2)
        grad_u = P - I
        if self.strain_type == 'cauchy':
            strain = 0.5 * (grad_u + grad_u.T)
        else:  # Green strain
            strain = 0.5 * (grad_u + grad_u.T + grad_u.T @ grad_u)
        return strain

    def compute_stress(self, strain):
        # TODO: Check this
        """Plane stress constitutive relation."""
        E, nu = self.E, self.nu
        C = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        e = np.array([strain[0, 0], strain[1, 1], strain[0, 1]])
        s = C @ e
        stress = np.array([[s[0], s[2]], [s[2], s[1]]])
        return stress

    def compute_element_forces(self, tri):
        """Return 3x2 element force matrix (per node/per axis)."""
        if tri['volume'] == 0:
            return np.zeros((3, 2))

        # Important part
        P = self.compute_deformation_gradient(tri)
        strain = self.compute_strain(P)
        stress = self.compute_stress(strain)

        # Shape function gradients in reference space
        # TODO: Check this
        X_inv = tri['X_inv']
        dphi = X_inv.T
        grad0 = -dphi[:, 0] - dphi[:, 1]
        grad1 = dphi[:, 0]
        grad2 = dphi[:, 1]
        grads = [grad0, grad1, grad2]

    
        # Elastic force matrix per node (3 nodes x 2 components)
        f = np.zeros((3, 2))
        for a, gradNi in enumerate(grads):
            fa = -tri['volume'] * (stress @ gradNi)
            f[a] = fa

        return f

    def assemble_global_matrices(self, apply_constraints=True):
        """Assemble global force vector and stiffness matrix."""
        if self.tri_indices_array is None:
            return np.zeros((len(self.particles), 2), dtype=np.float64)

        f_global = _assemble_forces_numba(
            self.positions,
            self.tri_indices_array,
            self.tri_x_inv_array,
            self.tri_volume_array,
            self.masses,
            self.fixed_mask,
            self.gravity,
            self.E,
            self.nu,
            self.strain_type == 'cauchy',
            False,
        )

        if self.time < 0.02:
            positions = self.positions
            center = np.mean(positions, axis=0)
            rotation_strength = 200.0
            for i in range(positions.shape[0]):
                r = positions[i] - center
                tangential = np.array([-r[1], r[0]])
                f_global[i] += rotation_strength * self.masses[i] * tangential

        if apply_constraints:
            f_global[self.fixed_mask] = 0.0

        return f_global


    def update(self, dt):
        """Update simulation with explicit Euler integration."""
        self.time += dt
        
        # Explicit Euler integration
        self.last_force_matrix = self.assemble_global_matrices(apply_constraints=True)
        damping = 1.0  # 0.995
        collision_coeff = 0.0  # Ground collisions currently absorb all velocity.
        _integrate_particles_numba(
            self.positions,
            self.velocities,
            self.masses,
            self.fixed_mask,
            self.last_force_matrix,
            dt,
            damping,
            self.ground_y,
            collision_coeff,
        )

    def get_triangle_positions(self):
        return [np.array([self.particles[i]['pos'] for i in tri['indices']])
                for tri in self.triangles]

    def get_particle_positions(self):
        return np.array([p['pos'] for p in self.particles])

    def _initialize_particle_arrays(self):
        """Create contiguous arrays for particle data and keep dict views in sync."""
        positions = np.array([p['pos'] for p in self.particles], dtype=np.float64)
        rest_positions = np.array([p['pos0'] for p in self.particles], dtype=np.float64)
        velocities = np.array([p['vel'] for p in self.particles], dtype=np.float64)
        masses = np.array([p['mass'] for p in self.particles], dtype=np.float64)
        fixed_mask = np.array([p['fixed'] for p in self.particles], dtype=bool)

        self.positions = np.ascontiguousarray(positions)
        self.positions0 = np.ascontiguousarray(rest_positions)
        self.velocities = np.ascontiguousarray(velocities)
        self.masses = np.ascontiguousarray(masses)
        self.fixed_mask = fixed_mask

        for idx, particle in enumerate(self.particles):
            particle['pos'] = self.positions[idx]
            particle['pos0'] = self.positions0[idx]
            particle['vel'] = self.velocities[idx]


def main():
    sim = FEMSoftBody(
        young_modulus=15000,
        poisson_ratio=0.3, #0.3,
        gravity=30,
        fix_top=False,
        strain_type='green', #cauchy', #'green',
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    initial_pos = sim.get_particle_positions()
    x_extent = np.max(np.abs(initial_pos[:, 0])) + 1.0
    y_extent = np.max(np.abs(initial_pos[:, 1])) + 1.0
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(-y_extent, y_extent)
    ax.axhline(y=sim.ground_y, color='k', lw=2)
    ax.set_title("2D FEM Soft Body Simulation")

    patches = []
    for _ in sim.triangles:
        patch = Polygon([[0, 0], [0, 0], [0, 0]], fc='lightblue', ec='blue', alpha=0.6)
        ax.add_patch(patch)
        patches.append(patch)

    pts, = ax.plot([], [], 'o', color='darkblue', ms=4)
    fixed_pts, = ax.plot([], [], 'o', color='red', ms=6)
    zero_forces = np.zeros_like(initial_pos)
    force_quiver = ax.quiver(initial_pos[:, 0], initial_pos[:, 1],
                             zero_forces[:, 0], zero_forces[:, 1],
                             color='orange', angles='xy', scale_units='xy',
                             scale=1.0, width=0.003, alpha=0.7)
    txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    frame = [0]
    force_scale = 0.003  # pretty-print forces without overwhelming the plot

    def init():
        force_quiver.set_offsets(initial_pos)
        force_quiver.set_UVC(zero_forces[:, 0], zero_forces[:, 1])
        return patches + [pts, fixed_pts, force_quiver, txt]

    def animate(_):
        dt = 0.01
        substeps = 1000
        sdt = dt / substeps
        for _ in range(substeps):
            sim.update(sdt)

        tri_pos = sim.get_triangle_positions()
        for patch, verts in zip(patches, tri_pos):
            patch.set_xy(verts)

        pos = sim.get_particle_positions()
        mask = np.array([p['fixed'] for p in sim.particles])
        pts.set_data(pos[~mask, 0], pos[~mask, 1])
        fixed_pts.set_data(pos[mask, 0], pos[mask, 1])
        forces = np.array(sim.last_force_matrix)
        scaled_forces = forces * force_scale
        force_quiver.set_offsets(pos)
        force_quiver.set_UVC(scaled_forces[:, 0], scaled_forces[:, 1])
        frame[0] += 1
        txt.set_text(f"Frame: {frame[0]}")
        return patches + [pts, fixed_pts, force_quiver, txt]

    anim = FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
