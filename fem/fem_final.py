import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numba import jit, prange
import numba


# Numba-optimized helper functions
@jit(nopython=True, cache=True)
def compute_deformation_gradient_numba(p0, p1, p2, X_inv):
    """Compute deformation gradient F = [p1 - p0, p2 - p0] * X_inv"""
    P = np.empty((2, 2))
    P[:, 0] = p1 - p0
    P[:, 1] = p2 - p0
    return P @ X_inv


@jit(nopython=True, cache=True)
def compute_strain_numba(P, strain_type):
    """Compute strain from deformation gradient"""
    grad_u = P - np.eye(2)
    if strain_type == 0:  # Cauchy
        strain = 0.5 * (grad_u + grad_u.T)
    else:  # Green
        strain = 0.5 * (grad_u + grad_u.T + grad_u.T @ grad_u)
    return strain


@jit(nopython=True, cache=True)
def compute_stress_numba(strain, E, nu):
    """Plane stress constitutive relation"""
    C_factor = E / (1 - nu**2)
    e00 = strain[0, 0]
    e11 = strain[1, 1]
    e01 = strain[0, 1]
    
    s00 = C_factor * (e00 + nu * e11)
    s11 = C_factor * (e11 + nu * e00)
    s01 = C_factor * ((1 - nu) / 2) * e01
    
    stress = np.array([[s00, s01], [s01, s11]])
    return stress


@jit(nopython=True, cache=True)
def compute_shape_gradients(X_inv):
    """Compute shape function gradients"""
    dphi = X_inv.T
    grad0 = -dphi[:, 0] - dphi[:, 1]
    grad1 = dphi[:, 0]
    grad2 = dphi[:, 1]
    return grad0, grad1, grad2


@jit(nopython=True, cache=True)
def compute_element_forces_numba(p0, p1, p2, X_inv, volume, E, nu, strain_type):
    """Compute element forces with Numba optimization"""
    if volume < 1e-12:
        return np.zeros((3, 2))
    
    P = compute_deformation_gradient_numba(p0, p1, p2, X_inv)
    strain = compute_strain_numba(P, strain_type)
    stress = compute_stress_numba(strain, E, nu)
    
    grad0, grad1, grad2 = compute_shape_gradients(X_inv)
    
    f = np.empty((3, 2))
    f[0] = -volume * (stress @ grad0)
    f[1] = -volume * (stress @ grad1)
    f[2] = -volume * (stress @ grad2)
    
    return f


@jit(nopython=True, cache=True)
def compute_B_matrix(grad0, grad1, grad2):
    """Build strain-displacement B matrix"""
    B = np.zeros((3, 6))
    for i in range(3):
        if i == 0:
            grad = grad0
        elif i == 1:
            grad = grad1
        else:
            grad = grad2
        
        B[0, 2*i] = grad[0]      # ε_xx from u_x
        B[1, 2*i+1] = grad[1]    # ε_yy from u_y
        B[2, 2*i] = grad[1]      # γ_xy from u_x
        B[2, 2*i+1] = grad[0]    # γ_xy from u_y
    return B


@jit(nopython=True, cache=True)
def compute_element_stiffness_numba(X_inv, volume, E, nu):
    """Compute element stiffness matrix with Numba"""
    if volume < 1e-12:
        return np.zeros((6, 6))
    
    # Material matrix
    C_factor = E / (1 - nu**2)
    C = np.zeros((3, 3))
    C[0, 0] = C_factor
    C[0, 1] = C_factor * nu
    C[1, 0] = C_factor * nu
    C[1, 1] = C_factor
    C[2, 2] = C_factor * (1 - nu) / 2
    
    grad0, grad1, grad2 = compute_shape_gradients(X_inv)
    B = compute_B_matrix(grad0, grad1, grad2)
    
    # K_e = V * B^T * C * B
    K_e = volume * (B.T @ C @ B)
    return K_e


@jit(nopython=True, parallel=True, cache=True)
def compute_all_element_forces(positions, indices_array, X_inv_array, volumes, E, nu, strain_type):
    """Compute forces for all elements in parallel"""
    n_elements = len(indices_array)
    forces = np.zeros((n_elements, 3, 2))
    
    for i in prange(n_elements):
        if volumes[i] > 1e-12:
            idx = indices_array[i]
            p0 = positions[idx[0]]
            p1 = positions[idx[1]]
            p2 = positions[idx[2]]
            forces[i] = compute_element_forces_numba(
                p0, p1, p2, X_inv_array[i], volumes[i], E, nu, strain_type
            )
    return forces


@jit(nopython=True, parallel=True, cache=True)
def compute_all_element_stiffness(X_inv_array, volumes, E, nu):
    """Compute stiffness matrices for all elements in parallel"""
    n_elements = len(volumes)
    stiffness = np.zeros((n_elements, 6, 6))
    
    for i in prange(n_elements):
        if volumes[i] > 1e-12:
            stiffness[i] = compute_element_stiffness_numba(
                X_inv_array[i], volumes[i], E, nu
            )
    return stiffness


@jit(nopython=True, cache=True)
def compute_element_stiffness_current_numba(p0, p1, p2, X_inv, volume, E, nu):
    """Compute element stiffness matrix for CURRENT configuration with Numba.
    This is key for proper implicit Euler - K must be computed at current positions."""
    if volume < 1e-12:
        return np.zeros((6, 6))
    
    # Compute deformation gradient for current configuration
    P = np.empty((2, 2))
    P[:, 0] = p1 - p0
    P[:, 1] = p2 - p0
    F = P @ X_inv
    
    # Current volume (determinant of F * original volume)
    detF = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    current_volume = volume * abs(detF)
    
    if current_volume < 1e-12:
        return np.zeros((6, 6))
    
    # Material matrix (plane stress)
    C_factor = E / (1 - nu**2)
    C = np.zeros((3, 3))
    C[0, 0] = C_factor
    C[0, 1] = C_factor * nu
    C[1, 0] = C_factor * nu
    C[1, 1] = C_factor
    C[2, 2] = C_factor * (1 - nu) / 2
    
    # Shape function gradients in current configuration
    # Need to transform reference gradients through inverse of F
    dphi = X_inv.T
    grad0_ref = -dphi[:, 0] - dphi[:, 1]
    grad1_ref = dphi[:, 0]
    grad2_ref = dphi[:, 1]
    
    # Build B matrix using reference gradients (standard for Total Lagrangian)
    B = np.zeros((3, 6))
    grads = [grad0_ref, grad1_ref, grad2_ref]
    for i in range(3):
        grad = grads[i]
        B[0, 2*i] = grad[0]
        B[1, 2*i+1] = grad[1]
        B[2, 2*i] = grad[1]
        B[2, 2*i+1] = grad[0]
    
    # Element stiffness: K_e = V * B^T * C * B
    K_e = volume * (B.T @ C @ B)
    return K_e


@jit(nopython=True, parallel=True, cache=True)
def compute_all_element_stiffness_current(positions, indices_array, X_inv_array, volumes, E, nu):
    """Compute stiffness matrices for all elements at current configuration in parallel"""
    n_elements = len(volumes)
    stiffness = np.zeros((n_elements, 6, 6))
    
    for i in prange(n_elements):
        if volumes[i] > 1e-12:
            idx = indices_array[i]
            p0 = positions[idx[0]]
            p1 = positions[idx[1]]
            p2 = positions[idx[2]]
            stiffness[i] = compute_element_stiffness_current_numba(
                p0, p1, p2, X_inv_array[i], volumes[i], E, nu
            )
    return stiffness


class FEMSoftBody:

    def __init__(self, young_modulus=5000, poisson_ratio=0.3, gravity=100,
                 fix_top=True, strain_type='green'):
        self.E = young_modulus
        self.nu = poisson_ratio
        self.gravity = gravity
        self.fix_top = fix_top
        self.strain_type = strain_type  # 'cauchy' or 'green'
        self.strain_type_num = 1 if strain_type == 'green' else 0  # For Numba

        self.particles = []
        self.triangles = []
        self.last_force_matrix = None
        self.time = 0.0

        self.create_mesh()
        self.precompute_rest_config()
        self.prepare_numba_arrays()

    def create_mesh(self):
        """Create a square grid mesh of triangular elements."""
        grid_size = 15
        spacing = 0.1
        offset_x = 2.0
        offset_y = 1.0
        left_wall = 0.5
        right_wall = 5.5
        ground_y = 5.0

        self.particles = []
        for j in range(grid_size):
            for i in range(grid_size):
                x = (offset_x + i * spacing) / (spacing * grid_size)
                y = (offset_y + j * spacing) / (spacing * grid_size)
                self.particles.append({
                    'pos': np.array([x, y], dtype=float),
                    'pos0': np.array([x, y], dtype=float),
                    'vel': np.zeros(2),
                    'mass': 1.0,
                    'fixed': (j == 0) and self.fix_top
                })

        positions = np.array([p['pos'] for p in self.particles])
        centroid = positions.mean(axis=0)
        for p in self.particles:
            p['pos'] -= centroid
            p['pos0'] -= centroid
            p['pos'][1] *= -1
            p['pos0'][1] *= -1

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
    
    def prepare_numba_arrays(self):
        """Prepare contiguous numpy arrays for Numba processing"""
        n_triangles = len(self.triangles)
        self.indices_array = np.array([tri['indices'] for tri in self.triangles], dtype=np.int32)
        self.X_inv_array = np.array([tri['X_inv'] for tri in self.triangles], dtype=np.float64)
        self.volumes = np.array([tri['volume'] for tri in self.triangles], dtype=np.float64)

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

        i0, i1, i2 = tri['indices']
        p0 = self.particles[i0]['pos']
        p1 = self.particles[i1]['pos']
        p2 = self.particles[i2]['pos']
        
        return compute_element_forces_numba(
            p0, p1, p2, tri['X_inv'], tri['volume'], 
            self.E, self.nu, self.strain_type_num
        )

    def compute_element_stiffness(self, tri):
        """Compute element stiffness matrix (6x6 for 3 nodes with 2 DOF each)."""
        return compute_element_stiffness_numba(
            tri['X_inv'], tri['volume'], self.E, self.nu
        )

    def assemble_global_matrices(self, apply_constraints=True):
        """Assemble global force vector and stiffness matrix."""
        n = len(self.particles)
        f_global = np.zeros((n, 2))
        K_global = lil_matrix((2*n, 2*n))  # Sparse stiffness matrix

        # Get current positions as contiguous array
        positions = np.array([p['pos'] for p in self.particles], dtype=np.float64)

        if self.time < 5.0:
            # --- Rotational external force field ---
            center = np.mean(positions, axis=0)
            # Constant rotation force to show continuous deformation
            if self.time < 3.5:
                rotation_strength = 250.0  # Strong constant force
            else:
                # Gradual slowdown
                rotation_strength = 250.0 * (1.0 - (self.time - 3.5) / 1.5)

            for i in range(n):
                r = positions[i] - center
                tangential = np.array([-r[1], r[0]])
                f_global[i] += rotation_strength * self.particles[i]['mass'] * tangential

        # Compute all element forces in parallel
        all_forces = compute_all_element_forces(
            positions, self.indices_array, self.X_inv_array, 
            self.volumes, self.E, self.nu, self.strain_type_num
        )
        
        # Compute stiffness at CURRENT configuration (key for implicit Euler!)
        all_stiffness = compute_all_element_stiffness_current(
            positions, self.indices_array, self.X_inv_array, 
            self.volumes, self.E, self.nu
        )

        # Assemble into global matrices
        for elem_idx in range(len(self.triangles)):
            indices = self.indices_array[elem_idx]
            f_e = all_forces[elem_idx]
            K_e = all_stiffness[elem_idx]
            
            # Assemble forces
            for local_idx in range(3):
                f_global[indices[local_idx]] += f_e[local_idx]
            
            # Assemble stiffness matrix
            for i in range(3):
                for j in range(3):
                    for di in range(2):
                        for dj in range(2):
                            row = 2 * indices[i] + di
                            col = 2 * indices[j] + dj
                            K_global[row, col] += K_e[2*i + di, 2*j + dj]

        if apply_constraints:
            # Apply constraints
            for i, p in enumerate(self.particles):
                if p['fixed']:
                    for d in range(2):
                        row = 2 * i + d
                        f_global[i, d] = 0.0
                        K_global[row, :] = 0
                        K_global[:, row] = 0
                        K_global[row, row] = 1.0

        return f_global, K_global


    def update(self, dt):
        """Update simulation with implicit (backward) Euler integration.
        
        Implicit Euler scheme:
        - v_{n+1} = v_n + dt * M^{-1} * f(x_{n+1})
        - x_{n+1} = x_n + dt * v_{n+1}
        
        Linearizing f(x_{n+1}) around x_n:
        f(x_{n+1}) ≈ f(x_n) - K * (x_{n+1} - x_n) = f(x_n) - K * dt * v_{n+1}
        
        Substituting:
        v_{n+1} = v_n + dt * M^{-1} * (f_n - K * dt * v_{n+1})
        (M + dt^2 * K) * v_{n+1} = M * v_n + dt * f_n
        """
        self.time += dt
        
        n = len(self.particles)
        damping = 0.998  # Velocity damping
        
        # Assemble force vector and stiffness matrix at current configuration
        f_matrix, K_global = self.assemble_global_matrices(apply_constraints=True)
        self.last_force_matrix = f_matrix
        
        # Convert to sparse format
        K_csr = K_global.tocsr()
        
        # Build mass matrix
        M = lil_matrix((2*n, 2*n))
        for i, p in enumerate(self.particles):
            M[2*i, 2*i] = p['mass']
            M[2*i+1, 2*i+1] = p['mass']
        M_csr = M.tocsr()
        
        # Current velocities
        v_n = np.zeros(2*n)
        for i, p in enumerate(self.particles):
            v_n[2*i] = p['vel'][0]
            v_n[2*i+1] = p['vel'][1]
        
        # Current forces
        f_n = np.zeros(2*n)
        for i in range(n):
            f_n[2*i] = f_matrix[i, 0]
            f_n[2*i+1] = f_matrix[i, 1]
        
        # Implicit Euler system: (M + dt^2 * K) * v_{n+1} = M * v_n + dt * f_n
        A = M_csr + (dt * dt) * K_csr
        b = M_csr @ v_n + dt * f_n
        
        # Solve for v_{n+1}
        try:
            v_new = spsolve(A, b)
            if not np.all(np.isfinite(v_new)):
                raise ValueError("Non-finite velocities")
        except Exception as e:
            # Fallback to explicit Euler
            print(f"Solver failed: {e}, using explicit")
            M_diag = np.array([self.particles[i//2]['mass'] for i in range(2*n)])
            v_new = v_n + dt * (f_n / M_diag)
        
        # Apply damping
        v_new *= damping
        
        # Update particles
        for i, p in enumerate(self.particles):
            if not p['fixed']:
                p['vel'][0] = v_new[2*i]
                p['vel'][1] = v_new[2*i+1]
                # Position update
                p['pos'] += p['vel'] * dt

        # Collisions
        for p in self.particles:
            if p['pos'][1] < self.ground_y:
                p['pos'][1] = self.ground_y
                p['vel'][1] *= -0.3
            if p['pos'][0] < self.left_wall:
                p['pos'][0] = self.left_wall
                p['vel'][0] *= -0.3
            if p['pos'][0] > self.right_wall:
                p['pos'][0] = self.right_wall
                p['vel'][0] *= -0.3

    def get_triangle_positions(self):
        return [np.array([self.particles[i]['pos'] for i in tri['indices']])
                for tri in self.triangles]

    def get_particle_positions(self):
        return np.array([p['pos'] for p in self.particles])


def main():
    sim = FEMSoftBody(
        young_modulus=15000,  # Moderate stiffness for visible deformation
        poisson_ratio=0.3,
        gravity=30,
        fix_top=False,
        strain_type='green',  # Green strain for large deformations/rotations
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
        substeps = 100
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
