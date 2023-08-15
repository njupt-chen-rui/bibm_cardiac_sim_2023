import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time


@ti.data_oriented
class body_2d_square:
    def __init__(self, len_of_square: float, nv_of_row: int) -> None:
        self.density = 1000.0
        self.len_of_square = len_of_square
        self.n = nv_of_row
        self.num_vertex = self.n * self.n
        self.vertex = ti.Vector.field(2, dtype=float, shape=(self.num_vertex,))
        self.num_tet = (self.n - 1) * (self.n - 1) * 2
        self.elements = ti.Vector.field(3, dtype=int, shape=(self.num_tet,))
        self.is_dirichlet_bou = ti.field(dtype=int, shape=(self.num_vertex,))
        self.init_vertex_and_elem()
        self.tet_fiber = ti.Vector.field(2, float, shape=(self.num_tet,))
        self.tet_sheet = ti.Vector.field(2, float, shape=(self.num_tet,))
        self.init_fiber()

        self.Vm = ti.field(float, shape=(self.num_vertex,))

    @ti.kernel
    def init_vertex_and_elem(self):
        len_of_elem = self.len_of_square / (self.n - 1)
        for id_v in range(self.num_vertex):
            j = id_v % self.n
            i = (id_v - j) // self.n
            self.vertex[id_v] = tm.vec2(j * len_of_elem, i * len_of_elem)
            if i == 0 or i == self.n - 1 or j == 0 or j == self.n - 1:
                self.is_dirichlet_bou[id_v] = 1
            else:
                self.is_dirichlet_bou[id_v] = 0

        for id_e in range(self.num_tet):
            j = id_e % (2 * (self.n - 1))
            i = (id_e - j) // (2 * (self.n - 1))
            k = j // 2
            if j % 2 == 0:
                self.elements[id_e] = tm.ivec3(self.n * i + k, self.n * i + k + 1, self.n * (i + 1) + k)
            else:
                self.elements[id_e] = tm.ivec3(self.n * i + k + 1, self.n * (i + 1) + k + 1, self.n * (i + 1) + k)

    @ti.kernel
    def init_fiber(self):
        for i in self.tet_fiber:
            self.tet_fiber[i] = tm.vec2(1.0, 0)
            self.tet_sheet[i] = tm.vec2(0, 1.0)


@ti.data_oriented
class diffusion_reaction_FN:
    """
    use 2nd-order Strang splitting: An intergrative smoothed particle hydrodynamics method for modeling cardiac function
    use FitzHugh-Nagumo model (single cell)
    """

    def __init__(self, body: body_2d_square):
        self.body = body
        self.Vm = self.body.Vm
        self.w = ti.field(float, shape=(body.num_vertex,))
        self.I_ext = ti.field(float, shape=(body.num_vertex,))
        self.init_Vm_w_and_I()

        # parameter of FitzHugh-Nagumo model
        self.a = 0.1
        self.epsilon_0 = 0.01
        self.beta = 0.5
        self.gamma = 1.0
        self.sigma = 0.0

        # parameter of diffusion model
        self.sigma_f = 1.0e-4
        self.sigma_s = 1.0e-4
        self.C_m = 1.0

        self.Dm = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.DmInv = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.Ds = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.F = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.Be = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.init_Ds_F_Be()
        self.fiber = ti.Vector.field(2, float, shape=(body.num_tet,))
        self.sheet = ti.Vector.field(2, float, shape=(body.num_tet,))
        self.init_fiber()
        self.DM = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))
        self.Me = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Ke = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.D = ti.Matrix.field(2, 2, float, shape=(body.num_tet,))

        # use API to solver cg
        self.nv = self.body.n * self.body.n
        self.nvb = (self.body.n - 2) * (self.body.n - 2)
        self.hash_v = ti.field(int, shape=(self.nvb,))
        self.hash_v_r = ti.field(int, shape=(self.nv,))
        self.init_hash()
        self.api_A = ti.linalg.SparseMatrixBuilder(self.nv, self.nv, max_num_triplets=self.nv * self.nv * 7)
        self.api_M = ti.linalg.SparseMatrixBuilder(self.nv, self.nv, max_num_triplets=self.nv * self.nv * 7)
        self.api_Vm = ti.ndarray(ti.f32, self.nv)
        self.api_b = ti.ndarray(ti.f32, self.nv)

        # vertex color
        self.vertex_color = ti.Vector.field(3, float, self.body.num_vertex)

    @ti.kernel
    def init_Vm_w_and_I(self):
        for i in self.Vm:
            self.Vm[i] = 0.0

        for i in self.w:
            self.w[i] = 0.0

        for i in self.I_ext:
            self.I_ext[i] = 0.0

    @ti.kernel
    def init_Ds_F_Be(self):
        for i in range(self.body.num_tet):
            self.Dm[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][2]][0]
            self.Dm[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][2]][1]

            self.Dm[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][2]][0]
            self.Dm[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][2]][1]

        for i in range(self.body.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()

        for i in range(self.body.num_tet):
            self.Ds[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][2]][0]
            self.Ds[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][2]][1]

            self.Ds[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][2]][0]
            self.Ds[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][2]][1]

        for i in range(self.body.num_tet):
            self.Be[i][0, 0] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 0] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]

            self.Be[i][0, 1] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 1] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]

        for i in range(self.body.num_tet):
            self.F[i] = self.Ds[i] @ self.DmInv[i]

    @ti.kernel
    def init_fiber(self):
        for i in range(self.body.num_tet):
            self.fiber[i] = self.F[i] @ self.body.tet_fiber[i]
            self.sheet[i] = self.F[i] @ self.body.tet_sheet[i]

    def init_hash(self):
        idv = 0
        for i in range(self.body.num_vertex):
            if self.body.is_dirichlet_bou[i] == 0:
                self.hash_v[idv] = i
                self.hash_v_r[i] = idv
                idv += 1
            else:
                self.hash_v_r[i] = -1

    @ti.kernel
    def calculate_reaction(self, dt: float):
        """
        2 ODEs:
        Rv: C_m dV_m/dt = I_ion(V_m, w)
        Rw: dw/dt = g(V_m, w)
        use Reaction-by-Reaction splitting method to decouple ODEs,
        update: R(t+dt)=Rv(dt/2){Rw(dt/2){Rw(dt/2){Rv(dt/2){R(t)}}}}
        rewrite a single reaction equation in a general form as
        dy/dt = q(y,t) - p(y,t)y
        use QSS method, a linearly approximated exact solution, which is unconditionally stable:
        y(t+dt) = y(t)exp(-p(y(t),t)dt) + q(y(t),t)/p(y(t),t) * (1-exp(-p(y(t),t)dt))
        """
        for i in self.Vm:
            if self.body.is_dirichlet_bou[i] == 0:
                self.calculate_Rv(i, dt * 0.5)
                self.calculate_Rw(i, dt * 0.5)
                self.calculate_Rw(i, dt * 0.5)
                self.calculate_Rv(i, dt * 0.5)

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = 1/C_m * [V_m(V_m + aV_m - V_m^2) - w] - a/C_m * V_m
        y = V_m, q(y,t) = 1/C_m * [V_m(V_m + aV_m - V_m^2) - w], p(y,t) = a/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * (self.a / self.C_m)) + (
                1.0 / self.C_m * (self.Vm[i] * self.Vm[i] * (1.0 + self.a - self.Vm[i]) - self.w[i])) / (
                                 self.a / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * (self.a / self.C_m)))

    @ti.func
    def calculate_Rw_old(self, i, dt):
        """
        dw/dt = epsilon_0 * beta * V_m * (V_m - sigma) - epsilon_0 * gamma * w
        y = w, q(y,t) = epsilon_0 * beta * V_m * (V_m - sigma), p(y,t) = epsilon_0 * gamma
        """
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * self.epsilon_0 * self.gamma) + (
                self.epsilon_0 * self.beta * self.Vm[i] * (self.Vm[i] - self.sigma) / (self.epsilon_0 * self.gamma)) * (
                               1.0 - tm.exp(-1.0 * dt * (self.epsilon_0 * self.gamma)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon_0 * (beta * V_m - sigma) - epsilon_0 * gamma * w
        y = w, q(y,t) = epsilon_0 * (beta * V_m - sigma), p(y,t) = epsilon_0 * gamma
        """
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * self.epsilon_0 * self.gamma) + (
                self.epsilon_0 * (self.beta * self.Vm[i] - self.sigma) / (self.epsilon_0 * self.gamma)) * (
                            1.0 - tm.exp(-1.0 * dt * (self.epsilon_0 * self.gamma)))

    @ti.kernel
    def calculate_M_and_K(self):
        fiber, sheet = ti.static(self.fiber, self.sheet)
        for i in range(self.body.num_tet):
            self.Me[i] = 0.25 * ti.abs(self.Be[i].determinant()) * \
                         tm.mat3([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])

            D_f = ti.Matrix([[fiber[i][0] * fiber[i][0],
                              fiber[i][1] * fiber[i][0]],
                             [fiber[i][0] * fiber[i][1],
                              fiber[i][1] * fiber[i][1]]], float)
            D_s = ti.Matrix([[sheet[i][0] * sheet[i][0],
                              sheet[i][1] * sheet[i][0]],
                             [sheet[i][0] * sheet[i][1],
                              sheet[i][1] * sheet[i][1]]], float)

            norm_f = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1]
            norm_s = sheet[i][0] * sheet[i][0] + sheet[i][1] * sheet[i][1]
            self.DM[i] = self.sigma_f / norm_f * D_f + self.sigma_s / norm_s * D_s
            J = self.F[i].determinant()
            self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

            J_phi = ti.Matrix([[-1., -1.],
                               [1., 0.],
                               [0., 1.]], float)

            B = J_phi @ self.Be[i].inverse()
            B_ = B.transpose()
            C = self.D[i] @ B_
            self.Ke[i] = ti.abs(self.Be[i].determinant()) * C.transpose() @ B_

            # self.Ke[i] = J * ti.abs(self.Be[i].determinant()) * self.DM[i] @ A @ A.transpose()
            # self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

    def calculate_diffusion(self, dt):
        self.calculate_M_and_K()

        # TODO: change CG from n * n to (n-m)*(n-m)
        self.assemble_A(self.api_A, dt)
        A = self.api_A.build()
        self.assemble_M(self.api_M)
        M = self.api_M.build()
        self.copy_Vm(self.api_Vm, self.Vm)
        self.api_b = M @ self.api_Vm
        self.mod_b(self.api_b)

        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.api_b)
        self.updateVm_api(dv)

    @ti.kernel
    def assemble_A(self, A: ti.types.sparse_matrix_builder(), dt: float):
        elem = ti.static(self.body.elements)
        for i in elem:
            idx = tm.ivec3(0, 0, 0)
            for k in ti.static(range(3)):
                idx[k] = elem[i][k]
            for n in ti.static(range(3)):
                for m in ti.static(range(3)):
                    if self.body.is_dirichlet_bou[idx[n]] == 0 and self.body.is_dirichlet_bou[idx[m]] == 0:
                        A[idx[n], idx[m]] += (self.Me[i][n, m] + self.Ke[i][n, m] * dt)

        for i in range(self.nv):
            if self.body.is_dirichlet_bou[i] == 1:
                A[i, i] += 1

    @ti.kernel
    def assemble_M(self, M: ti.types.sparse_matrix_builder()):
        elem = ti.static(self.body.elements)
        for i in elem:
            idx = tm.ivec3(0, 0, 0)
            for k in ti.static(range(3)):
                idx[k] = elem[i][k]
            for n in ti.static(range(3)):
                for m in ti.static(range(3)):
                    M[idx[n], idx[m]] += self.Me[i][n, m]

    @ti.kernel
    def copy_Vm(self, des: ti.types.ndarray(), source: ti.template()):
        for i in range(self.body.num_vertex):
            des[i] = source[i]

    @ti.kernel
    def mod_b(self, des: ti.types.ndarray()):
        for i in range(self.nv):
            if self.body.is_dirichlet_bou[i] == 1:
                des[i] = 0.0

    @ti.kernel
    def updateVm_api(self, dv: ti.types.ndarray()):
        for i in dv:
            self.Vm[i] = dv[i]


    def update_Vm(self, dt):
        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)

    @ti.kernel
    def init_Vm_w_experiment2(self):
        for i in self.Vm:
            x = self.body.vertex[i][0]
            y = self.body.vertex[i][1]
            # TODO: 2.5的边界处可能存在数值误差，有待检查
            if x > 0 and x <= 1.25 and y > 0 and y < 1.25 and self.body.is_dirichlet_bou[i] == 0:
                self.Vm[i] = 1.0
            else:
                self.Vm[i] = 0.0
            if x > 0 and x <= 1.25 and y >= 1.25 and y < 2.5 and self.body.is_dirichlet_bou[i] == 0:
                self.w[i] = 0.1
            elif x >= 1.25 and x < 2.5 and y >= 1.25 and y < 2.5 and self.body.is_dirichlet_bou[i] == 0:
                self.w[i] = 0.1
            else:
                self.w[i] = 0.0


def example2_1():
    body1 = body_2d_square(2.5, 100)
    ep1 = diffusion_reaction_FN(body=body1)
    ep1.sigma_f = 1.0e-4
    ep1.sigma_s = 1.0e-4
    ep1.init_Vm_w_experiment2()

    tol_time = 1000
    cnt = 10

    n = 100
    x = np.linspace(0, 2.5, n)
    y = np.linspace(0, 2.5, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    cur_time = 0.0
    sub_plot_row = 1
    sub_plot_col = 5
    if abs(cur_time - 0.0) < 1e-6:
        for idi in range(n):
            for idj in range(n):
                idv = idi * n + idj
                Z[idi, idj] = ep1.Vm[idv]

        plt.subplot(sub_plot_row, sub_plot_col, 1)
        plt.imshow(Z, origin='lower', extent=[0.0, 2.5, 0.0, 2.5],
                   cmap='jet', alpha=1.0)
    # plt.savefig("../doc/res/3.3.1.png")

    dt = 1.0 / cnt
    for tt in range(tol_time):
        for st in range(cnt):
            ep1.update_Vm(dt)
            cur_time += dt

            if abs(cur_time - 250.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]

                plt.subplot(sub_plot_row, sub_plot_col, 2)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.2.png")

            if abs(cur_time - 500.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 3)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.3.png")

            if abs(cur_time - 750.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 4)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.4.png")

            if abs(cur_time - 1000.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 5)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.5.png")

    plt.show()


def example2_2():
    body1 = body_2d_square(2.5, 100)
    ep1 = diffusion_reaction_FN(body=body1)
    ep1.sigma_f = 1.0e-4
    ep1.sigma_s = 2.5e-5
    ep1.init_Vm_w_experiment2()

    tol_time = 1000
    cnt = 10

    n = 100
    x = np.linspace(0, 2.5, n)
    y = np.linspace(0, 2.5, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    cur_time = 0.0
    sub_plot_row = 1
    sub_plot_col = 5
    if abs(cur_time - 0.0) < 1e-6:
        for idi in range(n):
            for idj in range(n):
                idv = idi * n + idj
                Z[idi, idj] = ep1.Vm[idv]

        plt.subplot(sub_plot_row, sub_plot_col, 1)
        plt.imshow(Z, origin='lower', extent=[0.0, 2.5, 0.0, 2.5],
                   cmap='jet', alpha=1.0)
    # plt.savefig("../doc/res/3.3.1.png")

    dt = 1.0 / cnt
    for tt in range(tol_time):
        for st in range(cnt):
            ep1.update_Vm(dt)
            cur_time += dt

            if abs(cur_time - 250.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]

                plt.subplot(sub_plot_row, sub_plot_col, 2)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.2.png")

            if abs(cur_time - 500.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 3)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.3.png")

            if abs(cur_time - 750.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 4)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.4.png")

            if abs(cur_time - 1000.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 5)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.5.png")

    plt.show()


def example2_3():
    body1 = body_2d_square(2.5, 100)
    ep1 = diffusion_reaction_FN(body=body1)
    ep1.sigma_f = 1.0e-4
    ep1.sigma_s = 1.0e-5
    ep1.init_Vm_w_experiment2()

    tol_time = 1000
    cnt = 10

    n = 100
    x = np.linspace(0, 2.5, n)
    y = np.linspace(0, 2.5, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    cur_time = 0.0
    sub_plot_row = 1
    sub_plot_col = 5
    if abs(cur_time - 0.0) < 1e-6:
        for idi in range(n):
            for idj in range(n):
                idv = idi * n + idj
                Z[idi, idj] = ep1.Vm[idv]

        plt.subplot(sub_plot_row, sub_plot_col, 1)
        plt.imshow(Z, origin='lower', extent=[0.0, 2.5, 0.0, 2.5],
                   cmap='jet', alpha=1.0)
    # plt.savefig("../doc/res/3.3.1.png")

    dt = 1.0 / cnt
    for tt in range(tol_time):
        for st in range(cnt):
            ep1.update_Vm(dt)
            cur_time += dt

            if abs(cur_time - 250.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]

                plt.subplot(sub_plot_row, sub_plot_col, 2)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.2.png")

            if abs(cur_time - 500.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 3)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.3.png")

            if abs(cur_time - 750.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 4)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.4.png")

            if abs(cur_time - 1000.0) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]
                plt.subplot(sub_plot_row, sub_plot_col, 5)
                plt.imshow(Z, extent=[0, 2.5, 0, 2.5], origin='lower',
                           cmap='jet', alpha=1.0)
                # plt.savefig("../doc/res/3.3.5.png")

    plt.show()


if __name__ == "__main__":
    # ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True, device_memory_fraction=0.9, device_memory_GB=4)
    ti.init(arch=ti.cpu)

    # isotropic
    example2_1()

    # anisotropic
    example2_2()

    # lager anisotropic
    example2_3()
