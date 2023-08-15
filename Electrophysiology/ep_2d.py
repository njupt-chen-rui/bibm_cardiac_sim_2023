import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
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
        self.init_vertex_and_elem()
        self.tet_fiber = ti.Vector.field(2, float, shape=(self.num_tet,))
        self.tet_sheet = ti.Vector.field(2, float, shape=(self.num_tet,))
        self.init_fiber()

        self.Vm = ti.field(float, shape=(self.num_vertex,))

        # self.Dm = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        # self.DmInv = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        # self.DmInvT = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        # self.init_DmInv()
        #
        # self.Vm = ti.field(float, shape=(self.num_vertex,))
        #
        # # volume
        # self.volume = ti.field(float, self.num_tet)
        # self.init_volume()
        #
        # self.tet_Ta = ti.field(float, shape=(self.num_tet,))
        # self.ver_Ta = ti.field(float, shape=(self.num_vertex,))
        # self.init_electrophysiology()

    @ti.kernel
    def init_vertex_and_elem(self):
        len_of_elem = self.len_of_square / (self.n - 1)
        for id_v in range(self.num_vertex):
            j = id_v % self.n
            i = (id_v - j) // self.n
            self.vertex[id_v] = tm.vec2(j * len_of_elem, i * len_of_elem)

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
class diffusion_reaction:
    """
    use 2nd-order Strang splitting: An intergrative smoothed particle hydrodynamics method for modeling cardiac function
    use Aliec-Panfilov model (single cell)
    """

    def __init__(self, body: body_2d_square):
        self.body = body
        self.Vm = self.body.Vm
        self.w = ti.field(float, shape=(body.num_vertex,))
        self.I_ext = ti.field(float, shape=(body.num_vertex,))
        self.init_Vm_w_and_I()

        # parameter of Aliec-Panfilov model
        self.k = 8.0
        self.a = 0.15
        self.b = 0.15
        self.epsilon_0 = 0.002
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.C_m = 1.0
        self.Eps = 2.22045e-16
        self.TinyReal = 2.71051e-20

        # parameter of diffusion model
        # TODO: change sigma
        # self.sigma_f = 7.643e-5 * 5e4
        # self.sigma_s = 3.494e-5 * 5e4
        self.sigma_f = 1.0
        self.sigma_s = 1.0

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

        # For conjugate gradient method
        self.cg_x = ti.field(float, self.body.num_vertex)
        self.cg_Ax = ti.field(float, self.body.num_vertex)
        self.cg_b = ti.field(float, self.body.num_vertex)
        self.cg_r = ti.field(float, self.body.num_vertex)
        self.cg_d = ti.field(float, self.body.num_vertex)
        self.cg_Ad = ti.field(float, self.body.num_vertex)
        self.pcg_M = ti.field(float, self.body.num_vertex)
        self.pcg_s = ti.field(float, self.body.num_vertex)
        self.cg_epsilon = 1.0e-3
        # debug
        self.cg_A = ti.field(float, shape=(self.body.num_vertex, self.body.num_vertex))

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
        # for i in range(self.body.num_tet):
        #     self.body.tet_fiber[i] = tm.vec3([0., -1.0, 0.])
        #     self.body.tet_sheet[i] = tm.vec3([1., 0., 0.])
        for i in range(self.body.num_tet):
            self.fiber[i] = self.F[i] @ self.body.tet_fiber[i]
            self.sheet[i] = self.F[i] @ self.body.tet_sheet[i]

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
            self.calculate_Rv(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rv(i, dt * 0.5)

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = kV_m/C_m (V_m + aV_m -V_m^2) + I_ext - (ka+w)/C_m * V_m
        y = V_m, q(y,t) = kV_m/C_m * (V_m + aV_m -V_m^2) + I_ext, p(y,t) = (ka+w)/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)) + (
                self.k * self.Vm[i] / self.C_m * (self.Vm[i] * (1.0 + self.a - self.Vm[i]))) / (
                                 (self.k * self.a + self.w[i]) / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon(V_m, w) * k * V_m * (1 + b - V_m) - epsilon(V_m, w) * w
        epsilon(V_m, w) = epsilon_0 + mu_1 * w / (mu_2 + V_m)
        y = w, q(y,t) = epsilon(V_m, w) * k * V_m * (1 + b - V_m), p(y,t) = epsilon(V_m, w)
        """
        epsilon_Vm_w = self.epsilon_0 + self.mu_1 * self.w[i] / (self.mu_2 + self.Vm[i])
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * epsilon_Vm_w) + (
                epsilon_Vm_w * self.k * self.Vm[i] * (1.0 + self.b - self.Vm[i]) / epsilon_Vm_w) * (
                               1.0 - tm.exp(-1.0 * dt * epsilon_Vm_w))

    @ti.kernel
    def calculate_M_and_K(self):
        fiber, sheet = ti.static(self.fiber, self.sheet)
        for i in range(self.body.num_tet):
            self.Me[i] = 0.25 / 2.0 * ti.abs(self.Be[i].determinant()) * \
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

    @ti.kernel
    def compute_RHS(self):
        for i in self.cg_b:
            self.cg_b[i] = 0.0
        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_tet):
            id0, id1, id2 = (self.body.elements[i][0], self.body.elements[i][1], self.body.elements[i][2])
            self.cg_b[id0] += (self.Me[i][0, 0] * self.Vm[id0] + self.Me[i][0, 1] * self.Vm[id1] +
                               self.Me[i][0, 2] * self.Vm[id2])
            self.cg_b[id1] += (self.Me[i][1, 0] * self.Vm[id0] + self.Me[i][1, 1] * self.Vm[id1] +
                               self.Me[i][1, 2] * self.Vm[id2])
            self.cg_b[id2] += (self.Me[i][2, 0] * self.Vm[id0] + self.Me[i][2, 1] * self.Vm[id1] +
                               self.Me[i][2, 2] * self.Vm[id2])


    @ti.func
    def A_mult_x(self, dt, dst, src):
        # lhs = Ax = (M + K * dt) * u(t+1)
        for i in range(self.body.num_vertex):
            dst[i] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2 = (self.body.elements[i][0], self.body.elements[i][1], self.body.elements[i][2])
            dst[id0] += (self.Me[i][0, 0] * src[id0] + self.Me[i][0, 1] * src[id1] +
                         self.Me[i][0, 2] * src[id2])
            dst[id1] += (self.Me[i][1, 0] * src[id0] + self.Me[i][1, 1] * src[id1] +
                         self.Me[i][1, 2] * src[id2])
            dst[id2] += (self.Me[i][2, 0] * src[id0] + self.Me[i][2, 1] * src[id1] +
                         self.Me[i][2, 2] * src[id2])

            dst[id0] += (self.Ke[i][0, 0] * src[id0] + self.Ke[i][0, 1] * src[id1] +
                         self.Ke[i][0, 2] * src[id2]) * dt
            dst[id1] += (self.Ke[i][1, 0] * src[id0] + self.Ke[i][1, 1] * src[id1] +
                         self.Ke[i][1, 2] * src[id2]) * dt
            dst[id2] += (self.Ke[i][2, 0] * src[id0] + self.Ke[i][2, 1] * src[id1] +
                         self.Ke[i][2, 2] * src[id2]) * dt


    @ti.func
    def dot(self, v1, v2):
        result = 0.0
        for i in range(self.body.num_vertex):
            result += v1[i] * v2[i]
        return result

    # cg #
    # --------------------------------------------------------------------------------------------------- #
    @ti.kernel
    def cg_before_ite(self, dt: float) -> float:
        for i in range(self.body.num_vertex):
            self.cg_x[i] = self.Vm[i]
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

        for i in range(self.body.num_vertex):
            # r = b - A @ x
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
            # d = r
            self.cg_d[i] = self.cg_r[i]

        delta_new = self.dot(self.cg_r, self.cg_r)
        return delta_new

    @ti.kernel
    def cg_run_iteration(self, dt: float, delta: float) -> float:
        delta_new = delta
        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)
        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_vertex):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]
            # r = b - A @ x || r = r - alpha * q
            self.cg_r[i] -= alpha * self.cg_Ad[i]
        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.cg_r)
        beta = delta_new / delta_old
        for i in range(self.body.num_vertex):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
        return delta_new

    def cg(self, dt: float):
        delta_new = self.cg_before_ite(dt)
        delta_0 = delta_new
        # ite, iteMax = 0, 100
        ite, iteMax = 0, 200
        while ite < iteMax and delta_new > (self.cg_epsilon**2) * delta_0:
            delta_new = self.cg_run_iteration(dt, delta_new)
            ite += 1

    # --------------------------------------------------------------------------------------------------- #

#     # pcg
#     # --------------------------------------------------------------------------------------------------- #
#     @ti.func
#     def M_inv_mult_r_Jacobi(self, dt, dst, src):
#         # A = M + K * dt
#         # M = diag(A)
#         # M^-1r = M^-1 @ r
#         for i in self.pcg_M:
#             self.pcg_M[i] = 0.0
#         for i in range(self.body.num_tet):
#             id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
#                                   self.body.elements[i][2], self.body.elements[i][3])
#             self.pcg_M[id0] += self.Me[i][0, 0] + self.Ke[i][0, 0] * dt
#             self.pcg_M[id1] += self.Me[i][1, 1] + self.Ke[i][1, 1] * dt
#             self.pcg_M[id2] += self.Me[i][2, 2] + self.Ke[i][2, 2] * dt
#             self.pcg_M[id3] += self.Me[i][3, 3] + self.Ke[i][3, 3] * dt
#         for i in range(self.body.num_vertex):
#             dst[i] = src[i] / self.pcg_M[i]
#
#     @ti.kernel
#     def pcg_before_ite(self, dt: float) -> float:
#         # x = u(t)
#         for i in range(self.body.num_vertex):
#             # self.cg_x[i] = self.Vm[i]
#             self.cg_x[i] = 0.0
#         # Ax = A @ x
#         self.A_mult_x(dt, self.cg_Ax, self.cg_x)
#
#         for i in range(self.body.num_vertex):
#             # r = b - A @ x
#             self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
#
#         # d = M^-1 @ r
#         self.M_inv_mult_r_Jacobi(dt, self.cg_d, self.cg_r)
#
#         delta_new = self.dot(self.cg_r, self.cg_d)
#         return delta_new
#
#     @ti.kernel
#     def pcg_run_iteration1(self, dt: float, delta: float) -> float:
#         delta_new = delta
#         # q = A @ d
#         self.A_mult_x(dt, self.cg_Ad, self.cg_d)
#         # alpha = delta_new / d.dot(q)
#         alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)
#
#         for i in range(self.body.num_vertex):
#             # x = x + alpha * d
#             self.cg_x[i] += alpha * self.cg_d[i]
#
#         # if ite % 50 == 0: r = b - A @ x
#         self.A_mult_x(dt, self.cg_Ax, self.cg_x)
#         for i in range(self.body.num_vertex):
#             self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
#
#         # s = M^-1 @ r
#         self.M_inv_mult_r_Jacobi(dt, self.pcg_s, self.cg_r)
#         delta_old = delta_new
#         delta_new = self.dot(self.cg_r, self.pcg_s)
#         beta = delta_new / delta_old
#         for i in range(self.body.num_vertex):
#             # d = r + beta * d
#             self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
#         return delta_new
#
#     @ti.kernel
#     def pcg_run_iteration2(self, dt: float, delta: float) -> float:
#         delta_new = delta
#         # q = A @ d
#         self.A_mult_x(dt, self.cg_Ad, self.cg_d)
#         # alpha = delta_new / d.dot(q)
#         alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)
#
#         for i in range(self.body.num_vertex):
#             # x = x + alpha * d
#             self.cg_x[i] += alpha * self.cg_d[i]
#             # if ite % 50 != 0: r = r - alpha * q
#             self.cg_r[i] -= alpha * self.cg_Ad[i]
#
#         # s = M^-1 @ r
#         self.M_inv_mult_r_Jacobi(dt, self.pcg_s, self.cg_r)
#         delta_old = delta_new
#         delta_new = self.dot(self.cg_r, self.pcg_s)
#         beta = delta_new / delta_old
#         for i in range(self.body.num_vertex):
#             # d = r + beta * d
#             self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
#         return delta_new
#
#     def pcg(self, dt: float):
#         delta_new = self.pcg_before_ite(dt)
#         delta_0 = delta_new
#         ite, iteMax = 0, 1000
#         while ite < iteMax and delta_new > (self.cg_epsilon ** 2) * delta_0:
#             if ite % 50 == 0:
#                 delta_new = self.pcg_run_iteration1(dt, delta_new)
#             else:
#                 delta_new = self.pcg_run_iteration2(dt, delta_new)
#             ite += 1
#         print("ite:{}, delta_0:{}, delta_new:{}".format(ite, delta_0, delta_new))
#     # --------------------------------------------------------------------------------------------------- #

    @ti.kernel
    def cgUpdateVm(self):
        for i in self.Vm:
            self.Vm[i] = self.cg_x[i]

    def calculate_diffusion(self, dt):
        self.calculate_M_and_K()
        self.compute_RHS()
        self.cg(dt)
        # self.pcg(dt)
        A = self.Me[0] + self.Ke[0] * dt
        x = tm.vec3(self.cg_x[0], self.cg_x[1], self.cg_x[2])
        b = self.Me[0] @ tm.vec3(self.Vm[0], self.Vm[1], self.Vm[2])
        # print(A @ x)
        # print(b)
        self.cgUpdateVm()

    def update_Vm(self, dt):
        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)
        # self.calculate_reaction(dt)
        # self.calculate_diffusion(dt)
        # self.calculate_diffusion(dt)
        # self.calculate_reaction(dt)

    @ti.kernel
    def get_near_vertex_index(self, x: float, y: float) -> int:
        vert = ti.static(self.body.vertex)
        res = 0
        for i in vert:
            if (vert[i][0] - x)**2 + (vert[i][1] - y)**2 < 1e-4:
                res = i
                print(i)
        return res

    def update(self, sub_steps):
        dt = 1. / 1.29 / 6. / sub_steps
        # dt = 1. / 60. / sub_steps
        for _ in range(sub_steps):
            self.update_Vm(dt)
        self.update_color()

    @ti.kernel
    def update_color(self):
        for i in self.vertex_color:
            self.vertex_color[i] = tm.vec3([self.Vm[i], self.Vm[i], 1])

    @ti.kernel
    def init_Vm_w_example1(self):
        for i in self.Vm:
            x = self.body.vertex[i][0]
            y = self.body.vertex[i][1]
            self.Vm[i] = tm.exp(-4.0 * ((x - 1.0) * (x - 1.0) + y * y))
            self.w[i] = 0.0

    @ti.kernel
    def init_Vm_w_experiment2(self):
        for i in self.Vm:
            x = self.body.vertex[i][0]
            y = self.body.vertex[i][1]
            if x >= 0 and x <= 1.25 and y >= 0 and y <= 1.25:
                self.Vm[i] = 1.0
            else:
                self.Vm[i] = 0.0
            if x >= 0 and x <= 1.25 and y >= 1.25 and y <= 2.5:
                self.w[i] = 0.1
            elif x >= 1.25 and x <= 2.5 and y >= 0 and y <= 2.5:
                self.w[i] = 0.1
            else:
                self.w[i] = 0.0


def example1():
    body1 = body_2d_square(2.5, 100)
    ep1 = diffusion_reaction(body=body1)
    ep1.sigma_f = 1.0
    ep1.sigma_s = 1.0
    ep1.init_Vm_w_experiment2()

    tol_time = 1000
    cnt = 10

    n = 100
    x = np.linspace(0, 2.5, n)
    y = np.linspace(0, 2.5, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    cur_time = 0.0
    target_time = 0.0
    if abs(cur_time - target_time) < 1e-6:
        for idi in range(n):
            for idj in range(n):
                idv = idi * n + idj
                Z[idi, idj] = ep1.Vm[idv]

        plt.subplot(2,3,1)
        # contours = plt.contour(X, Y, Z, 4, colors='black', linewidths=.5)
        # plt.clabel(contours, inline=True, fontsize=8)
        
        plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower',
                cmap='jet', alpha=1.0, vmin=0, vmax=1)
        # plt.colorbar()
        # plt.plot(0.3, 0.7, 'ko', label='P', markersize=3.0)
        # plt.text(0.25, 0.72, "P", fontsize=12, color="black", weight="light", verticalalignment="center")
        # plt.xlabel(r"$\mathrm{(a)}t_1=0$")

    dt = 1.0 / cnt
    flag = False
    for tt in range(tol_time):
        for st in range(cnt):
            ep1.update_Vm(dt)
            cur_time += dt
            if abs(cur_time - 50) < 1e-6:
                for idi in range(n):
                    for idj in range(n):
                        idv = idi * n + idj
                        Z[idi, idj] = ep1.Vm[idv]

                plt.subplot(2,3,2)
                
                plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower',
                        cmap='jet', alpha=1.0, vmin=0, vmax=1)
                plt.colorbar()
                plt.show()
                return
            
    # plt.show()


if __name__ == "__main__":
    # ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True, device_memory_fraction=0.9, device_memory_GB=4)
    ti.init(arch=ti.cpu)

    example1()
