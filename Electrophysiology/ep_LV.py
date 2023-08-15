import taichi as ti
import taichi.math as tm
import numpy as np
from Geometry.body_LV import Body
# from test1 import meshData
import matplotlib.pyplot as plt
import time
from data.LV1 import meshData
# from data.cube import meshData
# from data.heart import meshData


@ti.data_oriented
class diffusion_reaction:
    """
    use 2nd-order Strang splitting: An intergrative smoothed particle hydrodynamics method for modeling cardiac function
    use Aliec-Panfilov model (single cell)
    """

    def __init__(self, body: Body):
        self.body = body
        # self.Vm = ti.field(float, shape=(body.num_vertex,))
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
        self.sigma_f = 7.643e-5 * 5e4
        self.sigma_s = 3.494e-5 * 5e4
        self.sigma_n = 1.125e-5 * 5e4
        self.Dm = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Ds = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.F = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Be = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.init_Ds_F_Be()
        self.fiber = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.sheet = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.normal = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.init_fiber()
        self.DM = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Me = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.Ke = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.D = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))

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
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()

        for i in range(self.body.num_tet):
            self.Ds[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.Be[i][0, 0] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 0] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 0] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 1] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 1] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 1] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 2] = self.body.vertex[self.body.elements[i][3]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 2] = self.body.vertex[self.body.elements[i][3]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 2] = self.body.vertex[self.body.elements[i][3]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]

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
            self.normal[i] = self.F[i] @ tm.cross(self.body.tet_fiber[i], self.body.tet_sheet[i])

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
                self.k * self.Vm[i] / self.C_m * (self.Vm[i] * (1.0 + self.a - self.Vm[i])) + self.I_ext[i]) / (
                                 (self.k * self.a + self.w[i]) / self.C_m + self.TinyReal) * (
                                 1.0 - tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon(V_m, w) * k * V_m * (1 + b - V_m) - epsilon(V_m, w) * w
        epsilon(V_m, w) = epsilon_0 + mu_1 * w / (mu_2 + V_m)
        y = w, q(y,t) = epsilon(V_m, w) * k * V_m * (1 + b - V_m), p(y,t) = epsilon(V_m, w)
        """
        epsilon_Vm_w = self.epsilon_0 + self.mu_1 * self.w[i] / (self.mu_2 + self.Vm[i] + self.Eps)
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * epsilon_Vm_w) + (
                self.k * self.Vm[i] * (1.0 + self.b - self.Vm[i])) * (
                               1.0 - tm.exp(-1.0 * dt * epsilon_Vm_w))

    @ti.kernel
    def calculate_M_and_K(self):
        fiber, sheet, normal = ti.static(self.fiber, self.sheet, self.normal)
        for i in range(self.body.num_tet):
            self.Me[i] = 0.25 / 6.0 * ti.abs(self.Be[i].determinant()) * \
                         tm.mat4([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]]) * 6.0
            J_phi = ti.Matrix([[-1., -1., -1.],
                               [1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]], float)
            D_f = ti.Matrix([[fiber[i][0] * fiber[i][0],
                              fiber[i][1] * fiber[i][0],
                              fiber[i][2] * fiber[i][0]],
                             [fiber[i][0] * fiber[i][1],
                              fiber[i][1] * fiber[i][1],
                              fiber[i][2] * fiber[i][1]],
                             [fiber[i][0] * fiber[i][2],
                              fiber[i][1] * fiber[i][2],
                              fiber[i][2] * fiber[i][2]]], float)
            D_s = ti.Matrix([[sheet[i][0] * sheet[i][0],
                              sheet[i][1] * sheet[i][0],
                              sheet[i][2] * sheet[i][0]],
                             [sheet[i][0] * sheet[i][1],
                              sheet[i][1] * sheet[i][1],
                              sheet[i][2] * sheet[i][1]],
                             [sheet[i][0] * sheet[i][2],
                              sheet[i][1] * sheet[i][2],
                              sheet[i][2] * sheet[i][2]]], float)
            D_n = ti.Matrix([[normal[i][0] * normal[i][0],
                              normal[i][1] * normal[i][0],
                              normal[i][2] * normal[i][0]],
                             [normal[i][0] * normal[i][1],
                              normal[i][1] * normal[i][1],
                              normal[i][2] * normal[i][1]],
                             [normal[i][0] * normal[i][2],
                              normal[i][1] * normal[i][2],
                              normal[i][2] * normal[i][2]]], float)
            norm_f = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_s = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_n = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            self.DM[i] = self.sigma_f / norm_f * D_f + self.sigma_s / norm_s * D_s + self.sigma_n / norm_n * D_n
            J = self.F[i].determinant()
            A = J_phi @ self.Be[i].inverse() @ self.F[i].inverse()
            self.Ke[i] = 1.0 / 6.0 * J * ti.abs(self.Be[i].determinant()) * A @ self.DM[i] @ A.transpose() * 6.0
            self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

    @ti.kernel
    def compute_RHS(self):
        for i in self.cg_b:
            self.cg_b[i] = 0.0
        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_b[id0] += (self.Me[i][0, 0] * self.Vm[id0] + self.Me[i][0, 1] * self.Vm[id1] +
                               self.Me[i][0, 2] * self.Vm[id2] + self.Me[i][0, 3] * self.Vm[id3])
            self.cg_b[id1] += (self.Me[i][1, 0] * self.Vm[id0] + self.Me[i][1, 1] * self.Vm[id1] +
                               self.Me[i][1, 2] * self.Vm[id2] + self.Me[i][1, 3] * self.Vm[id3])
            self.cg_b[id2] += (self.Me[i][2, 0] * self.Vm[id0] + self.Me[i][2, 1] * self.Vm[id1] +
                               self.Me[i][2, 2] * self.Vm[id2] + self.Me[i][2, 3] * self.Vm[id3])
            self.cg_b[id3] += (self.Me[i][3, 0] * self.Vm[id0] + self.Me[i][3, 1] * self.Vm[id1] +
                               self.Me[i][3, 2] * self.Vm[id2] + self.Me[i][3, 3] * self.Vm[id3])

    @ti.func
    def A_mult_x(self, dt, dst, src):
        # lhs = Ax = (M + K * dt) * u(t+1)
        for i in range(self.body.num_vertex):
            dst[i] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            dst[id0] += (self.Me[i][0, 0] * src[id0] + self.Me[i][0, 1] * src[id1] +
                         self.Me[i][0, 2] * src[id2] + self.Me[i][0, 3] * src[id3])
            dst[id1] += (self.Me[i][1, 0] * src[id0] + self.Me[i][1, 1] * src[id1] +
                         self.Me[i][1, 2] * src[id2] + self.Me[i][1, 3] * src[id3])
            dst[id2] += (self.Me[i][2, 0] * src[id0] + self.Me[i][2, 1] * src[id1] +
                         self.Me[i][2, 2] * src[id2] + self.Me[i][2, 3] * src[id3])
            dst[id3] += (self.Me[i][3, 0] * src[id0] + self.Me[i][3, 1] * src[id1] +
                         self.Me[i][3, 2] * src[id2] + self.Me[i][3, 3] * src[id3])
            dst[id0] += (self.Ke[i][0, 0] * src[id0] + self.Ke[i][0, 1] * src[id1] +
                         self.Ke[i][0, 2] * src[id2] + self.Ke[i][0, 3] * src[id3]) * dt
            dst[id1] += (self.Ke[i][1, 0] * src[id0] + self.Ke[i][1, 1] * src[id1] +
                         self.Ke[i][1, 2] * src[id2] + self.Ke[i][1, 3] * src[id3]) * dt
            dst[id2] += (self.Ke[i][2, 0] * src[id0] + self.Ke[i][2, 1] * src[id1] +
                         self.Ke[i][2, 2] * src[id2] + self.Ke[i][2, 3] * src[id3]) * dt
            dst[id3] += (self.Ke[i][3, 0] * src[id0] + self.Ke[i][3, 1] * src[id1] +
                         self.Ke[i][3, 2] * src[id2] + self.Ke[i][3, 3] * src[id3]) * dt

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
        ite, iteMax = 0, 100
        while ite < iteMax and delta_new > (self.cg_epsilon**2) * delta_0:
            delta_new = self.cg_run_iteration(dt, delta_new)
            ite += 1

    # --------------------------------------------------------------------------------------------------- #

    # pcg
    # --------------------------------------------------------------------------------------------------- #
    @ti.func
    def M_inv_mult_r_Jacobi(self, dt, dst, src):
        # A = M + K * dt
        # M = diag(A)
        # M^-1r = M^-1 @ r
        for i in self.pcg_M:
            self.pcg_M[i] = 0.0
        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.pcg_M[id0] += self.Me[i][0, 0] + self.Ke[i][0, 0] * dt
            self.pcg_M[id1] += self.Me[i][1, 1] + self.Ke[i][1, 1] * dt
            self.pcg_M[id2] += self.Me[i][2, 2] + self.Ke[i][2, 2] * dt
            self.pcg_M[id3] += self.Me[i][3, 3] + self.Ke[i][3, 3] * dt
        for i in range(self.body.num_vertex):
            dst[i] = src[i] / self.pcg_M[i]

    @ti.kernel
    def pcg_before_ite(self, dt: float) -> float:
        # x = u(t)
        for i in range(self.body.num_vertex):
            # self.cg_x[i] = self.Vm[i]
            self.cg_x[i] = 0.0
        # Ax = A @ x
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

        for i in range(self.body.num_vertex):
            # r = b - A @ x
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]

        # d = M^-1 @ r
        self.M_inv_mult_r_Jacobi(dt, self.cg_d, self.cg_r)

        delta_new = self.dot(self.cg_r, self.cg_d)
        return delta_new

    @ti.kernel
    def pcg_run_iteration1(self, dt: float, delta: float) -> float:
        delta_new = delta
        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)
        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_vertex):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]

        # if ite % 50 == 0: r = b - A @ x
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)
        for i in range(self.body.num_vertex):
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]

        # s = M^-1 @ r
        self.M_inv_mult_r_Jacobi(dt, self.pcg_s, self.cg_r)
        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.pcg_s)
        beta = delta_new / delta_old
        for i in range(self.body.num_vertex):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
        return delta_new

    @ti.kernel
    def pcg_run_iteration2(self, dt: float, delta: float) -> float:
        delta_new = delta
        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)
        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_vertex):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]
            # if ite % 50 != 0: r = r - alpha * q
            self.cg_r[i] -= alpha * self.cg_Ad[i]

        # s = M^-1 @ r
        self.M_inv_mult_r_Jacobi(dt, self.pcg_s, self.cg_r)
        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.pcg_s)
        beta = delta_new / delta_old
        for i in range(self.body.num_vertex):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
        return delta_new

    def pcg(self, dt: float):
        delta_new = self.pcg_before_ite(dt)
        delta_0 = delta_new
        ite, iteMax = 0, 1000
        while ite < iteMax and delta_new > (self.cg_epsilon ** 2) * delta_0:
            if ite % 50 == 0:
                delta_new = self.pcg_run_iteration1(dt, delta_new)
            else:
                delta_new = self.pcg_run_iteration2(dt, delta_new)
            ite += 1
        print("ite:{}, delta_0:{}, delta_new:{}".format(ite, delta_0, delta_new))
    # --------------------------------------------------------------------------------------------------- #

    @ti.kernel
    def cgUpdateVm(self):
        for i in self.Vm:
            self.Vm[i] = self.cg_x[i]

    def calculate_diffusion(self, dt):
        self.calculate_M_and_K()
        self.compute_RHS()
        self.cg(dt)
        # self.pcg(dt)
        self.cgUpdateVm()

    def update_Vm(self, dt):
        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)

    @ti.kernel
    def apply_stimulation(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if (vert[i][0] - 8.5) * (vert[i][0] - 8.5) + (vert[i][1] - 5.) * (vert[i][1] - 5.) + \
                    vert[i][2] * vert[i][2] < (1. / 1.):
                self.I_ext[i] = 1.0
            # if (vert[i][0] - 8.5) * (vert[i][0] - 8.5) + (vert[i][1] + 15.) * (vert[i][1] + 15.) + \
            #                 vert[i][2] * vert[i][2] < (1. / 1.):
            #     self.I_ext[i] = 1.0

    @ti.kernel
    def cancel_stimulation(self):
        for i in self.I_ext:
            self.I_ext[i] = 0.0

    @ti.kernel
    def init_Vm_stimulation(self):
        for i in self.Vm:
            x, y, z = self.body.vertex[i][0], self.body.vertex[i][1], self.body.vertex[i][2]
            self.Vm[i] = tm.exp(-4. * ((x / 10)**2 + y**2 + z**2))
            # if x**2 + y**2 + z**2 < 10:
            #     self.Vm[i] = 1.0

    @ti.kernel
    def get_near_vertex_index(self, x: float, y: float, z: float) -> int:
        vert = ti.static(self.body.vertex)
        res = 0
        for i in vert:
            if (vert[i][0] - x)**2 + (vert[i][1] - y)**2 + (vert[i][2] - z)**2 < 1e-2:
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

    def debug_pcg(self, dt):
        self.calculate_M_and_K()
        self.debug_compute_RHS(dt)
        # self.pcg(dt)
        self.cg(dt)
        print(self.cg_x)
        self.debug_Ax(dt)
        # print(self.cg_Ax)
        # print(self.cg_b)
        # self.calculate_A(dt)
        # self.compute_RHS()

    @ti.kernel
    def calculate_A(self, dt: float):
        for i, j in self.cg_A:
            self.cg_A[i, j] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_A[id0, id0] += self.Me[i][0, 0] + self.Ke[i][0, 0] * dt
            self.cg_A[id0, id1] += self.Me[i][0, 1] + self.Ke[i][0, 1] * dt
            self.cg_A[id0, id2] += self.Me[i][0, 2] + self.Ke[i][0, 2] * dt
            self.cg_A[id0, id3] += self.Me[i][0, 3] + self.Ke[i][0, 3] * dt

            self.cg_A[id1, id0] += self.Me[i][1, 0] + self.Ke[i][1, 0] * dt
            self.cg_A[id1, id1] += self.Me[i][1, 1] + self.Ke[i][1, 1] * dt
            self.cg_A[id1, id2] += self.Me[i][1, 2] + self.Ke[i][1, 2] * dt
            self.cg_A[id1, id3] += self.Me[i][1, 3] + self.Ke[i][1, 3] * dt

            self.cg_A[id2, id0] += self.Me[i][2, 0] + self.Ke[i][2, 0] * dt
            self.cg_A[id2, id1] += self.Me[i][2, 1] + self.Ke[i][2, 1] * dt
            self.cg_A[id2, id2] += self.Me[i][2, 2] + self.Ke[i][2, 2] * dt
            self.cg_A[id2, id3] += self.Me[i][2, 3] + self.Ke[i][2, 3] * dt

            self.cg_A[id3, id0] += self.Me[i][3, 0] + self.Ke[i][3, 0] * dt
            self.cg_A[id3, id1] += self.Me[i][3, 1] + self.Ke[i][3, 1] * dt
            self.cg_A[id3, id2] += self.Me[i][3, 2] + self.Ke[i][3, 2] * dt
            self.cg_A[id3, id3] += self.Me[i][3, 3] + self.Ke[i][3, 3] * dt

    @ti.kernel
    def debug_Ax(self, dt: float):
        for i in self.cg_x:
            self.cg_x[i] = 1.
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

    @ti.kernel
    def debug_compute_RHS(self, dt: float):
        for i in self.cg_b:
            self.cg_b[i] = 0.0
        # A = M + K * dt
        # bi = \sum_j Aij
        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_b[id0] += (self.Me[i][0, 0] + self.Me[i][0, 1] +
                               self.Me[i][0, 2] + self.Me[i][0, 3])
            self.cg_b[id1] += (self.Me[i][1, 0] + self.Me[i][1, 1] +
                               self.Me[i][1, 2] + self.Me[i][1, 3])
            self.cg_b[id2] += (self.Me[i][2, 0] + self.Me[i][2, 1] +
                               self.Me[i][2, 2] + self.Me[i][2, 3])
            self.cg_b[id3] += (self.Me[i][3, 0] + self.Me[i][3, 1] +
                               self.Me[i][3, 2] + self.Me[i][3, 3])
            self.cg_b[id0] += (self.Ke[i][0, 0] + self.Ke[i][0, 1] +
                               self.Ke[i][0, 2] + self.Ke[i][0, 3]) * dt
            self.cg_b[id1] += (self.Ke[i][1, 0] + self.Ke[i][1, 1] +
                               self.Ke[i][1, 2] + self.Ke[i][1, 3]) * dt
            self.cg_b[id2] += (self.Ke[i][2, 0] + self.Ke[i][2, 1] +
                               self.Ke[i][2, 2] + self.Ke[i][2, 3]) * dt
            self.cg_b[id3] += (self.Ke[i][3, 0] + self.Ke[i][3, 1] +
                               self.Ke[i][3, 2] + self.Ke[i][3, 3]) * dt

    def show(self, sub_steps):
        windowLength = 1024
        lengthScale = min(windowLength, 512)
        light_distance = lengthScale / 25.
        vertex = self.body.vertex
        x_min = min(vertex[i][0] for i in range(vertex.shape[0]))
        x_max = max(vertex[i][0] for i in range(vertex.shape[0]))
        y_min = min(vertex[i][1] for i in range(vertex.shape[0]))
        y_max = max(vertex[i][1] for i in range(vertex.shape[0]))
        z_min = min(vertex[i][2] for i in range(vertex.shape[0]))
        z_max = max(vertex[i][2] for i in range(vertex.shape[0]))
        length = max(x_max - x_min, y_max - y_min, z_max - z_min)
        visualizeRatio = lengthScale / length / 10.
        center = np.array([(x_min + x_max) / 2., (y_min + y_max) / 2., (z_min + z_max) / 2.])  # * visualizeRatio

        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # camera.position(0.5, 1.0, 1.95)
        # camera.position(center[0] * 1, center[1] * 1, center[2] * 30)
        camera.position(0, 30, 50)
        camera.lookat(center[0], center[1], center[2])
        # print((center[0], center[1], center[2]))
        camera.fov(55)
        iter_time = 0
        while window.running:
            if iter_time == 0:
                self.apply_stimulation()
                iter_time += 1
            elif iter_time == 10:
                self.cancel_stimulation()
                iter_time += 1
            elif iter_time < 600:
                iter_time += 1
            elif iter_time == 600:
                iter_time = 0

            self.update(sub_steps)
            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            # scene.particles(pos, radius=0.02, color=(0, 1, 1))
            scene.mesh(vertex, indices=self.body.surfaces, two_sided=False, per_vertex_color=self.vertex_color)
            # scene.mesh(vertex, indices=self.body.surfaces, color=(1., 1., 1.), two_sided=False)

            # show the frame
            canvas.scene(scene)
            window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True)
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # edge
    edge_np = np.array(meshData['tetEdgeIds'], dtype=int)
    edge_np = edge_np.reshape((-1, 2))
    # surface tri index
    # surf_tri_np = np.array(meshData['tetSurfaceTriIds'], dtype=int)
    # surf_tri_np = surf_tri_np.reshape((-1, 3))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))

    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # num_edge_set
    num_edge_set_np = np.array(meshData['num_edge_set'], dtype=int)[0]
    # edge_set
    edge_set_np = np.array(meshData['edge_set'], dtype=int)
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
    bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)

    body = Body(pos_np, tet_np, edge_np, fiber_tet_np, sheet_tet_np, num_edge_set_np, edge_set_np, num_tet_set_np,
                tet_set_np, bou_tag_dirichlet_np, bou_tag_neumann_np)

    dr = diffusion_reaction(body)
    sigma_para = 5e-1
    dr.sigma_f = sigma_para
    dr.sigma_s = sigma_para
    dr.sigma_n = sigma_para
    # dr.init_Vm_stimulation()
    # dr.apply_stimulation()
    dr.update_color()
    dr.show(2)
    print(dr.Vm)

    # ind = dr.get_near_vertex_index(10, 1, 1)
    # print(ind)

    # dr.debug_pcg(0.01)
    # path_A = "path_A.txt"
    # path_b = "path_b.txt"
    # np.savetxt(path_A, dr.cg_A.to_numpy())
    # np.savetxt(path_b, dr.cg_b.to_numpy())
    # print(dr.cg_A)
    # print(dr.cg_b)
    # print(dr.cg_x)
    # print(dr.body.num_vertex)
    # print(dr.cg_A.shape)

    # total_time = 100
    # sub_time = 10
    # table_x = np.linspace(0, total_time, total_time * sub_time + 1)
    # ind = 6487  # 6487, 2647
    # vm = dr.Vm[ind]
    # table_y = np.array(vm)
    # # begin_time = time.time()
    # dt = 1.0 / sub_time
    # dr.init_Vm_stimulation()
    # for tt in range(total_time):
    #     for st in range(sub_time):
    #         # if tt == 0 and st == 0:
    #         #     dr.apply_stimulation()
    #         # if tt == 0 and st == 10:
    #         #     dr.cancel_stimulation()
    #         dr.update_Vm(dt)
    #         vm = dr.Vm[ind]
    #         table_y = np.append(table_y, vm)
    #         # end_time = time.time()
    #         # print(tt * sub_time + st)
    #         # print(end_time - begin_time)
    #         # begin_time = end_time
    # plt.plot(table_x, table_y)
    # # plt.axis([0, total_time, 0, 1])
    # plt.show()
    # print(table_y)
    # # print(dr.D[1000])
