import taichi as ti
import numpy as np
import taichi.math as tm
from data.cube import meshData
import tool.geometrytool as geo


@ti.data_oriented
class Body:
    def __init__(self, vert_np: np.ndarray, tet_np: np.ndarray, edge_np: np.ndarray, tet_fiber_np: np.ndarray,
                 tet_sheet_np: np.ndarray, num_edge_set_np, edge_set_np: np.ndarray, num_tet_set_np,
                 tet_set_np: np.ndarray, bou_tag_dirichlet_np: np.ndarray, bou_tag_neumann_np: np.ndarray) -> None:
        # len(vertex[0]) = 3, len(vertex) = num_vert
        self.density = 1000.0
        self.num_vertex = len(vert_np)
        self.vertex = ti.Vector.field(3, dtype=float, shape=(self.num_vertex, ))
        self.vertex.from_numpy(vert_np)
        self.np_vertex = vert_np
        # self.scale_vertex(100)
        # len(elements[0]) = 4, len(elements) = num_tet
        self.num_tet = len(tet_np)
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=(self.num_tet, ))
        self.elements.from_numpy(tet_np)
        self.np_elements = tet_np
        # edge constraint
        self.num_edge = len(edge_np)
        self.edge = ti.Vector.field(2, dtype=int, shape=(self.num_edge,))
        self.edge.from_numpy(edge_np)
        # tet_fiber
        self.tet_fiber = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_fiber.from_numpy(tet_fiber_np)
        # tet_sheet
        self.tet_sheet = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_sheet.from_numpy(tet_sheet_np)
        # num_edge_set
        self.num_edge_set = ti.field(int, ())
        self.num_edge_set[None] = num_edge_set_np
        # edge_set
        self.edge_set = ti.field(int, shape=(self.num_edge,))
        self.edge_set.from_numpy(edge_set_np)
        # num_tet_set
        self.num_tet_set = ti.field(int, ())
        self.num_tet_set[None] = num_tet_set_np
        # tet_set
        self.tet_set = ti.field(int, shape=(self.num_tet,))
        self.tet_set.from_numpy(tet_set_np)
        # bou_tag1: dirichlet boundary condition
        self.bou_tag_dirichlet = ti.field(int, shape=(self.num_vertex,))
        self.bou_tag_dirichlet.from_numpy(bou_tag_dirichlet_np)
        # bou_tag2: neumann boundary condition
        self.bou_tag_neumann = ti.field(int, shape=(self.num_vertex,))
        self.bou_tag_neumann.from_numpy(bou_tag_neumann_np)

        # variables for visualization
        surfaces = geo.get_surface_from_tet(vertex=vert_np, elements=tet_np)
        self.surfaces = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.surfaces.from_numpy(surfaces.reshape(-1))

        self.Dm = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInvT = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.init_DmInv()

        self.vel = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.init_vel()
        self.Vm = ti.field(float, shape=(self.num_vertex,))

        # volume
        self.volume = ti.field(float, self.num_tet)
        self.init_volume()
        # # 顶点fiber方向
        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber)
        # # 四面体fiber方向
        # self.tet_fiber = ti.Vector.field(3, float, shape=(self.num_tet,))
        # # 从顶点采样到四面体
        # self.sample_tet_fiber()

        self.tet_Ta = ti.field(float, shape=(self.num_tet,))
        self.ver_Ta = ti.field(float, shape=(self.num_vertex,))
        self.init_electrophysiology()

    @ti.kernel
    def get_fiber(self):
        for i in self.tet_fiber:
            self.tet_fiber[i] = tm.vec3(0.0, 1.0, 0.0)
            self.tet_sheet[i] = tm.vec3(1.0, 0.0, 0.0)

    @ti.kernel
    def init_electrophysiology(self):
        for i in self.elements:
            self.tet_Ta[i] = 60.0

        for i in self.vertex:
            self.ver_Ta[i] = 60.0

    @ti.kernel
    def set_Ta(self, value: float):
        for i in self.elements:
            self.tet_Ta[i] = value

        for i in self.vertex:
            self.ver_Ta[i] = value

    def get_min_y(self):
        min_y = 10000000.0
        # print("nv:", self.num_vertex)
        for i in range(self.num_vertex):
            # print("i:", i)
            if min_y > self.vertex[i][1]:
                min_y = self.vertex[i][1]
        print(min_y)
        return min_y

    @ti.kernel
    def scale_vertex(self, scale: float):
        for i in range(self.num_vertex):
            self.vertex[i] *= scale

    @ti.kernel
    def init_DmInv(self):
        Dm, vertex, tet = ti.static(self.Dm, self.vertex, self.elements)
        # 0412 -
        for i in range(self.num_tet):
            Dm[i][0, 0] = vertex[tet[i][1]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 0] = vertex[tet[i][1]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 0] = vertex[tet[i][1]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 1] = vertex[tet[i][2]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 1] = vertex[tet[i][2]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 1] = vertex[tet[i][2]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 2] = vertex[tet[i][3]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 2] = vertex[tet[i][3]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 2] = vertex[tet[i][3]][2] - vertex[tet[i][0]][2]

        for i in range(self.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()
            self.DmInvT[i] = self.DmInv[i].transpose()

    @ti.kernel
    def init_vel(self):
        for i in self.vel:
            self.vel[i] = tm.vec3([0., 0., 0.])

    @ti.kernel
    def init_volume(self):
        for i in self.volume:
            self.volume[i] = ti.abs(self.Dm[i].determinant()) / 6.0

    @ti.kernel
    def sample_tet_fiber(self):
        for i in range(self.num_tet):
            self.tet_fiber[i] = self.vert_fiber[self.elements[i][0]] + self.vert_fiber[self.elements[i][1]] + \
                                self.vert_fiber[self.elements[i][2]] + self.vert_fiber[self.elements[i][3]]
            self.tet_fiber[i] /= 4.0
            self.tet_fiber[i] /= tm.length(self.tet_fiber[i])

    @ti.kernel
    def translation(self, x: float, y: float, z: float):
        for i in self.vertex:
            self.vertex[i][0] += x
            self.vertex[i][1] += y
            self.vertex[i][2] += z

    @ti.kernel
    def set_Vm(self):
        for i in self.Vm:
            self.Vm[i] = self.vertex[i][1] * 30.0
            self.ver_Ta[i] = 0.5 * self.Vm[i]

        for i in self.elements:
            id0, id1, id2, id3 = self.elements[i][0], self.elements[i][1], self.elements[i][2], self.elements[i][3]
            self.tet_Ta[i] = (self.ver_Ta[id0] + self.ver_Ta[id1] + self.ver_Ta[id2] + self.ver_Ta[id3]) / 4.0

    @ti.kernel
    def get_dirichlet_bou(self):
        for i in self.vertex:
            if abs(self.vertex[i][1]) < 1e-12:
                self.bou_tag_dirichlet[i] = 1


@ti.data_oriented
class XPBD_SNH_with_active:
    def __init__(self, body: Body, num_pts_np: np.ndarray,
                 Youngs_modulus=100.0, Poisson_ratio=0.49,
                 dt=1. / 10., numSubsteps=1, numPosIters=10):
        self.body = body
        self.num_vertex = self.body.num_vertex
        self.num_element = self.body.num_tet
        self.dt = dt
        self.numSubsteps = numSubsteps
        self.h = self.dt / self.numSubsteps
        self.numPosIters = numPosIters
        self.friction = 1000.0
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.mass = ti.field(float, shape=(self.num_vertex,))
        self.mass_weight = ti.field(float, shape=(self.num_vertex,))
        self.gravity = tm.vec3(0.0, 0.0, 0.0)
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.pos = self.body.vertex
        self.prevPos = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.vel = self.body.vel
        self.dx = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.elements = self.body.elements
        self.invMass = ti.field(float, shape=(self.num_vertex,))
        self.vol = self.body.volume
        self.invVol = ti.field(float, shape=(self.num_element,))
        self.grads = ti.Vector.field(3, float, shape=(self.num_element, 4))
        self.tol_tet_set = self.body.num_tet_set[None]
        self.num_pts = ti.field(int, shape=(self.tol_tet_set,))
        self.num_pts.from_numpy(num_pts_np)
        self.invLa = 1.0 / self.LameLa
        self.invMu = 1.0 / self.LameMu
        self.tet_Ta = body.tet_Ta
        self.init()

        self.vert_tol_vol = ti.field(float, shape=(self.num_vertex,))
        self.vert_active_force = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.dx_ = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.Lagrange_multiplier = ti.field(float, shape=(self.num_element, 3))

        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber_np)
        # self.F = ti.Matrix.field(3, 3, float, shape=(self.num_element,))

    @ti.kernel
    def init(self):
        for i in self.pos:
            self.mass[i] = 0.0
            self.f_ext[i] = self.gravity

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            for j in ti.static(range(4)):
                eid = self.elements[i][j]
                self.mass[eid] += pm
        #         self.mass[eid] += pm * self.vol[i]
        #         self.mass_weight[eid] += self.vol[i]
        #
        # for i in self.pos:
        #     self.mass[i] /= self.mass_weight[i]

        for i in self.pos:
            self.mass[i] = 0.1

        for i in self.pos:
            self.invMass[i] = 1.0 / self.mass[i]

    def update(self):
        for _ in range(self.numSubsteps):
            self.sub_step()

    def sub_step(self):
        # self.cal_active_force()
        self.preSolve()
        self.solve_Gauss_Seidel_GPU()
        self.postSolve()

    @ti.kernel
    def cal_active_force(self):
        """
            force = (d psi / d F) @ (d F / dx)
            psi = Ta / 2 * (I_ff - 1)
            (d psi) / (d F) = Ta * F @ f0 @ f0^T
        """
        pos, ir, tet = ti.static(self.pos, self.body.DmInv, self.elements)
        for i in self.vert_active_force:
            self.vert_active_force[i] = 0.0
            self.vert_tol_vol[i] = 0.0

        for i in self.elements:
            vid = tm.ivec4(0, 0, 0, 0)
            for j in ti.static(range(4)):
                vid[j] = tet[i][j]
            f0 = self.body.tet_fiber[i]
            v1 = pos[vid[1]] - pos[vid[0]]
            v2 = pos[vid[2]] - pos[vid[0]]
            v3 = pos[vid[3]] - pos[vid[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            f = F @ f0
            dpsi = self.tet_Ta[i] * (f.outer_product(f0))
            # print()
            # print(self.tet_Ta[i])
            dpsidx = ti.Matrix([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]], float)
            tmp = tm.vec3(0., 0., 0.)
            for j in ti.static(range(3)):
                tmp[0] = 0.0
                tmp[1] = 0.0
                tmp[2] = 0.0
                for m in ti.static(range(3)):
                    tmp[0] += ir[i][j, m] * dpsi[0, m]
                    tmp[1] += ir[i][j, m] * dpsi[1, m]
                    tmp[2] += ir[i][j, m] * dpsi[2, m]
                dpsidx[j + 1, 0] += tmp[0]
                dpsidx[j + 1, 1] += tmp[1]
                dpsidx[j + 1, 2] += tmp[2]
                dpsidx[0, 0] -= tmp[0]
                dpsidx[0, 1] -= tmp[1]
                dpsidx[0, 2] -= tmp[2]

            vol = F.determinant()
            vol = abs(vol) / 6.0
            # for j in ti.static(range(4)):
            #     for m in ti.static(range(3)):
            #         self.vert_active_force[vid[j]][m] -= dpsidx[j, m] * vol
            #     self.vert_tol_vol[vid[j]] += vol
            #     # print(self.vert_tol_vol[vid[j]], vol)

            for j in ti.static(range(4)):
                for m in ti.static(range(3)):
                    self.vert_active_force[vid[j]][m] -= dpsidx[j, m] * vol

        # for i in range(self.num_vertex):
        #     self.vert_active_force[i] /= self.vert_tol_vol[i]

    @ti.kernel
    def preSolve(self):
        pos, vel = ti.static(self.pos, self.vel)
        for i in self.f_ext:
            # self.vert_active_force[i] = tm.vec3(0.0, -0.5 * self.body.ver_Ta[i], 0.0)
            # self.f_ext[i] = self.gravity + self.vert_active_force[i]
            self.f_ext[i] = self.gravity

        for i in self.pos:
            self.prevPos[i] = pos[i]
            vel[i] += self.h * self.f_ext[i] * self.invMass[i]
            pos[i] += self.h * vel[i]
            self.dx_[i] = pos[i] - self.prevPos[i]

        for i in self.elements:
            for j in ti.static(range(3)):
                self.Lagrange_multiplier[i, j] = 0.0

    @ti.kernel
    def postSolve(self):
        pos, vel = ti.static(self.pos, self.vel)
        for i in self.pos:
            if pos[i][1] < 0.0:
                pos[i][1] = 0.0
                v = self.prevPos[i] - pos[i]
                pos[i][0] += v[0] * tm.min(1.0, self.h * self.friction)
                pos[i][2] += v[2] * tm.min(1.0, self.h * self.friction)

        for i in pos:
            vel[i] = (pos[i] - self.prevPos[i]) / self.h

    def solve_Gauss_Seidel_GPU(self):
        for _ in range(self.numPosIters):
            left, right = 0, 0
            for set_id in range(self.tol_tet_set):
                if set_id == 0:
                    left = 0
                    right = self.num_pts[0]
                else:
                    left += self.num_pts[set_id - 1]
                    right += self.num_pts[set_id]
                self.solve_elem_Gauss_Seidel_GPU(left, right)

        self.solve_dirichlet_boundary()

    @ti.kernel
    def solve_dirichlet_boundary(self):
        for i in self.body.vertex:
            if self.body.bou_tag_dirichlet[i] == 1:
                # self.pos[i][1] = self.prevPos[i][1]
                self.pos[i] = self.prevPos[i]

        # for i in self.body.bou_epi:
        #     id0, id1, id2 = self.body.bou_epi[i][0], self.body.bou_epi[i][1], self.body.bou_epi[i][2]
        #     self.pos[id0] = self.prevPos[id0]
        #     self.pos[id1] = self.prevPos[id1]
        #     self.pos[id2] = self.prevPos[id2]

    @ti.kernel
    def solve_elem_Gauss_Seidel_GPU(self, left: int, right: int):
        pos, vel, tet, ir, g = ti.static(self.pos, self.vel, self.elements, self.body.DmInv, self.grads)
        for i in range(left, right):
            C = 0.0
            devCompliance = 1.0 * self.invMu
            volCompliance = 1.0 * self.invLa

            # tr(F) = 3
            id = tm.ivec4(0, 0, 0, 0)
            for j in ti.static(range(4)):
                id[j] = tet[i][j]

            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]

            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            r_s = tm.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
                          + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
                          + v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])
            r_s_inv = 1.0 / r_s
            g[i, 1] = tm.vec3(0., 0., 0.)
            g[i, 1] += F_col0 * (r_s_inv * ir[i][0, 0])
            g[i, 1] += F_col1 * (r_s_inv * ir[i][0, 1])
            g[i, 1] += F_col2 * (r_s_inv * ir[i][0, 2])

            g[i, 2] = tm.vec3(0., 0., 0.)
            g[i, 2] += F_col0 * (r_s_inv * ir[i][1, 0])
            g[i, 2] += F_col1 * (r_s_inv * ir[i][1, 1])
            g[i, 2] += F_col2 * (r_s_inv * ir[i][1, 2])

            g[i, 3] = tm.vec3(0., 0., 0.)
            g[i, 3] += F_col0 * (r_s_inv * ir[i][2, 0])
            g[i, 3] += F_col1 * (r_s_inv * ir[i][2, 1])
            g[i, 3] += F_col2 * (r_s_inv * ir[i][2, 2])

            C = r_s
            self.applyToElem(i, C, devCompliance, 0)

            # det(F) = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            dF0 = F_col1.cross(F_col2)
            dF1 = F_col2.cross(F_col0)
            dF2 = F_col0.cross(F_col1)

            g[i, 1] = tm.vec3(0., 0., 0.)
            g[i, 1] += dF0 * ir[i][0, 0]
            g[i, 1] += dF1 * ir[i][0, 1]
            g[i, 1] += dF2 * ir[i][0, 2]

            g[i, 2] = tm.vec3(0., 0., 0.)
            g[i, 2] += dF0 * ir[i][1, 0]
            g[i, 2] += dF1 * ir[i][1, 1]
            g[i, 2] += dF2 * ir[i][1, 2]

            g[i, 3] = tm.vec3(0., 0., 0.)
            g[i, 3] += dF0 * ir[i][2, 0]
            g[i, 3] += dF1 * ir[i][2, 1]
            g[i, 3] += dF2 * ir[i][2, 2]

            vol = F.determinant()
            C = vol - 1.0 - volCompliance / devCompliance
            self.applyToElem(i, C, volCompliance, 1)

            Iff = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            f0 = self.body.tet_fiber[i]
            f = F @ f0
            C = tm.sqrt(f.dot(f))
            C_inv = 1.0 / C
            dIff = f0.outer_product(f0)
            dIff0 = tm.vec3(dIff[0, 0], dIff[1, 0], dIff[2, 0])
            dIff1 = tm.vec3(dIff[0, 1], dIff[1, 1], dIff[2, 1])
            dIff2 = tm.vec3(dIff[0, 2], dIff[1, 2], dIff[2, 2])

            # g[i, 1] = tm.vec3(0., 0., 0.)
            # g[i, 1] += dIff0 * (C_inv * ir[i][0, 0])
            # g[i, 1] += dIff1 * (C_inv * ir[i][0, 1])
            # g[i, 1] += dIff2 * (C_inv * ir[i][0, 2])
            #
            # g[i, 2] = tm.vec3(0., 0., 0.)
            # g[i, 2] += dIff0 * (C_inv * ir[i][1, 0])
            # g[i, 2] += dIff1 * (C_inv * ir[i][1, 1])
            # g[i, 2] += dIff2 * (C_inv * ir[i][1, 2])
            #
            # g[i, 3] = tm.vec3(0., 0., 0.)
            # g[i, 3] += dIff0 * (C_inv * ir[i][2, 0])
            # g[i, 3] += dIff1 * (C_inv * ir[i][2, 1])
            # g[i, 3] += dIff2 * (C_inv * ir[i][2, 2])

            g[i, 1] = tm.vec3(0., 0., 0.)
            g[i, 1] -= dIff0 * (C_inv * ir[i][0, 0])
            g[i, 1] -= dIff1 * (C_inv * ir[i][0, 1])
            g[i, 1] -= dIff2 * (C_inv * ir[i][0, 2])

            g[i, 2] = tm.vec3(0., 0., 0.)
            g[i, 2] -= dIff0 * (C_inv * ir[i][1, 0])
            g[i, 2] -= dIff1 * (C_inv * ir[i][1, 1])
            g[i, 2] -= dIff2 * (C_inv * ir[i][1, 2])

            g[i, 3] = tm.vec3(0., 0., 0.)
            g[i, 3] -= dIff0 * (C_inv * ir[i][2, 0])
            g[i, 3] -= dIff1 * (C_inv * ir[i][2, 1])
            g[i, 3] -= dIff2 * (C_inv * ir[i][2, 2])

            if self.body.tet_Ta[i] > 0:
                self.applyToElem(i, C, 1.0 / self.body.tet_Ta[i], 2)

    @ti.func
    def applyToElem(self, elemNr, C, compliance, cid):
        g, pos, elem, h, invVol, invMass = ti.static(self.grads, self.pos, self.elements, self.h, self.invVol,
                                                     self.invMass)
        g[elemNr, 0] = tm.vec3(0., 0., 0.)
        g[elemNr, 0] -= g[elemNr, 1]
        g[elemNr, 0] -= g[elemNr, 2]
        g[elemNr, 0] -= g[elemNr, 3]

        w = 0.0
        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            w += (g[elemNr, i][0] * g[elemNr, i][0] + g[elemNr, i][1] * g[elemNr, i][1] + g[elemNr, i][2] * g[elemNr, i][2]) * invMass[eid]

        dlambda = 0.0
        if w != 0.0:
            alpha = compliance / h / h * invVol[elemNr]
            dlambda = (0.0 - C - alpha * self.Lagrange_multiplier[elemNr, cid]) / (w + alpha)

        self.Lagrange_multiplier[elemNr, cid] += dlambda

        # without proj
        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            pos[eid] += g[elemNr, i] * (dlambda * invMass[eid])


def example3():
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # edge
    edge_np = np.array(meshData['tetEdgeIds'], dtype=int)
    edge_np = edge_np.reshape((-1, 2))
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
    body.get_fiber()

    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_sys = XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np)

    # set example parameter
    body.set_Vm()
    body.get_dirichlet_bou()

    open_gui = True
    # open_gui = False
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    if open_gui:
        # init the window, canvas, scene and camera
        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color((1., 1., 1.))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(3.41801597, 1.65656349, 3.05081163)
        camera.lookat(2.7179826, 1.31246826, 2.42507068)
        camera.up(0., 1., 0.)

        iter_time = 0
        while window.running:
            dynamics_sys.update()

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            # scene.particles(body.vertex, radius=0.02, color=(0, 1, 1))
            # scene.mesh(body.vertex, indices=body.surfaces, per_vertex_color=vert_color, two_sided=False)
            scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, color=(0.5, 0.5, 0.5))
            # scene.lines(force_field, color=(0., 0.0, 1.), width=2.0)
            # scene.lines(pos_field, color=(0., 0.0, 1.), width=2.0)

            # show the frame
            canvas.scene(scene)
            window.show()

        # print(camera.curr_position)
        # print(camera.curr_lookat)
        # print(camera.curr_up)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    example3()


