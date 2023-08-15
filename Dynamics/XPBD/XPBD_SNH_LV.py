import taichi as ti
import numpy as np
import taichi.math as tm
from data.LV1 import meshData
from Geometry.body_LV import Body


@ti.data_oriented
class XPBD_SNH_with_active:
    def __init__(self, body: Body, num_pts_np: np.ndarray,
                 Youngs_modulus=1000.0, Poisson_ratio=0.49,
                 dt=1. / 6., numSubsteps=1, numPosIters=1):
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

        self.num_bou_endo_face = self.body.num_bou_endo_face
        self.bou_endo_face = self.body.bou_endo
        self.normal_bou_endo_face = ti.Vector.field(3, float, shape=(self.num_bou_endo_face,))

        self.num_bou_epi_face = self.body.num_bou_epi_face
        self.bou_epi_face = self.body.bou_epi
        self.normal_bou_epi_face = ti.Vector.field(3, float, shape=(self.num_bou_epi_face,))
        self.get_bou_face_normal()
        self.p_endo_lv = 15          # 15.0


        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber_np)
        # self.F = ti.Matrix.field(3, 3, float, shape=(self.num_element,))

    @ti.kernel
    def get_bou_face_normal(self):
        for i in self.bou_endo_face:
            id0, id1, id2 = self.bou_endo_face[i][0], self.bou_endo_face[i][1], self.bou_endo_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_face[i] = tm.normalize(n1)

        for i in self.bou_epi_face:
            id0, id1, id2 = self.bou_epi_face[i][0], self.bou_epi_face[i][1], self.bou_epi_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_epi_face[i] = tm.normalize(n1)

    @ti.kernel
    def init(self):
        for i in self.pos:
            self.mass[i] = 0.0
            self.f_ext[i] = tm.vec3(0, 0, 0)
            # self.f_ext[i] = tm.vec3(0, -10.0, 0)

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            for j in ti.static(range(4)):
                eid = self.elements[i][j]
                self.mass[eid] += pm

        for i in self.pos:
            self.invMass[i] = 1.0 / self.mass[i]

    def update(self):
        self.update_Ta()
        for _ in range(self.numSubsteps):
            self.sub_step()

    @ti.kernel
    def update_Ta(self):
        epsilon_0 = 1
        k_Ta = 47.9   # kPa
        for i in self.pos:
            V = self.body.Vm[i]
            epsilon = 10 * epsilon_0
            if V < 0.05:
                epsilon = epsilon_0
            Ta_old = self.body.ver_Ta[i]
            Ta_new = self.dt * epsilon * k_Ta * V + Ta_old
            Ta_new /= (1 + self.dt * epsilon)
            self.body.ver_Ta[i] = Ta_new

        for i in self.elements:
            vid = tm.ivec4(0, 0, 0, 0)
            ver_mass = tm.vec4(0, 0, 0, 0)
            sum_mass = 0.0
            for j in ti.static(range(4)):
                vid[j] = self.elements[i][j]
                ver_mass[j] = self.mass[vid[j]]
                sum_mass += ver_mass[j]
            self.tet_Ta[i] = 0.0
            for j in ti.static(range(4)):
                self.tet_Ta[i] += ver_mass[j] / sum_mass * self.body.ver_Ta[vid[j]]

    def sub_step(self):
        self.preSolve()
        self.solve_Gauss_Seidel_GPU()
        self.postSolve()

    @ti.kernel
    def preSolve(self):
        pos, vel = ti.static(self.pos, self.vel)
        for i in self.f_ext:
            self.f_ext[i] = 0.0

        for i in self.bou_endo_face:
            id0, id1, id2 = self.bou_endo_face[i][0], self.bou_endo_face[i][1], self.bou_endo_face[i][2]
            vert0, vert1, vert2 = self.pos[id0], self.pos[id1], self.pos[id2]
            p1 = vert1 - vert0
            p2 = vert2 - vert0
            n1 = tm.cross(p1, p2)
            self.normal_bou_endo_face[i] = tm.normalize(n1)
            self.f_ext[id0] += -1.0 * self.p_endo_lv * self.normal_bou_endo_face[i] / 3.0
            self.f_ext[id1] += -1.0 * self.p_endo_lv * self.normal_bou_endo_face[i] / 3.0
            self.f_ext[id2] += -1.0 * self.p_endo_lv * self.normal_bou_endo_face[i] / 3.0

        for i in self.pos:
            self.prevPos[i] = pos[i]
            vel[i] += self.h * self.f_ext[i] * self.invMass[i]
            pos[i] += self.h * vel[i]

        # for i in self.elements:
        #     for j in ti.static(range(3)):
        #         self.Lagrange_multiplier[i, j] = 0.0

    @ti.kernel
    def postSolve(self):
        pos, vel = ti.static(self.pos, self.vel)
        # for i in self.pos:
        #     if pos[i][1] < 0.0:
        #         pos[i][1] = 0.0
        #         v = self.prevPos[i] - pos[i]
        #         pos[i][0] += v[0] * tm.min(1.0, self.h * self.friction)
        #         pos[i][2] += v[2] * tm.min(1.0, self.h * self.friction)

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
                self.pos[i][1] = self.prevPos[i][1]
                # self.pos[i] = self.prevPos[i]

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
            self.applyToElem(i, C, devCompliance)

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
            self.applyToElem(i, C, volCompliance)

            # Iff = 1
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

            g[i, 1] = tm.vec3(0., 0., 0.)
            g[i, 1] += dIff0 * (C_inv * ir[i][0, 0])
            g[i, 1] += dIff1 * (C_inv * ir[i][0, 1])
            g[i, 1] += dIff2 * (C_inv * ir[i][0, 2])

            g[i, 2] = tm.vec3(0., 0., 0.)
            g[i, 2] += dIff0 * (C_inv * ir[i][1, 0])
            g[i, 2] += dIff1 * (C_inv * ir[i][1, 1])
            g[i, 2] += dIff2 * (C_inv * ir[i][1, 2])

            g[i, 3] = tm.vec3(0., 0., 0.)
            g[i, 3] += dIff0 * (C_inv * ir[i][2, 0])
            g[i, 3] += dIff1 * (C_inv * ir[i][2, 1])
            g[i, 3] += dIff2 * (C_inv * ir[i][2, 2])

            if self.body.tet_Ta[i] > 0:
                self.applyToElem(i, C, 1.0 / self.body.tet_Ta[i])

    @ti.func
    def applyToElem(self, elemNr, C, compliance):
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
            dlambda = -C / (w + alpha)

        # without proj
        for i in ti.static(range(4)):
            eid = elem[elemNr][i]
            pos[eid] += g[elemNr, i] * (dlambda * invMass[eid])

        # with proj
        # # id = tm.ivec4(0, 0, 0, 0)
        # # for j in ti.static(range(4)):
        # #     id[j] = self.elements[elemNr][j]
        # # v1 = pos[id[1]] - pos[id[0]]
        # # v2 = pos[id[2]] - pos[id[0]]
        # # v3 = pos[id[3]] - pos[id[0]]
        # # Ds = tm.mat3(v1, v2, v3)
        # # Ds = Ds.transpose()
        # # F = Ds @ self.body.DmInv[elemNr]
        # for i in ti.static(range(4)):
        #     eid = elem[elemNr][i]
        #     proj_a = g[elemNr, i] * (dlambda * invMass[eid])
        #     proj_b = self.vert_fiber[eid]
        #     # proj_b = F @ self.vert_fiber[eid]
        #     # proj_a1 = (tm.dot(proj_a, proj_b)) / (tm.dot(proj_b, proj_b)) * proj_b
        #     proj_a1 = (tm.dot(proj_a, proj_b)) / (tm.sqrt(tm.dot(proj_a, proj_a))) * (tm.sqrt(tm.dot(proj_b, proj_b))) * proj_a
        #     proj_a2 = proj_a - proj_a1
        #     pos[eid] += proj_a2


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
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
    # body.show()
    sys = XPBD_SNH_with_active(body=body)
    sys.update_Gauss_Seidel()

