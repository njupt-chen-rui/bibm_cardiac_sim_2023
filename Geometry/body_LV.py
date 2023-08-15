"""
    Body consists of vertexes and elements
"""
import taichi as ti
import numpy as np
import tool.geometrytool as geo
import taichi.math as tm
from data.LV1 import meshData


@ti.data_oriented
class Body:
    def __init__(self, vert_np: np.ndarray, tet_np: np.ndarray, edge_np: np.ndarray, tet_fiber_np: np.ndarray,
                 tet_sheet_np: np.ndarray, num_edge_set_np, edge_set_np: np.ndarray, num_tet_set_np,
                 tet_set_np: np.ndarray, bou_tag_dirichlet_np: np.ndarray, bou_tag_neumann_np: np.ndarray,
                 bou_endo_np: np.ndarray, bou_epi_np: np.ndarray) -> None:
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

        self.num_bou_endo_face = len(bou_endo_np)
        self.bou_endo = ti.Vector.field(3, int, shape=(self.num_bou_endo_face,))
        self.bou_endo.from_numpy(bou_endo_np)

        self.num_bou_epi_face = len(bou_epi_np)
        self.bou_epi = ti.Vector.field(3, int, shape=(self.num_bou_epi_face,))
        self.bou_epi.from_numpy(bou_epi_np)

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

    def show(self):
        windowLength = 1024
        lengthScale = min(windowLength, 512)
        light_distance = lengthScale / 25.

        x_min = min(self.vertex[i][0] for i in range(self.vertex.shape[0]))
        x_max = max(self.vertex[i][0] for i in range(self.vertex.shape[0]))
        y_min = min(self.vertex[i][1] for i in range(self.vertex.shape[0]))
        y_max = max(self.vertex[i][1] for i in range(self.vertex.shape[0]))
        z_min = min(self.vertex[i][2] for i in range(self.vertex.shape[0]))
        z_max = max(self.vertex[i][2] for i in range(self.vertex.shape[0]))
        length = max(x_max - x_min, y_max - y_min, z_max - z_min)
        visualizeRatio = lengthScale / length / 10.
        center = np.array([(x_min + x_max) / 2., (y_min + y_max) / 2., (z_min + z_max) / 2.])  # * visualizeRatio

        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        camera.position(0.5, 1.0, 50)
        # camera.position(center[0] * 1, center[1] * 1, center[2] * 30)
        camera.lookat(center[0], center[1], center[2])
        camera.fov(55)
        while window.running:

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            # scene.particles(pos, radius=0.02, color=(0, 1, 1))
            scene.mesh(self.vertex, indices=self.surfaces, color=(1.0, 0, 0), two_sided=False)

            # show the frame
            canvas.scene(scene)
            window.show()


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
    # Youngs_Modulus = 1000.
    # Poisson_Ratio = 0.49
    # material = Stable_Neo_Hookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    # material = Stable_Neo_Hookean_with_active(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio,
    #                                           active_tension=60)
    # sys = PBD_with_Continuous_Materials(body, material, 10)
    #
    # sys.show()

    body.translation(0, 50, 0)
    for i in range(body.num_vertex):
        if body.vertex[i][1] < 0:
            print(body.vertex[i][1])
