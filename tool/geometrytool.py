import numpy as np


def get_surface_from_tet(vertex: np.ndarray, elements: np.ndarray):
    """get the triangles of the mesh, and the outer surface of the mesh"""
    surfaces = set()
    indexes = [(0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)]
    for id_ele, ele in enumerate(elements):
        center = np.array([0., 0., 0.])
        faces = []
        for i in range(4):
            center += vertex[ele[i]]
        for i in range(3):
            center[i] /= 4.
        for index in indexes:
            (i, j, k) = index
            v0, v1, v2 = vertex[ele[i]], vertex[ele[j]], vertex[ele[k]]
            v0_1 = v1 - v0
            v0_2 = v2 - v0
            vc_0 = v0 - center
            norm = np.cross(v0_1, v0_2)
            sign = np.dot(vc_0, norm)
            if sign > 0:
                faces.append((ele[i], ele[j], ele[k]))
            else:
                faces.append((ele[i], ele[k], ele[j]))

        for face in faces:
            face_inv = (face[0], face[2], face[1])
            if face_inv in surfaces:
                surfaces.remove(face_inv)
            else:
                surfaces.add(face)

    surfaces = np.array(list(surfaces))
    return surfaces
