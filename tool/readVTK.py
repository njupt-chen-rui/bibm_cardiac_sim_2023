import numpy as np
import vtk


def read_vtk(filename="../data/heart1/heart_origin_bou_tag.vtk"):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()

    points = np.array(reader.GetOutput().GetPoints().GetData())
    cells = np.array(reader.GetOutput().GetCells().GetData())
    bou_tag = np.array(reader.GetOutput().GetPointData().GetArray("bou_tag"))
    mesh = np.array([cells[i * 5 + 1: i * 5 + 5] for i in range(int(cells.shape[0] / 5))])

    # print(len(mesh[0]))
    # print(mesh.shape)
    return points, mesh, bou_tag
