"""Utility functions to load data."""
from pkg_resources import resource_filename
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


RCUBE = resource_filename(__name__, "data/rcube.obj")
UVMAP = resource_filename(__name__, "data/util-mark6.png")
DTYPE = torch.float
DEVICE = "cpu"


def read_obj(fname):
    """Parse an obj file, return ndarray of correct data type."""
    v, vt, f, tf = [], [], [], []
    with open(fname) as fid:
        lines = fid.readlines()

    for line in lines:
        if "vt" in line:
            vt.append([float(i) for i in line.split()[1:]])
        elif "v" in line:
            v.append([float(i) for i in line.split()[1:]])
        if "f" in line:
            lf = [[int(j) for j in s.split("/")] for s in line.split()[1:]]
            v_faces, t_faces = zip(*lf)
            f.append(v_faces)
            tf.append(t_faces)

    verts = np.array(v, dtype=np.float32)
    uv = np.array(vt, dtype=np.float)
    faces = np.array(f, dtype=np.int64) - 1
    uvfaces = np.array(tf, dtype=np.int64) - 1
    return verts, faces, uv, uvfaces


class Rcube:
    """test object - a cube with the vertices rotated."""

    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (v, f, uv, uvf), t = read_obj(RCUBE), ToTensor()
        self.v = torch.from_numpy(v).to(device)
        self.uv = torch.from_numpy(uv).to(device)
        self.uvmap = t(Image.open(UVMAP).convert("RGB")).to(device)
        self.f = torch.from_numpy(f).to(device)
        self.uvf = torch.from_numpy(uvf).to(device)
        self.device = device

    def tris(self):
        """Triangles as tensors.
        returns tuple: (vertex_triangles, uv_triangles)
        """
        return self.v[self.f], self.uv[self.uvf]

    def uvmap(self):
        """Return the uvmap image as PIL image """
        t = ToPILImage()
        return t(self.uvmap)


def area2d(a, b, c):
    """
    Vectorised area of 2d parallelogram (divide by 2 for triangle)
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    NB: colinear points result in zero area.
    """
    return (b[0] - a[:, 0]) * (c[1] - a[:, 1]) - \
        (b[1] - a[:, 1]) * (c[0] - a[:, 0])


def mask(w1, w2, w3):
    """
    From weighted triangles, decide if all are greater than zero.
    NOTE: is this operation out of graph as non - differentiable?
    """
    stacked = torch.stack([w1, w2, w3])
    clamped = torch.clamp(stacked, min=0)
    product = torch.prod(clamped, dim=0)
    _mask = torch.zeros_like(product, dtype=torch.long, device=DEVICE)
    _mask[product > 0] = 1
    return _mask


def pts_in_tri(pts, tri):
    """
    Return the barycentric coords of each point in pts for the triangle tri,
    iff the point is in the triangle.
    tri : tensor 3 x 2 (3 points by (x, y))
    pts: tensor n x 2 (n points by (x, y))

    return
        tensor: m x 5 (m x [x, y, b1, b2, b3]), where m is the number of points
        that were found to be in the triangle. If no points
    """

    # area of tri
    w = area2d(tri[None, 0], tri[1], tri[2])

    if torch.isclose(w, torch.zeros_like(w)):
        # print("Triangle is degenerate")
        return None

    # signed weighted area of subtriangles, NB: any could be zero
    pAB = area2d(pts, tri[0], tri[1]) * w
    pBC = area2d(pts, tri[1], tri[2]) * w
    pCA = area2d(pts, tri[2], tri[0]) * w

    # mask for pts that are in the triangle, if zero don't calc barycentrics.
    m = mask(pAB, pBC, pCA)

    if torch.allclose(m, torch.zeros_like(m)):
        # print("No points in triangle.")
        return None

    bx = pBC[m > 0] / pAB[m > 0]
    by = pCA[m > 0] / pAB[m > 0]
    bz = 1.0 - bx - by
    return torch.cat(
        [pts[m > 0], bx[..., None], by[..., None], bz[..., None]], dim=1)


if __name__ == "__main__":
    rcube = Rcube()
    tris, uvs = rcube.tris()
    print(tris.device)
    print(uvs.device)
    print(rcube.device)
    rcube.uvmap().save("uvmap.jpg")
