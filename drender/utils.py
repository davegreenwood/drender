"""Utility functions to load data."""
from pkg_resources import resource_filename
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


RCUBE = resource_filename(__name__, "data/rcube.obj")
UVMAP = resource_filename(__name__, "data/util-mark6.png")
DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def assert_size(x, y):
    """Make sure x and y are equal powers of 2 """
    assert x == y
    assert x in [8, 16, 32, 64, 128, 256, 512, 1024]


def backface_cull(tris):
    """
    Return only the triangles that face forward. Tris are assumed to be
    in view space."""
    normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    m = torch.tensor([0, 0, 1.0], dtype=DTYPE, device=DEVICE) @ normals
    return tris[m > 0]


class Rcube:
    """test object - a cube with the vertices rotated."""

    def __init__(self):
        device = DEVICE
        (v, f, uv, uvf), t = read_obj(RCUBE), ToTensor()
        self.v = torch.from_numpy(v).to(device)
        self.uv = torch.from_numpy(uv).to(device)
        self.uvmap = t(Image.open(UVMAP).transpose(
            Image.FLIP_TOP_BOTTOM).convert("RGB")).to(device)
        self.f = torch.from_numpy(f).to(device)
        self.uvf = torch.from_numpy(uvf).to(device)
        self.device = device

    def get_data(self):
        """Triangles as tensors.
        returns tuple: (vertex_triangles, uv_triangles, uvmap)
        """
        return self.v[self.f], self.uv[self.uvf], self.uvmap

    def get_uvmap(self):
        """Return the uvmap image as PIL image """
        t = ToPILImage()
        return t(self.uvmap)
