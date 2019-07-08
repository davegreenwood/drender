"""Utility functions to load data."""
from pkg_resources import resource_filename
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


RCUBE = resource_filename(__name__, "data/rcube.obj")
UVMAP = resource_filename(__name__, "data/util-mark6.png")


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

    def __init__(self, size: int = 256):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (v, f, uv, uvf), t = read_obj(RCUBE), ToTensor()
        self.v = torch.from_numpy(v).to(device)
        self.uv = torch.from_numpy(uv).to(device)
        self.map = t(Image.open(UVMAP).convert("RGB")).to(device)
        self.f = torch.from_numpy(f).to(device)
        self.uvf = torch.from_numpy(uvf).to(device)
        self.size = size
        self.device = device

    def tris(self):
        """Triangles scaled to fit a square image of (size * size).
        returns tuple: (vertex_triangles, uv_triangles)
        """
        _tris = self.v[self.f]
        _tris += torch.tensor([1.0, 1.0, 0.0], device=self.device)
        _tris *= self.size / 2.0
        return _tris, self.uv[self.uvf]

    def uvmap(self):
        """Return the uvmap image as PIL image """
        t = ToPILImage()
        return t(self.map)


if __name__ == "__main__":
    rcube = Rcube()
    tris, uvs = rcube.tris()
    print(tris.device)
    print(uvs.device)
    print(rcube.device)
    rcube.uvmap().save("uvmap.jpg")
