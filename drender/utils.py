"""Utility functions to load data."""
from pkg_resources import resource_filename
import numpy as np
import torch
import cv2
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


RCUBE = resource_filename(__name__, "data/rcube.obj")
UVMAP = resource_filename(__name__, "data/util-mark6.png")
DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image2uvmap(image, device=DEVICE):
    """From a PIL image return a UV map tensor."""
    t = ToTensor()
    return t(image.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)).to(device)


def uvmap(fname=None, device=DEVICE):
    """
    Return the utility uvmap as a tensor. If fname is None uvmap will
    be default, or if fname is a valid image - use that."""
    fname = fname or UVMAP
    image = Image.open(fname)
    return image2uvmap(image, device)


def uvimg(fname=None):
    """
    Return the utility uvmap as a PIL Image. If fname is None uvmap will
    be default, or if fname is a valid image file - use that."""
    fname = fname or UVMAP
    return Image.open(fname)


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
    uv = np.array(vt, dtype=np.float32)
    faces = np.array(f, dtype=np.int64) - 1
    uvfaces = np.array(tf, dtype=np.int64) - 1
    return verts, faces, uv, uvfaces


def numpy_cube(scale=1.0):
    """Return numpy unit cube triangles. """
    _, f, uv, uvf = read_obj(RCUBE)
    v = scale * np.array([
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5]], dtype=np.float32)
    return v, f, uv, uvf


def assert_size(x, y):
    """Make sure x and y are equal powers of 2 """
    assert x == y
    assert x in [8, 16, 32, 64, 128, 256, 512, 1024]


class Rodrigues(torch.autograd.Function):
    """Wrap the cv2 rodrigues function. """

    @staticmethod
    def forward(ctx, r):
        dtype, device = r.dtype, r.device
        _r = r.detach().cpu().numpy()
        _rotm, _jacob = cv2.Rodrigues(_r)
        jacob = torch.tensor(_jacob).to(dtype=dtype, device=device)
        rotation_matrix = torch.tensor(_rotm).to(dtype=dtype, device=device)
        ctx.save_for_backward(jacob)
        return rotation_matrix

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output is 3x3, grad_input is shape 3, """
        jacob, = ctx.saved_tensors
        # TODO: Check Jacobian is correct - 1x9 * 9x3 or 9x3 * 3*1
        grad_input = jacob @ grad_output.contiguous().view(-1, 1)
        return grad_input.view(-1)


class Rcube:
    """test object - a cube with the vertices rotated."""

    def __init__(self, scale=1.0):
        (v, f, uv, uvf), totensor = read_obj(RCUBE), ToTensor()
        self.v = torch.from_numpy(v * scale).to(DEVICE)
        self.uv = torch.from_numpy(uv).to(DEVICE)
        self.f = torch.from_numpy(f).to(DEVICE)
        self.uvf = torch.from_numpy(uvf).to(DEVICE)
        self.uvmap = totensor(Image.open(UVMAP).transpose(
            Image.FLIP_TOP_BOTTOM).convert("RGB")).to(DEVICE)
        self.device = DEVICE

    def get_uvmap(self):
        """Return the uvmap image as PIL image.
        Undo the inversion of the map."""
        topil = ToPILImage()
        return topil(self.uvmap).transpose(
            Image.FLIP_TOP_BOTTOM).convert("RGB")


class Pcube(Rcube):
    """A cube with parameters to rotate."""

    def __init__(self, scale=1.0):
        super(Pcube, self).__init__()
        v, _, _, _ = numpy_cube(scale)
        self.v = torch.tensor(v, dtype=DTYPE, device=DEVICE)
        self.tris = self.v[self.f]
        self.rodrigues_fn = Rodrigues.apply

    def posed(self, r):
        """
        Triangles as tensors.
        params: r a rotation value
        returns tensor: 14*3*3
        """
        v = self.rodrigues_fn(r) @ self.v.t()
        return v.t()
