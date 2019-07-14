"""Render an object"""
import torch
import numpy as np

DTYPE = torch.float
DEVICE = "cpu"


def backface_cull(tris):
    """
    Return only the triangles that face forward. Tris are assumed to be
    in view space."""
    normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    m = torch.tensor([0, 0, 1.0], dtype=DTYPE, device=DEVICE) @ normals
    return tris[m > 0]


def aabb(tri):
    """
    Tri is tensor in screen space - screen space is 2d in -1, 1.
    returns axis aligned box clamped to screen space.
    tri : tensor 3 x 2 (3 points by (x, y))
    TODO: What if min and max are equal ie. the tri is off screen...
    """
    _min, _max = tri.min(dim=0)[0], tri.max(dim=0)[0]
    return torch.cat(
        [torch.clamp(_min, min=-1, max=1),
         torch.clamp(_max, min=-1, max=1)])


def box2pts(bb, size):
    """
    return (x, y) coordinates within a bounding box.
    NOTE: This is non differentiable - does it matter?
    """
    bb += 1.0
    bb /= 2
    bb *= size
    x1, y1, x2, y2 = bb.long()
    xx, yy = torch.meshgrid(
        [torch.arange(x1, x2), torch.arange(y1, y2)])
    return torch.stack([xx, yy], dim=-1).reshape(2, -1)


def area2d(a, b, c):
    """
    Vectorised area of 2d parallelogram (divide by 2 for triangle)
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    NB: colinear points result in zero area.
    """
    return (b[0] - a[:, 0]) * (c[1] - a[:, 1]) - \
        (b[1] - a[:, 1]) * (c[0] - a[:, 0])


def barys(pAB, pBC, pCA):
    """sub triangle areas to barycentric
    TODO: potential division by zero here.
    """
    bx = pBC / pAB
    by = pCA / pAB
    bz = 1.0 - bx - by
    return torch.cat([bx[..., None], by[..., None], bz[..., None]], dim=-1)


def inside_mask(w1, w2, w3):
    """
    From weighted triangles, decide if all are greater than zero.
    returns a mask: [1, 1, 0, ... 1, 0]
    NOTE: is this operation out of graph as non - differentiable?
    """
    stacked = torch.stack([w1, w2, w3])
    clamped = torch.clamp(stacked, min=0)
    product = torch.prod(clamped, dim=0)
    m = torch.zeros_like(product, dtype=torch.uint8, device=DEVICE)
    m[product > 0] = 1
    return m


def assert_size(x, y):
    """Make sure x and y are equal powers of 2 """
    assert x == y
    assert x in [8, 16, 32, 64, 128, 256, 512, 1024]


def lookup_table(size, dtype=DTYPE, device=DEVICE):
    """return a square table (size x size), of 2d points (x, y) in -1, 1"""
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device),
         torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)])
    return torch.stack([xx, yy], dim=-1)


class Render(torch.nn.Module):
    """Render trangles in view space (-1, 1,)"""

    def __init__(self, size):
        super(Render, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float
        self.pts = None
        self.result = None
        self.zmin = 0.0
        self.size = size
        self.dtype = dtype
        self.device = device

    def forward(self, tris):
        super(Render, self).forward()
        return self.render(tris)

    def render(self, tris):
        """do render """
        zmin = tris.view(-1).view(-1, 3).min(0)[0][-1]
        self.result = torch.zeros(
            [self.size, self.size, 4], dtype=self.dtype, device=self.device)
        self.zbuffer = zmin * torch.ones(
            [self.size, self.size, 1], dtype=self.dtype, device=self.device)
        self.pts = lookup_table(
            self.size, dtype=self.dtype, device=self.device)
        for t in tris:
            self.raster(t)
        self.zbuffer -= zmin
        self.zbuffer /= self.zbuffer.view(-1).max(0)[0]

    def aabb_idx(self, tri):
        """
        Return the (x, y) indices of points within the triangle AABB.
        If, the box is empty the coords will be None, None
        """
        bb = aabb(tri)
        x, y = box2pts(bb, self.size)
        return x, y

    def raster(self, tri):
        """
        triangle tri, iff the point is in the triangle.
        tri : tensor 3 x 2 (3 points by (x, y))
        """

        # mostly 2d - assuming the triangle is in view space
        tri2d = tri[:, :2]

        # index to everything
        x, y = self.aabb_idx(tri2d)
        if x.shape[0] == 0:
            return None

        # area of tri
        w = area2d(tri2d[None, 0], tri2d[1], tri2d[2])
        if torch.isclose(w, torch.zeros_like(w)):
            return None

        # signed weighted area of subtriangles, NB: any could be zero
        pts = self.pts[x, y, :]
        pAB = area2d(pts, tri2d[0], tri2d[1]) * w
        pBC = area2d(pts, tri2d[1], tri2d[2]) * w
        pCA = area2d(pts, tri2d[2], tri2d[0]) * w

        # mask for pts that are in the triangle,
        pts_msk = inside_mask(pAB, pBC, pCA)
        if torch.allclose(pts_msk, torch.zeros_like(pts_msk)):
            return None

        # interpolated 3d points to consider for render
        pts3d = barys(pAB, pBC, pCA) @ tri

        # keep points that are nearer than existing zbuffer
        zbuffer = self.zbuffer[x, y, :].view(-1)
        zbf_msk = pts3d[:, 2].view(-1) >= zbuffer
        if torch.allclose(zbf_msk, torch.zeros_like(zbf_msk)):
            return None

        # render points nearer and in triangle
        rnd_msk = pts_msk * zbf_msk

        self.zbuffer[x[rnd_msk], y[rnd_msk], :] = pts3d[rnd_msk, 2, None]
        self.result[x[rnd_msk], y[rnd_msk], :3] = pts3d[rnd_msk, :3]
        self.result[x[rnd_msk], y[rnd_msk], 3] = 1.0
        return None
