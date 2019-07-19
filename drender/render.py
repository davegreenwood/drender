"""Render an object"""
import torch

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def area2d(a, b, c):
    """
    Vectorised area of 2d parallelogram (divide by 2 for triangle)
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    NB: colinear points result in zero area.
    """
    w = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
        (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    return w


def barys(pCB, pCA, w):
    """
    Sub triangle areas to barycentric.
    """
    w1 = pCB / w
    w2 = pCA / w
    w3 = 1.0 - w1 - w2
    return w1, w2, w3


def bary_interp(tri, w1, w2, w3):
    """bary centric weights to points in triangle - 2d or 3d"""
    v1, v2, v3, = tri
    return w1[..., None] * v1 + w2[..., None] * v2 + w3[..., None] * v3


def lookup_table(size, dtype=DTYPE, device=DEVICE):
    """return a square table (size x size), of 2d points (x, y) in -1, 1"""
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device),
         torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)])
    t = torch.stack([xx, yy], dim=-1)
    return torch.rot90(t, 1)


def backface_cull(tris):
    """
    Return a mask for the triangles that face forward. Tris are assumed to be
    in view space."""
    dtype, device = tris.dtype, tris.device
    normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    m = torch.tensor([0, 0, 1.0], dtype=dtype, device=device) @ normals
    return m > 0


class Render(torch.nn.Module):
    """Render trangles in view space (-1, 1,)"""

    def __init__(self, size, uvs, uvmap):
        super(Render, self).__init__()
        self.uvs = uvs.to(dtype=DTYPE, device=DEVICE) * 2.0 - 1.0
        self.size = size
        self.uvmap = uvmap
        self.pts = None
        self.result = None
        self.zbuffer = None
        self.dtype = DTYPE
        self.device = DEVICE

    def aabbmsk(self, tri):
        """Convert bounding box of 2d triangle to mask."""
        (xmin, ymin), (xmax, ymax) = tri.min(dim=0)[0], tri.max(dim=0)[0]
        msk_max = self.pts[..., 0] <= xmax
        msk_max *= self.pts[..., 1] <= ymax
        msk_min = self.pts[..., 0] >= xmin
        msk_min *= self.pts[..., 1] >= ymin
        return msk_min * msk_max

    def forward(self, tris):
        self.render(tris)
        return self.result

    def render(self, tris):
        """do render """
        self.result = torch.zeros(
            [4, self.size, self.size], dtype=DTYPE, device=DEVICE)
        self.zbuffer = torch.zeros(
            [self.size, self.size], dtype=DTYPE, device=DEVICE) + \
            tris.view(-1).view(-1, 3).min(0)[0][-1]
        self.pts = lookup_table(
            self.size, dtype=self.dtype, device=self.device)
        for tri, uv in zip(tris, self.uvs):
            self.raster(tri, uv)

    def raster(self, tri, uv):
        """
        triangle tri, iff the point is in the triangle.
        tri : tensor 3 x 2 (3 points by (x, y))
        """

        # mostly 2d - assuming the triangle is in view space
        tri2d = tri[:, :2]

        # signed area of tri - free back face cull here
        w = area2d(tri2d[0], tri2d[1], tri2d[2])
        if w < 1e-9:
            return None

        # index to everything
        bb_msk = self.aabbmsk(tri2d)

        # signed area of subtriangles, NB: any could be zero
        pts = self.pts[bb_msk]
        pAB = area2d(tri2d[1], pts, tri2d[0])
        pCB = area2d(tri2d[2], pts, tri2d[1])
        pCA = area2d(tri2d[0], pts, tri2d[2])

        # mask for pts that are in the triangle,
        pts_msk = torch.clamp(pAB, min=0) * \
            torch.clamp(pCB, min=0) * torch.clamp(pCA, min=0) > 0
        if pts_msk.sum() == 0:
            return None
        bb_msk[bb_msk] = pts_msk

        # barycentric coordinates
        w1, w2, w3 = barys(pCB, pCA, w)

        # interpolated 3d pixels to consider for render
        pts3d = bary_interp(tri, w1, w2, w3)

        # keep points that are nearer than existing zbuffer
        zbf_msk = pts3d[pts_msk, 2] >= self.zbuffer[bb_msk]
        if zbf_msk.sum() == 0:
            return None

        bb_msk[bb_msk] = zbf_msk
        pts_msk[pts_msk] = zbf_msk

        # interpolated uvs for rgb
        ptsUV = bary_interp(uv, w1, w2, w3)

        rgb = torch.grid_sampler_2d(
            self.uvmap[None, ...], ptsUV[None, None, ...], 0, 0).squeeze()

        # fill buffers
        self.zbuffer[bb_msk] = pts3d[pts_msk, 2]
        self.result[:3, bb_msk] = rgb[..., pts_msk]
        self.result[3, bb_msk] = 1.0
