"""Render an object"""
import torch

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AABB(torch.autograd.Function):
    """
    Return indexing values for a bounding box to be used:
    xmin, ymin, xmax, ymax = aabb
    slice = array2d[xmin:xmax, ymin:ymax]
    """

    @staticmethod
    def forward(ctx, aabb, size):
        """
        aabb is a tensor we want to clamp to -1, 1 then scale to screen space.
        """
        aabb += 1.0
        aabb /= 2
        aabb *= size
        aabb += torch.tensor((0., 0., 1., 1.))
        aabb = torch.clamp(aabb.detach(), min=0, max=size)

        ctx.save_for_backward(aabb)
        ctx.save_for_backward(size)
        aabb = aabb.long()
        aabb.to(device=DEVICE)
        return aabb

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward pass.
        """
        aabb, size = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input[aabb < 0] = 0
        grad_input[aabb > size] = size
        grad_input /= size
        grad_input *= 2
        return grad_input, None


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


class Render(torch.nn.Module):
    """Render trangles in view space (-1, 1,)"""

    def __init__(self, size, uvs, uvmap):
        super(Render, self).__init__()
        device = DEVICE
        dtype = DTYPE
        self.pts = None
        self.result = None
        self.zmin = 0.0
        self.size = size
        self.dtype = dtype
        self.device = device
        self.aabbfn = AABB.apply
        self.uvs = uvs.to(dtype=dtype, device=device) * 2.0 - 1.0
        self.uvmap = uvmap

    def aabb2idx(self, tri):
        """Convert bounding box of 2d triangle to indexing values."""
        aabb = torch.cat([tri.min(dim=0)[0], tri.max(dim=0)[0]])
        xmin, ymin, xmax, ymax = self.aabbfn(aabb, self.size)
        return self.size - ymax, self.size - ymin, xmin, xmax

    def forward(self, tris):
        self.render(tris)
        return self.result

    def render(self, tris):
        """do render """
        zmin = tris.view(-1).view(-1, 3).min(0)[0][-1]
        self.result = torch.zeros(
            [4, self.size, self.size], dtype=self.dtype, device=self.device)
        self.zbuffer = zmin * torch.ones(
            [self.size, self.size], dtype=self.dtype, device=self.device)
        self.pts = lookup_table(
            self.size, dtype=self.dtype, device=self.device)

        for t, uv in zip(tris, self.uvs):
            self.raster(t, uv)

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
        r1, r2, c1, c2 = self.aabb2idx(tri2d)

        # signed area of subtriangles, NB: any could be zero
        pts = self.pts[r1:r2, c1:c2, :]
        pAB = area2d(tri2d[1], pts, tri2d[0])
        pCB = area2d(tri2d[2], pts, tri2d[1])
        pCA = area2d(tri2d[0], pts, tri2d[2])

        # mask for pts that are in the triangle,
        pts_msk = torch.clamp(pAB, min=0) * \
            torch.clamp(pCB, min=0) * torch.clamp(pCA, min=0) > 0
        if pts_msk.sum() == 0:
            return None

        # barycentric coordinates
        w1, w2, w3 = barys(pCB, pCA, w)

        # interpolated 3d pixels to consider for render
        pts3d = bary_interp(tri, w1, w2, w3)

        # keep points that are nearer than existing zbuffer
        zbf_msk = pts3d[:, :, 2] >= self.zbuffer[r1:r2, c1:c2]
        if zbf_msk.sum() == 0:
            return None

        # interpolated uvs for rgb
        ptsUV = bary_interp(uv, w1, w2, w3)

        rgb = torch.grid_sampler_2d(
            self.uvmap[None, ...], ptsUV[None, ...], 0, 0)[0, :3, ...]

        # render points that are nearer AND in triangle
        rmsk = pts_msk * zbf_msk

        # fill buffers
        self.zbuffer[r1:r2, c1:c2][rmsk] = pts3d[:, :, 2][rmsk]
        self.result[:3, r1:r2, c1:c2][..., rmsk] = rgb[..., rmsk]
        self.result[3, r1:r2, c1:c2][rmsk] = 1.0
