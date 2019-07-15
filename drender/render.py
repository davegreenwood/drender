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
        aabb = torch.clamp(aabb.detach(), min=-1, max=1)
        aabb += 1.0
        aabb /= 2
        aabb *= size

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


def barys(pCB, pCA, w, tri):
    """
    Sub triangle areas to barycentric, mult by tri vertices.
    """
    w1 = pCB / w
    w2 = pCA / w
    w3 = 1.0 - w1 - w2
    v1, v2, v3, = tri
    b = w1[..., None] * v1 + w2[..., None] * v2 + w3[..., None] * v3
    return b


def lookup_table(size, dtype=DTYPE, device=DEVICE):
    """return a square table (size x size), of 2d points (x, y) in -1, 1"""
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device),
         torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)])
    t = torch.stack([xx, yy], dim=-1)
    return t


class Render(torch.nn.Module):
    """Render trangles in view space (-1, 1,)"""

    def __init__(self, size):
        super(Render, self).__init__()
        device = DEVICE
        self.pts = None
        self.result = None
        self.zmin = 0.0
        self.size = size
        self.dtype = torch.float
        self.device = device
        self.aabb2idx = AABB.apply

    def forward(self, tris):
        super(Render, self).forward()
        return self.render(tris)

    def render(self, tris):
        """do render """
        zmin = tris.view(-1).view(-1, 3).min(0)[0][-1]
        self.result = torch.zeros(
            [self.size, self.size, 4], dtype=self.dtype, device=self.device)
        self.zbuffer = zmin * torch.ones(
            [self.size, self.size], dtype=self.dtype, device=self.device)
        self.pts = lookup_table(
            self.size, dtype=self.dtype, device=self.device)
        for t in tris:
            self.raster(t)

    def raster(self, tri):
        """
        triangle tri, iff the point is in the triangle.
        tri : tensor 3 x 2 (3 points by (x, y))
        """

        # mostly 2d - assuming the triangle is in view space
        tri2d = tri[:, :2]

        # index to everything
        aabb = torch.cat([tri2d.min(dim=0)[0], tri2d.max(dim=0)[0]])
        xmin, ymin, xmax, ymax = self.aabb2idx(aabb, self.size)

        # signed area of tri - free back face cull here
        w = area2d(tri2d[0], tri2d[1], tri2d[2])
        if w < 1e-9:
            return None

        # signed area of subtriangles, NB: any could be zero
        pts = self.pts[xmin:xmax, ymin:ymax, :]
        pAB = area2d(tri2d[1], pts, tri2d[0])
        pCB = area2d(tri2d[2], pts, tri2d[1])
        pCA = area2d(tri2d[0], pts, tri2d[2])

        # mask for pts that are in the triangle,
        pts_msk = torch.clamp(pAB, min=0) * \
            torch.clamp(pCB, min=0) * torch.clamp(pCA, min=0) > 0
        if pts_msk.sum() == 0:
            return None

        # interpolated 3d pixels to consider for render
        pts3d = barys(pCB, pCA, w, tri)

        # keep points that are nearer than existing zbuffer
        zbf_msk = pts3d[:, :, 2] >= self.zbuffer[xmin:xmax, ymin:ymax]
        if zbf_msk.sum() == 0:
            return None

        # render points that are nearer AND in triangle
        rmsk = pts_msk * zbf_msk

        # fill buffers
        self.zbuffer[xmin:xmax, ymin:ymax][rmsk] = pts3d[:, :, 2][rmsk]
        self.result[xmin:xmax, ymin:ymax, :3][rmsk] = pts3d[:, :, :3][rmsk]
        self.result[xmin:xmax, ymin:ymax, 3][rmsk] = 1.0
