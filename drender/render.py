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
        aabb.to(DEVICE)
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
    return (b[0] - a[..., 0]) * (c[1] - a[..., 1]) - \
        (b[1] - a[..., 1]) * (c[0] - a[..., 0])


def barys(pAB, pBC, w, tri):
    """
    Sub triangle areas to barycentric, mult by tri vertices.
    """
    assert w > 0
    w1 = pAB / w
    w2 = pBC / w
    w3 = 1.0 - w1 - w2
    v1, v2, v3, = tri
    return w1[..., None] * v1 + w2[..., None] * v2 + w3[..., None] * v3


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

    def raster(self, tri):
        """
        triangle tri, iff the point is in the triangle.
        tri : tensor 3 x 2 (3 points by (x, y))
        """

        # mostly 2d - assuming the triangle is in view space
        tri2d = tri[:, :2]

        # index to everything - flip x, y to rotate image
        x, y = self.aabb_idx(tri2d)
        if x.shape[0] == 0:
            return None

        # signed area of tri - could do back face cull here
        w = area2d(tri2d[None, 0], tri2d[1], tri2d[2])
        if torch.isclose(w, torch.zeros_like(w)):
            return None

        # signed area of subtriangles, NB: any could be zero
        pts = self.pts[x, y, :]
        pAB = area2d(pts, tri2d[0], tri2d[1])
        pBC = area2d(pts, tri2d[1], tri2d[2])
        pCA = area2d(pts, tri2d[2], tri2d[0])

        # mask for pts that are in the triangle,
        pts_msk = inside_mask(pAB, pBC, pCA)
        if torch.allclose(pts_msk, torch.zeros_like(pts_msk)):
            return None

        # interpolated 3d pixels to consider for render
        pts3d = barys(pAB[pts_msk], pBC[pts_msk], w, tri)

        # keep points that are nearer than existing zbuffer
        zbuffer = self.zbuffer[x[pts_msk], y[pts_msk], :].view(-1)
        zpoints = pts3d[:, 2].view(-1)
        zbf_msk = torch.zeros_like(pts_msk)
        zbf_msk[pts_msk] = zpoints >= zbuffer

        if torch.allclose(zbf_msk, torch.zeros_like(zbf_msk)):
            return None

        # render points that are nearer and in triangle
        rnd_msk = pts_msk * zbf_msk

        # fill buffers
        self.zbuffer[x[rnd_msk], y[rnd_msk], 0] = pts3d[:, 2]
        self.result[x[rnd_msk], y[rnd_msk], :3] = pts3d[:, :3]
        self.result[x[rnd_msk], y[rnd_msk], 3] = 1.0
