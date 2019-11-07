"""
Using many of the functions for rendering,
here we project points from one 2d view to UV space.
For each point: test against each triangle.

No spatial partitioning as we assume relatively small numbers of points.

"""

import torch
from torchvision.transforms import ToTensor
from .render import subarea2d, area2d, backface_cull, points_mask, barys, \
        bary_interp


DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def aabb(tri):
    """get bounding box of 2d triangle."""
    (xmin, ymin), (xmax, ymax) = tri.min(dim=0)[0], tri.max(dim=0)[0]
    return (xmin, ymin, xmax, ymax)


def test_point(p, tri):
    """Test a point against an nd triangle. If point is in tri, return
    interpolated point, else None."""
    tri2d = tri[:, :2]
    bb = aabb(tri2d)
    if p[0] < bb[0] or p[0] > bb[2]:
        return None
    if p[1] < bb[1] or p[1] > bb[3]:
        return None
    w = area2d(tri2d[0], tri2d[1], tri2d[2])
    if w < 1e-9:
        return None
    pAB, pCB, pCA = subarea2d(p, tri2d)
    if not points_mask(pAB, pCB, pCA):
        return None
    w1, w2, w3 = barys(pCB, pCA, w)
    return bary_interp(tri, w1, w2, w3)


class Project(torch.nn.Module):
    """Project 2d points in a [0, size] range to [0, 1] UV space
    size: int, the size of the image space, eg: 512
    """

    def __init__(self, size, faces, uv, uvfaces,
                 dtype=DTYPE,
                 device=DEVICE):
        super(Project, self).__init__()
        self.uv = uv.to(dtype=dtype, device=device)
        self.f = faces.to(dtype=torch.int64, device=device)
        self.uvf = uvfaces.to(dtype=torch.int64, device=device)
        self.size = size
        self.dtype = dtype
        self.device = device
        self.result = []

    def pt_size(self, points):
        """
        Scale points from image space [0, size-1] to projected space [-1, 1]
        """
        p = torch.stack([points[:, 0], self.size - points[:, 1]], dim=1)
        return p / (self.size - 1) * 2 - 1

    def cull(self, vertices):
        tris = vertices[self.f]
        mask = backface_cull(tris)
        z_inv = 1.0 / tris[:, :, 2, None]
        uvs = self.uv[self.uvf] * z_inv
        return torch.cat([tris, uvs, z_inv], dim=2)[mask]

    def project(self, vertices, points):
        tris = self.cull(vertices)
        self.result = []
        for p in self.pt_size(points):
            uvpt = torch.tensor([0.0, 0.0, -float("Inf")], device=self.device)
            for tri in tris:
                ptnd = test_point(p, tri)
                if ptnd is None:
                    continue
                ptz = 1 / ptnd[None, -1]
                uv = torch.cat([ptnd[3:5] * ptz, ptz], dim=0)
                if uv[-1] > uvpt[-1]:
                    uvpt = uv
            self.result.append(uvpt)

    def forward(self, vertices, points):
        self.project(vertices, points)
        return torch.stack(self.result, dim=0)
