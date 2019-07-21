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


def subarea2d(pts, tri2d):
    """The three sub areas of a point tested against a triangle in 2d."""
    pAB = area2d(tri2d[1], pts, tri2d[0])
    pCB = area2d(tri2d[2], pts, tri2d[1])
    pCA = area2d(tri2d[0], pts, tri2d[2])
    return pAB, pCB, pCA


def points_mask(pAB, pCB, pCA):
    """Return a mask: 1 if all areas > 0 else 0 """
    return torch.clamp(pAB, min=0) * \
        torch.clamp(pCB, min=0) * \
        torch.clamp(pCA, min=0) > 0


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
    m = normals @ torch.tensor([0, 0, 1.0], dtype=dtype, device=device)
    return m > 0


class Render(torch.nn.Module):
    """
    Render trangles in view space (-1, 1,).
    args:
        size: (int) the square size of the resulting image
        faces: (tensor k * 3, int64) the indices to vertices to form triangles.
        uv: (tensor m * 2) the uv coordinates in 0.0 - 1.0
        uvfaces: (tensor n * 3, int64) the indices to uv to form triangles.
        uvmap: (tensor 3 * x * y) upside down rgb image as tensor.
        NB: faces, uv, uvfaces and uvmap are static data. Only vertex positions
        change during rendering, which are passed to the forward function.
    """

    def __init__(self, size, faces, uv, uvfaces, uvmap):
        super(Render, self).__init__()
        self.uv = uv.to(dtype=DTYPE, device=DEVICE) * 2.0 - 1.0
        self.f = faces.to(dtype=torch.int64, device=DEVICE)
        self.uvf = uvfaces.to(dtype=torch.int64, device=DEVICE)
        self.pts = lookup_table(size, dtype=DTYPE, device=DEVICE)
        self.size = size
        self.uvmap = uvmap
        self.result = None
        self.zbuffer = None
        self.dtype = DTYPE
        self.device = DEVICE

    def aabbmsk(self, tri):
        """Convert bounding box of 2d triangle to mask."""
        (xmin, ymin), (xmax, ymax) = tri.min(dim=0)[0], tri.max(dim=0)[0]
        msk_max = (self.pts[..., 0] <= xmax) * (self.pts[..., 1] <= ymax)
        msk_min = (self.pts[..., 0] >= xmin) * (self.pts[..., 1] >= ymin)
        return msk_min * msk_max

    def cull(self, vertices):
        """back face cull - return tuple of vert triangles and uv triangles"""
        mask = backface_cull(vertices[self.f])
        return vertices[self.f][mask], self.uv[self.uvf][mask]

    def forward(self, vertices):
        self.render(vertices)
        return self.result

    def render(self, vertices):
        """do render """
        self.result = torch.zeros(
            [4, self.size, self.size], dtype=DTYPE, device=DEVICE)
        self.zbuffer = torch.zeros(
            [self.size, self.size], dtype=DTYPE, device=DEVICE) + \
            vertices.min(0)[0][-1]
        tris, uvs = self.cull(vertices)
        for tri, uv in zip(tris, uvs):
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
        pAB, pCB, pCA = subarea2d(self.pts[bb_msk], tri2d)

        # mask for pts that are in the triangle,
        pts_msk = points_mask(pAB, pCB, pCA)
        if pts_msk.sum() == 0:
            return None
        bb_msk[bb_msk] = pts_msk

        # barycentric coordinates
        w1, w2, w3 = barys(pCB[pts_msk], pCA[pts_msk], w)

        # interpolated 3d pixels to consider for render
        pts3d = bary_interp(tri, w1, w2, w3)

        # keep points that are nearer than existing zbuffer
        zbf_msk = pts3d[:, 2] >= self.zbuffer[bb_msk]
        if zbf_msk.sum() == 0:
            return None

        bb_msk[bb_msk] = zbf_msk

        # interpolated uvs for rgb
        ptsUV = bary_interp(uv, w1[zbf_msk], w2[zbf_msk], w3[zbf_msk])

        rgb = torch.grid_sampler_2d(
            self.uvmap[None, ...], ptsUV[None, None, ...], 0, 0)[0, :, 0, :]

        # fill buffers
        self.zbuffer[bb_msk] = pts3d[zbf_msk, 2]
        self.result[:3, bb_msk] = rgb
        self.result[3, bb_msk] = 1.0

# -----------------------------------------------------------------------------
# Reverse render - project the view to UV space
# -----------------------------------------------------------------------------


class Reverse(Render):
    """
    Project the vertices back to UV space.
    Because the output projection is known, we can precompute the
    rasterisation once, then each call to forward is much faster.
    """

    def __init__(self, size, faces, uv, uvfaces, uvmap):
        super(Reverse, self).__init__(size, faces, uv, uvfaces, uvmap)
        self.uvmap = None
        self.wUV = None
        self.get_uvpts()

    def get_uvpts(self):
        """
        Similar to raster in super.
        TODO: refactoring
        """
        self.wUV = []
        for uvtri in self.uv[self.uvf]:
            # no negative areas in UV space - no zbuffer
            w = area2d(uvtri[0], uvtri[1], uvtri[2])
            pAB, pCB, pCA = subarea2d(self.pts, uvtri)
            pts_msk = points_mask(pAB, pCB, pCA)
            w1, w2, w3 = barys(pCB[pts_msk], pCA[pts_msk], w)
            idx = torch.nonzero(pts_mask)
            self.wUV.append([idx, w1, w2, w3])

    def cull(self, vertices):
        """back face cull"""
        mask = backface_cull(vertices[self.f])
        return vertices[self.f][mask], self.wUV[mask]

    def forward(self, vertices, image):
        self.render(vertices, image)
        return self.result

    def raster(self, tri2d, weights):
        idx, w1, w2, w3 = weights
        ptsUV = bary_interp(tri2d, w1, w2, w3)
        rgb = torch.grid_sampler_2d(
            self.uvmap[None, ...], ptsUV[None, None, ...], 0, 0)[0, :, 0, :]
        # fill buffers
        self.result[:3, idx] = rgb
        self.result[3, idx] = 1.0

    def render(self, vertices, image):
        self.uvmap = image
        self.result = torch.zeros(
            [4, self.size, self.size], dtype=DTYPE, device=DEVICE)
        tris, uvws = backface_cull(vertices[self.f])
        for tri, uvw in zip(tris, uvws):
            self.raster(tri, uvw)
