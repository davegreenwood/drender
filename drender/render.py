"""Render an object"""
import torch
from PIL import Image
from .utils import image2uvmap

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


def face_normals(tris):
    """calculate the face normal of the tris"""
    normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return normals / torch.norm(normals)


def backface_cull(tris):
    """
    Return a mask for the triangles that face forward. Tris are assumed to be
    in view space."""
    dtype, device = tris.dtype, tris.device
    normals = face_normals(tris)
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

    def __init__(self, size, faces, uv, uvfaces, uvmap,
                 dtype=DTYPE,
                 device=DEVICE):
        super(Render, self).__init__()
        self.uv = uv.to(dtype=dtype, device=device) * 2.0 - 1.0
        self.f = faces.to(dtype=torch.int64, device=device)
        self.uvf = uvfaces.to(dtype=torch.int64, device=device)
        self.pts = lookup_table(size, dtype=dtype, device=device)
        self.size = size
        self.uvmap = uvmap
        self.result = None
        self.zbuffer = None
        self.nmap = None
        self.dtype = dtype
        self.device = device

    def aabbmsk(self, tri):
        """Convert bounding box of 2d triangle to mask."""
        (xmin, ymin), (xmax, ymax) = tri.min(dim=0)[0], tri.max(dim=0)[0]
        msk_max = (self.pts[..., 0] <= xmax) & (self.pts[..., 1] <= ymax)
        msk_min = (self.pts[..., 0] >= xmin) & (self.pts[..., 1] >= ymin)
        return msk_min & msk_max

    def vertex_normals(self, vertices):
        """Calculate the vertex normals.
        Returns a normal for each vertex, the mean of each face normal."""
        fnorms = face_normals(vertices[self.f])
        vnorms = torch.zeros_like(vertices)
        vnorms.data[self.f[:, 0]] += fnorms
        vnorms.data[self.f[:, 1]] += fnorms
        vnorms.data[self.f[:, 2]] += fnorms
        return vnorms / torch.norm(vnorms)

    def normal_map(self):
        """
        Reshape and return the normal map as an image tensor.
        The normal values are scaled to fit in the 0-1 range.
        """
        nmap = self.nmap / torch.norm(self.nmap, dim=2, keepdim=True)
        nmap = nmap * 0.5 + 0.5
        return nmap.permute(2, 0, 1).contiguous().view(
            3, self.size, self.size)

    def z_map(self):
        """Return the z-buffer as an image"""
        mask = self.result[3] > 0
        zmap = self.zbuffer - self.zbuffer.min()
        zmap[mask] = ((zmap[mask] - zmap[mask].min()) /
                      (zmap[mask].max() - zmap[mask].min()))
        return zmap[None, ...]

    def cull(self, vertices):
        """
        Back face cull:
        return tensor N x 3 x 6 (3 points by (x, y, z, u/z, v/z, 1/z))
        """
        tris = vertices[self.f]
        z_inv = torch.ones([tris.shape[0], 3, 1]) / tris[:, :, 2, None]
        vnorms = self.vertex_normals(vertices)[self.f] * z_inv
        uvs = self.uv[self.uvf] * z_inv
        mask = backface_cull(tris)
        return torch.cat([tris, uvs, vnorms, z_inv], dim=2)[mask]

    def forward(self, vertices):
        self.render(vertices)
        return self.result

    def render(self, vertices):
        """do render """
        self.result = torch.zeros(
            [4, self.size, self.size], dtype=self.dtype, device=self.device)
        self.nmap = torch.zeros(
            [self.size, self.size, 3], dtype=self.dtype, device=self.device)
        self.zbuffer = torch.zeros(
            [self.size, self.size], dtype=self.dtype, device=self.device) + \
            vertices.min(0)[0][-1]
        tris = self.cull(vertices)
        for tri in tris:
            self.raster(tri)

    def raster(self, tri):
        """
        render a triangle tri.
        tri : tensor 3 x 9:
        (3 points by (x, y, z, u/z, v/z, nx/z, ny/z, nz/z, 1/z))
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

        # interpolated 9d pixels to consider for render
        pts9d = bary_interp(tri, w1, w2, w3)

        # keep points that are nearer than existing zbuffer
        ptsZ = 1 / pts9d[:, -1]
        zbf_msk = ptsZ >= self.zbuffer[bb_msk]
        if zbf_msk.sum() == 0:
            return None

        bb_msk[bb_msk] = zbf_msk
        # interpolated verts
        inVerts = pts9d[:, :-1] * ptsZ[..., None]
        rgb = torch.grid_sampler_2d(
            self.uvmap[None, ...],
            inVerts[None, None, :, 3:5], 0, 0)[0, :, 0, :]

        # fill buffers
        self.zbuffer[bb_msk] = ptsZ[zbf_msk]
        self.nmap[bb_msk, :] = inVerts[zbf_msk, 5:8]
        # allow alpha in uv map
        if self.uvmap.shape[0] == 4:
            self.result[:, bb_msk] = rgb[:, zbf_msk]
        else:
            self.result[:3, bb_msk] = rgb[:3, zbf_msk]
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

    def __init__(self, size, faces, uv, uvfaces, dtype=DTYPE, device=DEVICE):
        super(Reverse, self).__init__(
            size, faces, uv, uvfaces, uvmap=None, dtype=dtype, device=device)
        self.uvmap = None
        self.wUV = None
        self.uv_weights()

    def uv_weights(self):
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
            idx = torch.nonzero(pts_msk)
            self.wUV.append([idx, w1, w2, w3])

    def cull(self, vertices):
        """back face cull"""
        tris = vertices[self.f]
        z = - tris[:, :, 2, None]
        vnorms = self.vertex_normals(vertices)[self.f]
        mask = backface_cull(tris)
        idx = torch.nonzero(mask)
        return torch.cat([tris * z, vnorms * z, z], dim=2)[mask], \
            [self.wUV[i] for i in idx]

    def forward(self, vertices, image):
        self.render(vertices, image)
        return self.result

    def render(self, vertices, image):
        """
        If image is a PIL rgb image convert, or if tensor, pass directly.
        The resulting image must be channels * height * width.
        """
        if isinstance(image, Image.Image):
            self.uvmap = image2uvmap(image, self.device)
        else:
            self.uvmap = image
        self.nmap = torch.zeros(
            [self.size, self.size, 3], dtype=self.dtype, device=self.device)
        self.zbuffer = torch.zeros(
            [self.size, self.size], dtype=self.dtype, device=self.device) + \
            vertices.min(0)[0][-1]
        c, h, w = self.uvmap.shape[0] + 1, self.size, self.size
        self.result = torch.zeros(
            [c, h, w], dtype=self.dtype, device=self.device)
        tris, uvws = self.cull(vertices)
        for tri, uvw in zip(tris, uvws):
            self.raster(tri, uvw)

    def raster(self, tri, weights):
        """rasterize the triangle."""
        idx, w1, w2, w3 = weights
        pts = bary_interp(tri, w1, w2, w3)
        ptsZ = 1 / pts[:, -1:]

        # interpolated verts
        inVerts = pts[:, :-1] * ptsZ

        result = torch.grid_sampler_2d(
            self.uvmap[None, ...],
            inVerts[None, None, :, 0:2],
            0, 0)[0, :, 0, :]

        # fill buffers
        self.zbuffer[idx[:, 0], idx[:, 1]] = ptsZ[..., 0]
        self.nmap[idx[:, 0], idx[:, 1], :] = inVerts[:, 3:6]
        self.result[:-1, idx[:, 0], idx[:, 1]] = result
        self.result[-1, idx[:, 0], idx[:, 1]] = 1.0
