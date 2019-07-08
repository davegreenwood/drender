"""Render an object"""
import torch
import time
import numpy as np
from PIL import Image
from drender.utils import Rcube
import matplotlib.pyplot as plt


def inside_outside(pixels, triangles):
    """
    pixels: ndarray [npixels, 2(x, y)]
    triangles: ndarray [ntris, 3, 2(x, y)]
    """
    pixels_f = pixels.float()

    edge_0 = triangles[:, 1] - triangles[:, 0]
    edge_1 = triangles[:, 2] - triangles[:, 1]
    edge_2 = triangles[:, 0] - triangles[:, 2]

    norm = edge_0[:, 0] * edge_2[:, 1] - edge_0[:, 1] * edge_2[:, 0] + 1e-8
    norm = torch.unsqueeze(norm, dim=-1)

    # vectors from vertices to the point
    p0 = triangles[:, None, 0] - pixels_f
    p1 = triangles[:, None, 1] - pixels_f
    p2 = triangles[:, None, 2] - pixels_f

    t1 = (edge_0[:, 0, None] * p0[:, :, 1] -
          edge_0[:, 1, None] * p0[:, :, 0]) * norm
    t2 = (edge_1[:, 0, None] * p1[:, :, 1] -
          edge_1[:, 1, None] * p1[:, :, 0]) * norm
    t3 = (edge_2[:, 0, None] * p2[:, :, 1] -
          edge_2[:, 1, None] * p2[:, :, 0]) * norm

    bx = t2 / t1
    by = t3 / t1
    bz = 1.0 - bx - by
    barycentrics = torch.stack([bx, by, bz], dim=-1)

    mask = torch.clamp(t1, min=0.) * torch.clamp(t2, min=0.) \
        * torch.clamp(t3, min=0.)

    return barycentrics, mask

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    size = 256
    rcube = Rcube(size)
    tris, uvs = rcube.tris()
    xx, yy = torch.meshgrid([torch.arange(0, size), torch.arange(0, size)])
    pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    t0 = time.time()
    barycentrics, mask = inside_outside(pixels, tris[:, :, :2])
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = np.zeros([size, size])
    img[mask.sum(0).reshape(size, size).numpy() > 0] = 1.0
    Image.fromarray(img * 255).convert("RGB").save("rcube.jpg")


# class Raster(object):
#     """"""

#     def __init__(self, size: int = 256):
#         super(Raster, self).__init__()

