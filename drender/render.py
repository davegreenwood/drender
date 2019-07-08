"""Render an object"""
import torch


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
