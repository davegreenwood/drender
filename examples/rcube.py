import torch
import time
import numpy as np
from PIL import Image
from drender.utils import Rcube
from drender.render import inside_outside

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # set defaults
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = 256
    rcube = Rcube(size)
    tris, uvs = rcube.tris()

    xx, yy = torch.meshgrid([
        torch.arange(0, size).to(device),
        torch.arange(0, size).to(device)])
    pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    print(tris.device)
    print(pixels.device)

    t0 = time.time()
    barycentrics, mask = inside_outside(pixels, tris[:, :, :2])
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = np.zeros([size, size])
    img[mask.sum(0).reshape(size, size).cpu().numpy() > 0] = 1.0
    Image.fromarray(img * 255).convert("RGB").save("rcube.jpg")
