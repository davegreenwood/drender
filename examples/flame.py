import torch
import time
import numpy as np
from PIL import Image
from drender.utils import Rcube
from drender.render import inside_outside
from h5flame.model import Flame


# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = 256
    model = Flame()
    tris = torch.from_numpy(model.v[model.f].astype(np.float32))
    tris.to(device)
    # position
    tris += torch.tensor((1.0, 1.0, 0.0))
    tris *= size / 2
    xx, yy = torch.meshgrid([torch.arange(0, size), torch.arange(0, size)])
    pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    pixels.to(device)

    t0 = time.time()
    barycentrics, mask = inside_outside(pixels, tris[:, :, :2])
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = np.zeros([size, size])
    img[mask.sum(0).reshape(size, size).numpy() > 0] = 1.0
    Image.fromarray(img * 255).convert("RGB").save("flame.jpg")
