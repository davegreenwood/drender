import torch
import time
import numpy as np
from PIL import Image
from drender.render import Render
from h5flame.model import Flame

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = 256
    model = Flame()
    tris = torch.tensor(model.v[model.f], dtype=DTYPE, device=DEVICE)
    tris *= 4
    print(tris.dtype, tris.device)
    rnd = Render(size)

    t0 = time.time()
    rnd.render(tris)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = rnd.result.cpu().numpy()[:, :, :3].squeeze()
    img -= img.min()
    img /= img.max()
    img *= 255
    Image.fromarray(img.astype(np.uint8)).convert("RGB").save("flame.jpg")
