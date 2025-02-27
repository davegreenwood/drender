import time
import torch
from torchvision.transforms import ToPILImage
from drender.utils import Rcube
from drender.render import Render
from PIL import Image
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

print(DEVICE)

# set defaults
size = 1024
topil = ToPILImage()
rcube = Rcube()
rnd = Render(size, rcube.f, rcube.uv, rcube.uvf, rcube.uvmap)

v = rcube.v
v[:, 2] -= 2
v[:, :2] /= -v[:, 2, None]

t0 = time.time()
result = rnd.forward(v)
t1 = time.time()
print(f"time: {t1-t0:0.2f}")

# image out
img = topil(result.cpu())
img.convert("RGB").save("rcube.jpg")

nmap = topil(rnd.normal_map())
nmap.convert("RGB").save("rnmap.jpg")

zmap = topil(rnd.z_map())
zmap.convert("RGB").save("rzmap.jpg")
