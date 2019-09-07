import time
import torch
from torchvision.transforms import ToPILImage
from drender.utils import uvmap
from drender.render import Render
from dcamera.camera import Pinhole
from h5flame.model import Flame

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

print(DEVICE)

topil = ToPILImage()
size = 128
model = Flame()

v = torch.tensor(model.v, dtype=DTYPE, device=DEVICE)
uv = torch.tensor(model.uv, dtype=DTYPE, device=DEVICE)
f = torch.tensor(model.f, dtype=torch.int64, device=DEVICE)
uvf = torch.tensor(model.uvf, dtype=torch.int64, device=DEVICE)

cam = Pinhole(f=5, t=[0, -0.03, 1])
vp = cam.project(v)

print(vp.min(0)[0])
print(vp.max(0)[0])

rnd = Render(size, f, uv, uvf, uvmap())

t0 = time.time()
result = rnd.forward(vp)
t1 = time.time()
print(f"time: {t1-t0:0.2f}")

# image out
img = topil(result.cpu())
img.convert("RGB").save("flame.jpg")

nmap = topil(rnd.normal_map())
nmap.convert("RGB").save("flamenmap.jpg")

nmap = topil(rnd.z_map())
nmap.convert("RGB").save("flame_zmap.jpg")
