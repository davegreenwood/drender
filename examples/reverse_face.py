import time
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from dcamera.camera import Pinhole
from drender.utils import uvmap
from drender.render import Reverse
from flame.numpy.model import Flame

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

print(DEVICE)

topil = ToPILImage()
uv_size = 512
model = Flame()

v = torch.tensor(model.v, dtype=DTYPE, device=DEVICE)
uv = torch.tensor(model.uv, dtype=DTYPE, device=DEVICE)
f = torch.tensor(model.f, dtype=torch.int64, device=DEVICE)
uvf = torch.tensor(model.uvf, dtype=torch.int64, device=DEVICE)
image = Image.open("examples/face.png")

cam = Pinhole(f=5, t=[0, -0.03, 1])
vp = cam.project(v)

t0 = time.time()
rev = Reverse(uv_size, f, uv, uvf, device=DEVICE)
t1 = time.time()
print(f"time to build point table: {t1-t0:0.2f}")

t0 = time.time()
result = rev(vp, image)
t1 = time.time()
print(f"time to render: {t1-t0:0.2f}")

# image out
img = topil(result.cpu()[:3, ...])
img.convert("RGB").save("reverse_flame.jpg")

img = topil(rev.normal_map().cpu())
img.convert("RGB").save("reverse_norm_flame.jpg")

img = topil(rev.z_map().cpu())
img.convert("RGB").save("reverse_z_flame.jpg")
