# %%
import time
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from dcamera.camera import Pinhole
from drender.utils import Rcube
from drender.render import Render, Reverse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

print(DEVICE)

# set defaults
uv_size, target_size = 500, 500
topil, toten = ToPILImage(), ToTensor()
rcube = Rcube()

# render
rnd = Render(target_size, rcube.f, rcube.uv, rcube.uvf, rcube.uvmap)
rev = Reverse(uv_size, rcube.f, rcube.uv, rcube.uvf)

cam = Pinhole(f=1, t=[0, 0, 2])
vp = cam.project(rcube.v)

# %%

target = topil(rnd(vp))

t0 = time.time()
result = rev(vp, target)
t1 = time.time()
print(f"time: {t1-t0:0.2f}")

# image out
target.convert("RGB").save("target_cube.jpg")

img = topil(result.cpu()[:3, ...])
img.convert("RGB").save("reverse_cube.jpg")

img = topil(rev.normal_map().cpu())
img.convert("RGB").save("reverse_norm_cube.jpg")

img = topil(rev.z_map().cpu())
img.convert("RGB").save("reverse_z_cube.jpg")
