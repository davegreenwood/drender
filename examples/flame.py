import time
import torch
from torchvision.transforms import ToPILImage
from drender.utils import uvmap
from drender.render import Render
from h5flame.model import Flame

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    topil = ToPILImage()
    size = 64
    model = Flame()

    v = torch.tensor(model.v * 4, dtype=DTYPE, device=DEVICE)
    uv = torch.tensor(model.uv, dtype=DTYPE, device=DEVICE)
    f = torch.tensor(model.f, dtype=torch.int64, device=DEVICE)
    uvf = torch.tensor(model.uvf, dtype=torch.int64, device=DEVICE)

    rnd = Render(size, f, uv, uvf, uvmap())

    t0 = time.time()
    result = rnd.forward(v)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = topil(result.cpu())
    img.convert("RGB").save("flame.jpg")
