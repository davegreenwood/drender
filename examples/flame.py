import time
import torch
from torchvision.transforms import ToPILImage
from drender.utils import Rcube
from drender.render import Render
from h5flame.model import Flame

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    topil = ToPILImage()
    size = 1024
    model = Flame()
    rcube = Rcube()
    _, _, uvmap = rcube.get_data()
    tris = torch.tensor(model.v[model.f], dtype=DTYPE, device=DEVICE)
    uvs = torch.tensor(model.uv[model.uvf], dtype=DTYPE, device=DEVICE)
    tris *= 4

    print(tris.dtype, tris.device)
    rnd = Render(size, uvs, uvmap)

    t0 = time.time()
    result = rnd.forward(tris)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = topil(result.cpu())
    img.convert("RGB").save("flame.jpg")
