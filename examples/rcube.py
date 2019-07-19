import time
from torchvision.transforms import ToPILImage
from drender.utils import Rcube
from drender.render import Render

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # set defaults
    size = 1024
    topil = ToPILImage()
    rcube = Rcube()
    rnd = Render(size, rcube.f, rcube.uv, rcube.uvf, rcube.uvmap)

    t0 = time.time()
    result = rnd.forward(rcube.v)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = topil(result.cpu())
    img.convert("RGB").save("rcube.jpg")
