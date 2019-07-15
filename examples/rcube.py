import time
import numpy as np
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
    tris, uvs, uvmap = rcube.get_data()
    rnd = Render(size, uvs, uvmap)

    t0 = time.time()
    result = rnd.forward(tris)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = topil(result)
    img.convert("RGB").save("rcube.jpg")
