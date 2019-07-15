import time
import numpy as np
from PIL import Image
from drender.utils import Rcube
from drender.render import Render

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # set defaults
    size = 1024
    rcube = Rcube()
    tris, uvs, uvmap = rcube.get_data()
    rnd = Render(size, uvs, uvmap)

    t0 = time.time()
    rnd.render(tris)
    t1 = time.time()
    print(f"time: {t1-t0:0.2f}")

    # image out
    img = rnd.result.cpu().numpy()[:, :, :3].squeeze()
    img -= img.min()
    img /= img.max()
    img *= 255
    Image.fromarray(img.astype(np.uint8)).convert("RGB").save("rcube.jpg")
