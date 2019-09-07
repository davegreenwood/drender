import torch
from torchvision.transforms import ToPILImage
from drender.utils import Pcube
from drender.render import Render
import matplotlib.pyplot as plt

DTYPE = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------


def project(v):
    """Simple projection"""
    v = v - torch.tensor([0, 0, 2], dtype=DTYPE, device=DEVICE)
    v = torch.cat([v[:, :2] / -v[:, 2, None], v[:, 2, None]], dim=1)
    return v * 2

print(DEVICE)
# torch.autograd.set_detect_anomaly(True)

size = 256
pcube = Pcube()

r_y = torch.tensor([0.5, -0.7, -0.6]).detach()
y_t = project(pcube.posed(r_y))


topil = ToPILImage()
rnd = Render(size, pcube.f, pcube.uv, pcube.uvf, pcube.uvmap)
y = rnd.forward(y_t)
y_img = topil(y)

# value to find
r = torch.zeros(3, requires_grad=True)

# fitting
num_epochs = 120
lrn_rate = 0.015
loss_func = torch.nn.L1Loss()
optimiser = torch.optim.Adam([r], lr=lrn_rate)

for epoch in range(num_epochs):
    y_prime = rnd(project(pcube.posed(r)))
    optimiser.zero_grad()
    loss = loss_func(y, y_prime)
    loss.backward()
    optimiser.step()

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=[10, 5])
    ax[0].imshow(y_img.convert("RGB"))
    ax[0].set_title("target")
    ax[1].imshow(topil(y_prime).convert("RGB"))
    ax[1].set_title(f"current, epoch: {epoch:>03}")
    fig.savefig(f"/Users/Shared/tmp/fig/pcube{epoch:>03}.jpg", dpi=100)
    plt.close()

    print(f"Epoch {epoch+1:4d}, "
          f"Loss: {loss:5.8f}, "
          "Values: {:0.3f} {:0.3f} {:0.3f}".format(*r))

"""
make a movie:
ffmpeg -r 25 \
    -pattern_type glob \
    -i '/Users/Shared/tmp/fig/*.jpg' \
    -c:v libx264 \
    -pix_fmt yuv420p \
    out.mp4

make a gif:
ffmpeg -f image2 -pattern_type glob -i '/Users/Shared/tmp/fig/*.jpg' out.gif
"""
