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

size = 512
pcube = Pcube()

r_y = torch.tensor([0.5, -0.7, 0.2]).detach()
y_t = pcube.posed(r_y)

topil = ToPILImage()
rnd = Render(256, pcube.uvs, pcube.uvmap)
y = rnd.forward(y_t)
y_img = topil(y)

# value to find
r = torch.zeros(3, requires_grad=True)

# fitting
num_epochs = 100
lrn_rate = 0.01
loss_func = torch.nn.L1Loss()
optimiser = torch.optim.Adam([r], lr=lrn_rate)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=[10, 5])

for epoch in range(num_epochs):
    y_prime_t = pcube.posed(r)
    y_prime = rnd.forward(y_prime_t)
    y_p_img = topil(y_prime)
    optimiser.zero_grad()
    loss = loss_func(y, y_prime)
    loss.backward()

    optimiser.step()
    ax[0].imshow(y_img.convert("RGB"))
    ax[0].set_title("target")
    ax[1].imshow(y_p_img.convert("RGB"))
    ax[1].set_title(f"current, epoch: {epoch:>02}")
    fig.savefig(f"fig/pcube{epoch:>03}.jpg", dpi=240)

    print(f"Epoch {epoch+1:4d}, "
          f"Loss: {loss:5.8f}, "
          "Values: {:0.3f} {:0.3f} {:0.3f}".format(*r))


"""
make a movie:
ffmpeg -r 25 \
    -pattern_type glob \
    -i 'fig/*.jpg' \
    -c:v libx264 \
    -pix_fmt yuv420p \
    out.mp4
"""