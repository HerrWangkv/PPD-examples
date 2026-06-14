"""
Grid: input | WPD-base r12 | WPD-drop_ll r12  for Night prompt, 8 samples.
Rows = samples, cols = Input / WPD-base / WPD-drop_ll
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

N      = 8
OUT    = "figures/weather_night/grid_night.png"
D      = "figures/weather_night"

fig, axes = plt.subplots(N, 3, figsize=(3*5, N*1.5),
                         gridspec_kw=dict(hspace=0.03, wspace=0.03))

for c, title in enumerate(["Input", "WPD-base  r=12", "WPD-drop_ll  r=12"]):
    axes[0, c].set_title(title, fontsize=9, fontweight='bold', pad=4)

for i in range(N):
    axes[i, 0].imshow(np.array(Image.open(f"{D}/input_img{i}.png")))
    axes[i, 1].imshow(np.array(Image.open(f"{D}/WPD-base_img{i}.png")))
    axes[i, 2].imshow(np.array(Image.open(f"{D}/WPD-drop_ll_img{i}.png")))
    for c in range(3):
        axes[i, c].axis('off')

plt.suptitle('Night prompt  —  WPD-base vs WPD-drop_ll  (J=4, r=12)',
             fontsize=10, y=1.002)
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"Saved: {OUT}")
