"""
Compose grid figures from figures/weather/*.png (after all 3 workers finish).
Run: python plot_weather_prompts.py
Outputs one grid per prompt: figures/weather/grid_<prompt>.png
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

METHODS       = ["WPD-base", "WPD-drop_ll"]
RADII         = [8, 12, 20, 24]
N_IMG         = 1
PROMPT_SLUGS  = ["day_clear", "overcast", "rain", "night", "fog"]
PROMPT_LABELS = ["Day (clear)", "Overcast", "Rain", "Night", "Fog"]

OUT_DIR = "figures/weather"
row_labels = [f"{m}\nr={r}" for m in METHODS for r in RADII]
n_rows = len(row_labels)
n_cols = 1 + N_IMG   # input col + one col per image

for slug, label in zip(PROMPT_SLUGS, PROMPT_LABELS):
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 1.5),
                             gridspec_kw=dict(hspace=0.04, wspace=0.03))

    for c, title in enumerate(["Input"] + [f"Image {i+1}" for i in range(N_IMG)]):
        axes[0, c].set_title(title, fontsize=8, fontweight='bold', pad=3)

    row = 0
    for mname in METHODS:
        for r in RADII:
            axes[row, 0].set_ylabel(f"{mname}\nr={r}", fontsize=7,
                                    rotation=0, labelpad=65, va='center')
            # input col: show image 0
            inp = Image.open(f"{OUT_DIR}/input_img0.png")
            axes[row, 0].imshow(np.array(inp))
            axes[row, 0].axis('off')
            for i in range(N_IMG):
                path = f"{OUT_DIR}/{mname}_r{r}_img{i}_{slug}.png"
                if os.path.exists(path):
                    axes[row, 1 + i].imshow(np.array(Image.open(path)))
                else:
                    axes[row, 1 + i].text(0.5, 0.5, 'missing', ha='center',
                                          va='center', transform=axes[row, 1+i].transAxes)
                axes[row, 1 + i].axis('off')
            row += 1

    plt.suptitle(f"Prompt: {label}  (J=4, same noise seed per image)",
                 fontsize=9, y=1.002)
    out = f"{OUT_DIR}/grid_{slug}.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

print("Done.")
