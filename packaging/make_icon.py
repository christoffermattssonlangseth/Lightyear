"""Draw the Lightyear app icon (a four-point star on a violet squircle) and build Lightyear.icns."""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_alpha(0)
ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                            fc="#5b4bd6", ec="none"))
t = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 2
r = np.where(np.arange(8) % 2 == 0, 0.30, 0.075)
ax.add_patch(Polygon(np.c_[0.5 + r * np.cos(t), 0.5 + r * np.sin(t)], fc="white", ec="none"))
for dx, dy, s in [(0.72, 0.72, 0.035), (0.29, 0.31, 0.022), (0.74, 0.33, 0.016)]:
    tt = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 2
    rr = np.where(np.arange(8) % 2 == 0, s, s * 0.3)
    ax.add_patch(Polygon(np.c_[dx + rr * np.cos(tt), dy + rr * np.sin(tt)], fc="#d9d3ff", ec="none"))
png = out / "icon_1024.png"
fig.savefig(png, transparent=True)
iconset = out / "Lightyear.iconset"
iconset.mkdir(exist_ok=True)
for size in (16, 32, 128, 256, 512):
    for scale in (1, 2):
        name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
        subprocess.run(["sips", "-z", str(size * scale), str(size * scale), str(png), "--out", str(iconset / name)],
                       check=True, capture_output=True)
subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out / "Lightyear.icns")], check=True)
print(out / "Lightyear.icns")
