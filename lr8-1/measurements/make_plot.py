#!/usr/bin/env python3
"""
make_plot.py
Reads results_of_calculations.npz (must contain arrays 'x' and 'u'),
creates a plot with dark background, saves PNG and a Markdown file that embeds the PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import sys
import os

npz_file = 'results_of_calculations.npz'
out_png = 'plot_u_vs_x.png'
out_md  = 'plot.md'

if not os.path.exists(npz_file):
    print(f"Error: '{npz_file}' not found in the current directory ({os.getcwd()}).", file=sys.stderr)
    sys.exit(2)

# load
data = np.load(npz_file)
if 'x' not in data or 'u' not in data:
    print(f"Error: '{npz_file}' must contain arrays named 'x' and 'u'. Found: {list(data.keys())}", file=sys.stderr)
    sys.exit(2)

x = np.array(data['x'])
u = np.array(data['u'])

# flatten 1D data safely
x = x.ravel()

# try to make u 1D that matches x length
if u.shape != x.shape:
    # common situations:
    if u.ndim > 1 and u.shape[0] == x.shape[0]:
        # pick last column (often u[:, -1] = final time)
        u = u[:, -1].ravel()
    elif u.ndim > 1 and u.shape[1] == x.shape[0]:
        # maybe u is shape (time, x) and we want the last time
        u = u[-1, :].ravel()
    else:
        # fallback: try a simple ravel but warn if lengths mismatch
        u = u.ravel()
        if u.shape[0] != x.shape[0]:
            print("Warning: lengths of x and u do not match after heuristics."
                  f" len(x)={x.shape[0]}, len(u)={u.shape[0]}. Plot may be incorrect.", file=sys.stderr)

# axis limits a,b from x range
a = float(np.min(x))
b = float(np.max(x))

# plotting
style.use('dark_background')   # as you requested
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(a, b)
ax.set_ylim(-2.0, 2.0)
ax.set_xlabel('x')
ax.set_ylabel('u')

# explicit color requested in your snippet; OK because you asked for it
ax.plot(x, u, color='y', ls='-', lw=2)

fig.tight_layout()
fig.savefig(out_png, dpi=200)
print(f"Saved plot image -> {out_png}")

# also write a lightweight markdown file embedding the image
md_content = f"""# Plot of u vs x

This figure was generated from `{npz_file}` (arrays `x` and `u`).

![u vs x]({out_png})

*Plot created with matplotlib (dark background) — x-limits set to [{a}, {b}], y-limits to [-2.0, 2.0].*
"""

with open(out_md, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Saved markdown -> {out_md}")

# show the plot interactively (optional)
plt.show()
