import numpy as np
import matplotlib.pyplot as plt

def plot_npz(filename='results_of_calculations.npz',
             xname='x', yname='y', uname='u', tname='t',
             a=None, b=None, c=None, d=None,
             vmin=None, vmax=None, figsize=(8,6)):
    """
    Load data from an .npz file and display a pcolor plot.

    - filename: path to the .npz file
    - xname, yname, uname: keys inside the npz for x, y and u arrays
    - tname: optional key for time / a scalar to show in the title
    - a,b,c,d: optional axis limits; if None they will be taken from x and y
    - vmin, vmax: optional color limits
    """
    data = np.load(filename)
    # ensure keys exist
    for key in (xname, yname, uname):
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {filename}. Available: {list(data.keys())}")

    x = data[xname]
    y = data[yname]
    u = data[uname]
    t = data[tname] if tname in data else None

    # grid
    X, Y = np.meshgrid(x, y)  # shapes: (len(y), len(x))

    # determine correct orientation of u for plotting
    if u.shape == X.shape:
        u_plot = u
    elif u.shape == (len(x), len(y)):
        # common case: u was created with shape (len(x), len(y)) => transpose
        u_plot = u.T
    else:
        # try to broadcast if possible, otherwise inform user
        try:
            # attempt to reshape if one dimension matches
            if u.size == X.size:
                u_plot = u.reshape(X.shape)
            else:
                raise ValueError
        except Exception:
            raise ValueError(f"Shape mismatch: x->({len(x)},), y->({len(y)},), "
                             f"meshgrid->{X.shape}, u->{u.shape}. "
                             "Transpose or reshape your u so it matches (len(y), len(x)).")

    # axis limits
    a = x.min() if a is None else a
    b = x.max() if b is None else b
    c = y.min() if c is None else c
    d = y.max() if d is None else d

    # plotting
    plt.style.use('dark_background')
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(xlim=(a, b), ylim=(c, d))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # use pcolor (keeps your original call). shading='auto' avoids warnings about dims.
    pc = ax.pcolor(X, Y, u_plot, shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label(uname)

    if t is not None:
        # show scalar time if present; if it's an array or has many values, this will show its repr
        try:
            t_val = float(np.asarray(t))
            ax.set_title(f't = {t_val}')
        except Exception:
            ax.set_title(f't = {t}')

    plt.show()


if __name__ == '__main__':
    # default call — assumes file is in current directory and keys are x,y,u,t
    plot_npz('results_of_calculations.npz')
