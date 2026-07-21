import numpy as np
from matplotlib import colormaps, colors
import matplotlib.colors as mcolors
import colorsys

def adjust_saturation(cmap_name="Blues", factor=1.0, N=256, for_pyvista=False):
    base = colormaps[cmap_name](np.linspace(0, 1, N))  # (N,4) RGBA
    rgb = base[..., :3]
    gray = rgb.mean(axis=1, keepdims=True)
    rgb = np.clip(gray + factor * (rgb - gray), 0, 1)  # <1=desat, >1=upsat

    if for_pyvista:
        return [[i/(N-1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
                for i, (r, g, b) in enumerate(rgb)]
    else:
        base[..., :3] = rgb
        return colors.ListedColormap(base)


def adjust_color_saturation(color, factor=1.0):
    """Adjust saturation of a single RGB/hex color."""
    rgb = np.array(mcolors.to_rgb(color))
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s = np.clip(s * factor, 0, 1)
    return np.array(colorsys.hls_to_rgb(h, l, s))



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --- sRGB <-> CIE Lab conversion ---

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(np.clip(rgb, 0, 1))
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.00000, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1/3), 7.787 * xyz + 16/116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

def _lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = a / 500 + fy, fy - b / 200
    def finv(t): return np.where(t**3 > 0.008856, t**3, (t - 16/116) / 7.787)
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883])
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]])
    return np.clip(_linear_to_srgb(xyz @ M_inv.T), 0, 1)


# --- the actual function ---

def white_zero_cmap(cmap_name, n=256, reverse=False):
    """
    Returns a version of `cmap_name` that starts at pure white (L=100, a=0, b=0)
    and follows the original colormap's hue/chroma path in Lab space.
    """
    cmap = plt.get_cmap(cmap_name)
    rgb  = cmap(np.linspace(0, 1, n))[:, :3]
    lab  = _rgb_to_lab(rgb)

    # In Lab: interpolate from white toward original values
    t = np.linspace(0, 1, n)[:, None]          # (n,1) for broadcasting
    white_lab = np.array([100., 0., 0.])
    lab_new   = white_lab + t * (lab - white_lab)  # straight line in Lab

    colors = np.concatenate([_lab_to_rgb(lab_new), np.ones((n, 1))], axis=1)
    if reverse:
        colors = colors[::-1]
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_white", colors)


Blues_w   = white_zero_cmap("Blues")
Purples_w = white_zero_cmap("Purples")
Greens_w  = white_zero_cmap("Greens")