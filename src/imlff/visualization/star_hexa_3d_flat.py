
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_hex_star(ax, xyz, scale, color, weight=2.0, zorder=3, alpha=1.0):
    """Draw a LaTeX-style asterisk (*) lying flat in the z-plane of xyz.

    3 arms through center, quartic taper (thin shaft, wider rounded tips).
    Tip size is independent of scale.

    Args:
        ax:     Axes3D to draw on.
        xyz:    (3,) center position in world space.
        scale:  Arm half-length (center-to-tip radius).
        color:  Face color.
        weight: Arm thickness multiplier (1.0 = default).
        zorder, alpha: standard Matplotlib kwargs.
    """
    xyz = np.asarray(xyz, dtype=float)
    w_tip    = scale * 0.20 * weight   # tip half-width, scales with arm length
    w_center = w_tip * 0.30           # shaft half-width at center
    n_side   = 24
    n_cap    = 16

    ss     = np.linspace(-1.0, 1.0, n_side)
    widths = w_center + (w_tip - w_center) * ss**2   # quadratic: smooth concave waist

    polys = []
    for k in range(3):   # 3 diameters = 6 arms
        theta = k * np.pi / 3 + np.pi / 6
        d = np.array([ np.cos(theta),  np.sin(theta), 0.0])
        p = np.array([-np.sin(theta),  np.cos(theta), 0.0])

        verts = []

        # Right edge: s = -1 → +1, offset +p*w
        for s, w in zip(ss, widths):
            verts.append(xyz + s * scale * d + w * p)

        # Rounded cap at +tip: sweep from +p side → outside (+d) → -p side
        tip_r = xyz + scale * d
        for a in np.linspace(theta + np.pi / 2, theta - np.pi / 2, n_cap):
            verts.append(tip_r + w_tip * np.array([np.cos(a), np.sin(a), 0.0]))

        # Left edge: s = +1 → -1, offset -p*w
        for s, w in zip(ss[::-1], widths[::-1]):
            verts.append(xyz + s * scale * d - w * p)

        # Rounded cap at -tip: sweep from -p side → outside (-d) → +p side
        tip_l = xyz - scale * d
        for a in np.linspace(theta + 3 * np.pi / 2, theta + np.pi / 2, n_cap):
            verts.append(tip_l + w_tip * np.array([np.cos(a), np.sin(a), 0.0]))

        polys.append(np.array(verts))

    ax.add_collection3d(Poly3DCollection(polys, color=color, alpha=alpha,
                                         zorder=zorder, linewidths=0))
