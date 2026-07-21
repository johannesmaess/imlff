"""
arrows_3d_flat: draw arrows that lie flat in 3D space on a 3D axes.

Each arrow is rendered as two filled polygons (shaft rectangle + head triangle)
whose corners are computed in world space and passed to Poly3DCollection,
so the 3D axes handles perspective/projection natively.

Intended interface compatibility with the arrows_3d() helper used in plot_streamline.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


def _px_per_world(start, end, ax):
    """Estimate display pixels per world unit along the start→end segment."""
    pts = np.array([start, end])
    M = ax.get_proj()
    xs, ys, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], M)
    disp = ax.transData.transform(np.column_stack([xs, ys]))
    px_dist = np.linalg.norm(disp[1] - disp[0])
    world_dist = np.linalg.norm(end - start)
    return px_dist / world_dist if world_dist > 1e-12 else 1.0


def arrows_3d_flat(
    ends,
    starts=None,
    ax=None,
    # geometry
    # head_length=0.05,     # absolute head length in world units (fixed, independent of arrow length)
    # shaft_width=0.006,    # half-width of the shaft rectangle, world units
    head_length=0.06, shaft_width=0.014,
    head_concave=0.2,    # 0 = flat back (triangle), 1 = fully swept; controls concavity
    mutation_scale=None,  # accepted for interface compatibility, not used
    # style
    arrowstyle="-|>",     # '-|>' with head, '-' shaft only
    color="white",
    alpha=1.0,
    zorder=2,
    label=None,
    # compatibility kwargs accepted but ignored
    shrinkA=None,  # accepted for interface compatibility, not used (screen-space param)
    shrinkB=None,  # accepted for interface compatibility, not used (screen-space param)
    linewidth=None,
    linestyle=None,
    capstyle=None,
    joinstyle=None,
    **kwargs,
):
    """Draw arrows lying flat in 3D space, compatible with Axes3D.

    Each arrow is a filled shaft rectangle + triangular head, rendered as a
    Poly3DCollection so the axes' own projection (including perspective) is used.

    Args:
        ends (ndarray):   (N, 3) arrow tip positions in world space.
        starts (ndarray): (N, 3) arrow tail positions. Defaults to origin.
        ax:               Axes3D to draw on. Defaults to plt.gca().
        head_length:      Absolute head length in world units (clamped to arrow length).
        shaft_width:      Half-width of the shaft rectangle, world units.
        mutation_scale:   Accepted for interface compatibility, has no effect.
        arrowstyle:       '-|>' = head + shaft, '-' = shaft only.
        shrinkA/shrinkB:  Shorten arrow at tail/tip by this world-space amount.
        color, alpha, zorder, label: standard Matplotlib kwargs.
        linewidth, linestyle, capstyle, joinstyle: accepted for interface
            compatibility with arrows_3d() but have no effect.
    """
    ends   = np.asarray(ends,   dtype=float)
    starts = np.zeros_like(ends) if starts is None else np.asarray(starts, dtype=float)

    if ends.ndim == 1:
        ends   = ends[None]
        starts = starts[None]

    if ax is None:
        ax = plt.gca()

    has_head = arrowstyle != "-"
    sw = shaft_width

    shaft_polys = []
    head_polys  = []

    for start, end in zip(starts, ends):
        vec    = end - start
        length = np.linalg.norm(vec)
        if length < 1e-12:
            continue
        d = vec / length

        # perpendicular in the plane of the arrow
        # for horizontal arrows d is in xy, so cross(d, z_hat) stays in xy
        z_hat = np.array([0.0, 0.0, 1.0])
        perp  = np.cross(d, z_hat)
        if np.linalg.norm(perp) < 1e-9:
            perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
        perp /= np.linalg.norm(perp)

        if shrinkA or shrinkB:
            scale = _px_per_world(start, end, ax)
            sa = (shrinkA or 0) / scale
            sb = (shrinkB or 0) / scale
            s  = start + d * sa
            e  = end   - d * sb
            d2 = e - s
            l2 = np.linalg.norm(d2)
            if l2 < 1e-12:
                continue
            d2 /= l2
        else:
            s, e, l2, d2 = start, end, length, d

        # rounded cap helper: semicircle at point `pt` facing direction `back`
        n_cap = 9
        def _cap(pt, back):
            ts = np.linspace(-np.pi / 2, np.pi / 2, n_cap)
            return pt + sw * np.cos(ts)[:, None] * back + sw * np.sin(ts)[:, None] * perp

        if has_head:
            hl          = min(head_length, l2)
            neck        = e - d2 * hl
            concave_pt  = neck + d2 * (hl * head_concave)
            hw          = shaft_width * 2.5

            # shaft: straight top edge → rounded tail → straight bottom edge
            cap = _cap(s, -d2)                          # semicircle at tail, facing back
            shaft_polys.append(np.vstack([
                (concave_pt + perp * sw)[None],
                cap[::-1],                              # top → back → bottom
                (concave_pt - perp * sw)[None],
            ]))
            # 4-point head: left-wing → tip → right-wing → concave-back
            head_polys.append(np.array([neck       + perp * hw,
                                        e,
                                        neck       - perp * hw,
                                        concave_pt]))
        else:
            cap_s = _cap(s, -d2)                        # rounded tail
            cap_e = _cap(e,  d2)                        # rounded front
            shaft_polys.append(np.vstack([cap_e[::-1], cap_s]))

    if not shaft_polys:
        return

    common = dict(color=color, alpha=alpha, zorder=zorder, linewidths=0)

    col = Poly3DCollection(shaft_polys, **common)
    if label is not None:
        col.set_label(label)
    ax.add_collection3d(col)

    if head_polys:
        ax.add_collection3d(Poly3DCollection(head_polys, **common))