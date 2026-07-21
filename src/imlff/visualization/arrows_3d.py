
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter


def arrows_3d(ends, starts=None, ax=None, label=None, head_frac=0.1, **kwargs):
    """3D plot of multiple arrows

    Args:
        ends (ndarray): (N, 3) size array of arrow end coordinates
        starts (ndarray): (N, 3) size array of arrow start coordinates.
            Assume start position of (0, 0, 0) if not given
        ax (Axes3DSubplot): existing axes to add to
        label (str): legend label to apply to this group of arrows
        head_frac (float): fraction of arrow length used for the solid head segment
        kwargs (dict): additional arrow properties
    """
    if ends.shape == (3,):
        ends = ends[None]
        
    if starts is None:
        starts = np.zeros_like(ends)
    elif starts.shape == (3,):
        starts = starts[None]

    assert starts.shape == ends.shape, "`starts` and `ends` shape must match"
    assert len(ends.shape) == 2 and ends.shape[1] == 3, \
        "`starts` and `ends` must be shape (N, 3)"

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    if ax is None:
        ax = plt.figure().add_subplot(111, projection='3d')

    # Shaft kwargs: no arrowhead, keep dashed style
    shaft_kwargs = kwargs.copy()
    shaft_kwargs['arrowstyle'] = '-'
    
    # Head kwargs: solid line, with arrowhead.
    # matplotlib's -|> head needs linewidth > 0 to render at full size,
    # so we decouple it from the shaft linewidth (default 1).
    head_kwargs = kwargs.copy()
    head_kwargs.pop('linestyle', None)
    head_kwargs.pop('linewidth', None)   # drop shaft linewidth
    head_kwargs.setdefault('linewidth', 1)
    head_kwargs.setdefault('capstyle', 'butt')
    head_kwargs.setdefault('joinstyle', 'miter')
    head_kwargs.setdefault('arrowstyle', '-|>')

    arrow_base = dict(mutation_scale=20, shrinkA=0, shrinkB=0)

    artists = []
    for ind, (s, e) in enumerate(np.stack((starts, ends), axis=1)):

        tip_start = s + (1 - head_frac) * (e - s)
        print(tip_start)

        # --- Shaft: skip entirely when linewidth=0 (tip-only mode) ---
        if kwargs.get('linewidth', 1) != 0:
            shaft_props = {**arrow_base, **shaft_kwargs}
            shaft = Arrow3D(
                [s[0], tip_start[0]],
                [s[1], tip_start[1]],
                [s[2], tip_start[2]],
                label=label if ind == 0 else None,
                **shaft_props
            )
            ax.add_artist(shaft)
            artists.append(shaft)
        
        if '>' in kwargs.get('arrowstyle', ''):
            # --- Head: runs from tip_start to e, always solid ---
            head_props = {**arrow_base, **head_kwargs}
            # carry label here when shaft is skipped (linewidth=0)
            if kwargs.get('linewidth', 1) == 0:
                head_props['label'] = label if ind == 0 else None
            head = Arrow3D(
                [tip_start[0], e[0]],
                [tip_start[1], e[1]],
                [tip_start[2], e[2]],
                **head_props
            )
            ax.add_artist(head)
            artists.append(head)

    return artists