import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from imlff.common.util import proj_path

beta = 0.5

# h is front axis, x is depth axis
NGRID=400
h_range = np.linspace(-2.5, 2.5, NGRID)
x_range = np.linspace(-2.5, 2.5, NGRID)
H, X = np.meshgrid(h_range, x_range)   # H varies along front, X along depth
Z_ising = np.tanh(beta * H + X)
Z_diag  = H.copy() # * 0. - 1.

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False

# ---- Fixed-point curve ----
h_scan = np.linspace(-2.5, 2.5, 5000)
x_fp   = np.linspace(-2.5, 2.5, 500)
h_fp   = []
for xv in x_fp:
    F  = lambda h: np.tanh(beta*h + xv) - h
    sc = np.where(np.diff(np.sign(F(h_scan))))[0]
    h_fp.append(brentq(F, h_scan[sc[0]], h_scan[sc[0]+1]) if sc.size else np.nan)
h_fp = np.array(h_fp)

# ---- Facecolors: compute base colors per face ----
above = Z_ising >= Z_diag  # per vertex
rows, cols = Z_ising.shape
z_min, z_max = float(Z_ising.min()), float(Z_ising.max())
fc_base = np.zeros((rows-1, cols-1, 4))
face_above = np.zeros((rows-1, cols-1), dtype=bool)
for i in range(rows-1):
    for j in range(cols-1):
        z_face = 0.25*(Z_ising[i,j]+Z_ising[i+1,j]+Z_ising[i,j+1]+Z_ising[i+1,j+1])
        t = (z_face - z_min) / (z_max - z_min + 1e-9)
        shade = 1 - (0.28 + 0.44 * t)
        fc_base[i,j] = plt.cm.Blues(shade)
        # face is "above" diag if majority of corners are above
        face_above[i,j] = (above[i,j].astype(int) + above[i+1,j] + above[i,j+1] + above[i+1,j+1]) >= 2

# Split into two copies: above-diag gets high zorder, below-diag gets low zorder
fc_above = fc_base.copy()
fc_below = fc_base.copy()
fc_above[~face_above, 3] = 0.0   # hide below-diag faces in the "above" copy
fc_above[ face_above, 3] = 0.98
fc_below[ face_above, 3] = 0.0   # hide above-diag faces in the "below" copy
fc_below[~face_above, 3] = 0.98

# Ising below diag (render behind the diagonal plane)
ax.plot_surface(H, X, Z_ising, facecolors=fc_below,
                linewidth=0, antialiased=True, shade=False, zorder=1)
# Diagonal plane
ax.plot_surface(H, X, Z_diag, color='#1a1a1a', alpha=0.42,
                linewidth=0, antialiased=True, zorder=2)
# Ising above diag (render in front of the diagonal plane)
ax.plot_surface(H, X, Z_ising, facecolors=fc_above,
                linewidth=0, antialiased=True, shade=False, zorder=3)

# ---- Border frames ----
_bkw = dict(color='#333333', lw=0.8, alpha=0.7, zorder=4)
# Ising surface border
for Z in [Z_ising]:
    ax.plot(H[0,:],  X[0,:],  Z[0,:],  **_bkw)   # front edge
    ax.plot(H[-1,:], X[-1,:], Z[-1,:], **_bkw)   # back edge
    ax.plot(H[:,0],  X[:,0],  Z[:,0],  **_bkw)   # left edge
    ax.plot(H[:,-1], X[:,-1], Z[:,-1], **_bkw)   # right edge
# Diagonal plane border — split by whether diag is above/below Ising
_bkw_hi = dict(color='#333333', lw=0.8, alpha=0.7, zorder=4)   # diag on top
_bkw_lo = dict(color='#333333', lw=0.8, alpha=0.7, zorder=0)   # diag under Ising
_diag_edges = [
    (H[0,:],  X[0,:],  Z_diag[0,:],  Z_ising[0,:]),
    (H[-1,:], X[-1,:], Z_diag[-1,:], Z_ising[-1,:]),
    (H[:,0],  X[:,0],  Z_diag[:,0],  Z_ising[:,0]),
    (H[:,-1], X[:,-1], Z_diag[:,-1], Z_ising[:,-1]),
]
for eh, ex, ed, ei in _diag_edges:
    diag_on_top = ed >= ei
    # find contiguous segments
    changes = np.where(np.diff(diag_on_top.astype(int)))[0]
    splits = np.split(np.arange(len(eh)), changes + 1)
    for seg in splits:
        if len(seg) < 2:
            continue
        kw = _bkw_hi if diag_on_top[seg[0]] else _bkw_lo
        ax.plot(eh[seg], ex[seg], ed[seg], **kw)

# ---- Fixed-point curve: (h_fp, x_fp, h_fp) ----
HOVER_FP = 0.12
si,ei=4,-4
ax.plot(h_fp[si:ei], x_fp[si:ei], (h_fp+HOVER_FP)[si:ei], color='black', lw=8, alpha=0.4, zorder=11)
ax.plot(h_fp[si:ei], x_fp[si:ei], (h_fp+HOVER_FP)[si:ei], color='crimson', lw=3.5, ls='--', zorder=12,
        label=r'Fixed-point curve $h^*(x)$')

# ---- Trajectories ----
HOVER = 0.08
def traj_on_surface(x0, h0, beta, n_iter=35, n_interp=50):
    hs = [h0]
    h_cur = h0
    for _ in range(n_iter):
        h_new = np.tanh(beta * h_cur + x0)
        hs.append(h_new)
        h_cur = h_new
        if abs(h_new - hs[-2]) < 1e-5:
            break
    h_fine = []
    for i in range(len(hs)-1):
        h_fine.extend(np.linspace(hs[i], hs[i+1], n_interp, endpoint=False))
    h_fine.append(hs[-1])
    h_fine = np.array(h_fine)
    return h_fine, np.full_like(h_fine, x0), np.tanh(beta*h_fine+x0)+HOVER, np.array(hs)

traj_specs = [
    (-1.8, -2., '#00eeff'),
    ( 0.3, -2., '#ffff33'),
    ( 2.0, -2., '#ee44ff'),
]
for x0, h0, color in traj_specs:
    hf, xf, zf, hs_steps = traj_on_surface(x0, h0, beta)
    ax.plot(hf, np.full_like(hf, x0), zf, 
            label=f'Solver trajectory (x={x0})',
            color=color, lw=4.5, alpha=1.0, zorder=13)
    zs_steps = np.tanh(beta*hs_steps+x0)+HOVER
    ax.scatter(hs_steps, np.full_like(hs_steps, x0), zs_steps,
               color=color, s=40, depthshade=False, zorder=15,
               edgecolors='k', linewidths=0.5)
    # START = ball
    ax.scatter([h0], [x0], [np.tanh(beta*h0+x0)+HOVER],
               color=color, s=100, marker='o', depthshade=False, zorder=16,
               edgecolors='k', linewidths=1.2)
    # END = star
    F = lambda h: np.tanh(beta*h+x0) - h
    hstar = brentq(F, -2.5, 2.5)
    ax.scatter([hstar], [x0], [hstar+HOVER], color=color, s=320, marker='*',
               edgecolors='k', linewidths=0.8, depthshade=False, zorder=16)

ax.set_xlabel(r'$h$', fontsize=14, labelpad=10)
ax.set_ylabel(r'$x$', fontsize=14, labelpad=10)
ax.set_zlabel(r'$\tanh(h/2 + x)$', fontsize=12, labelpad=10)
ax.set_title(rf'$h^{{(k+1)}} = \tanh(\beta h^{{(k)}} + x),\quad \beta={beta}$',
             fontsize=13, pad=15)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(2.5, -2.5)
# ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
ax.view_init(elev=24, azim=150)
ax.view_init(elev=8, azim=160)

# ---- Style options (uncomment to try) ----

# Option A: remove grey pane backgrounds entirely
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Option B: make panes transparent with subtle edge
# ax.xaxis.pane.set_alpha(0.05)
# ax.yaxis.pane.set_alpha(0.05)
# ax.zaxis.pane.set_alpha(0.05)
# ax.xaxis.pane.set_edgecolor((0, 0, 0, 0.15))
# ax.yaxis.pane.set_edgecolor((0, 0, 0, 0.15))
# ax.zaxis.pane.set_edgecolor((0, 0, 0, 0.15))

# Option C: remove grid lines
ax.grid(False)

# Option D: remove grid lines on individual axes
# ax.xaxis._axinfo['grid'].update({'linewidth': 0})
# ax.yaxis._axinfo['grid'].update({'linewidth': 0})
# ax.zaxis._axinfo['grid'].update({'linewidth': 0})

# Option E: lighter/dashed grid
# ax.xaxis._axinfo['grid'].update({'color': (0.8, 0.8, 0.8, 0.5), 'linestyle': '--', 'linewidth': 0.5})
# ax.yaxis._axinfo['grid'].update({'color': (0.8, 0.8, 0.8, 0.5), 'linestyle': '--', 'linewidth': 0.5})
# ax.zaxis._axinfo['grid'].update({'color': (0.8, 0.8, 0.8, 0.5), 'linestyle': '--', 'linewidth': 0.5})

# Option F: remove axis lines (spines)
# ax.xaxis.line.set_visible(False)
# ax.yaxis.line.set_visible(False)
# ax.zaxis.line.set_visible(False)

# Option G: remove tick marks but keep labels
# ax.tick_params(axis='both', length=0)

# Option H: minimal — no panes, no grid, no spines
# ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
# ax.grid(False)
# ax.xaxis.line.set_visible(False); ax.yaxis.line.set_visible(False); ax.zaxis.line.set_visible(False)


# ---- Present ----

plt.tight_layout()
pat = proj_path('../imlff-paper/figures/ising/ising3d.pdf')
plt.savefig(pat, dpi=150, bbox_inches='tight')
plt.show()