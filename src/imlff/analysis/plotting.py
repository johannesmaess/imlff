import numpy as np
import pandas as pd

import matplotlib

from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import matplotlib.pyplot as plt

## Convergence plot

def order_of_convergence(a):
    a = np.asarray(a, dtype=float).copy()
    nan = np.isnan(a)
    cnt = np.cumsum(nan, axis=0) * nan
    last_idx = np.where(~nan, np.arange(a.shape[0])[:, None], -1).max(axis=0)
    last_val = a[last_idx, np.arange(a.shape[1])]
    a[nan] = (last_val[None, :] - cnt)[nan]
    order = np.argsort(a[-1])
    return order

def read_degrees_and_model_tags(forw):
    set_ws_degrees = set()
    set_model_tags = set()

    for data in forw.values():
        for (model_tag, warmstart_degree) in data:
            set_ws_degrees.add(warmstart_degree)
            set_model_tags.add(model_tag)
            
    return sorted(set_ws_degrees), sorted(set_model_tags)


def convergence_plots(
    data, # list of nested dicts
    
    get_colour,
    get_label,
    get_linestyle,
    
    temp_tols = None,
    
    data_labels = ['Forward', 'Backward'],
    run_model_tags_func = lambda _: True,

    seperate_degrees = False,
    plot_ws_degrees = None,
    # plot_ws_degrees = read_degrees(forw),

    red_line_at = 0.01,
    # plot_up_to = 0.0001,
    plot_up_to = 0,

    overlay_configs = False,
    fig = None,

    plot_median = True,
    plot_confidence = True,
    plot_samples = None, # give a list of indices or a slice object
    
    pull_preceding_label_into_legend_title=':',
    avg_cost_over_datas = True,
    show=True,
    no_legend=False,
    
):
    if plot_ws_degrees is None:
        plot_ws_degrees, _ = read_degrees_and_model_tags(data[0])

    if temp_tols is None:
        temp_tols = data[0].keys()
        
    figs=[]
    axss=[]

    for temp_tol in temp_tols:
        temp, tol = temp_tol
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        n_rows = len(plot_ws_degrees) if seperate_degrees else 1

        n_cols = len(data_labels)
        if (fig is None) or not overlay_configs:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols+3, 7*n_rows))
            # add axs
            if n_rows == 1 and n_cols == 1: axs = np.array([[axs]])
            elif               n_cols == 1: axs = axs[:, None]
            elif n_rows == 1              : axs = axs[None, :]
        
        if avg_cost_over_datas:
            mean_cost_store = {}
        for i_col, (data, ax_row, title) in enumerate(zip(data, axs.T, data_labels)):
            if temp_tol not in data:
                continue
            data = data[temp_tol]
            # arrays: list of np.ndarray, each of shape (10, 100)
            x = np.arange(1,11)
            
            for ax, ws_degree in zip(ax_row, plot_ws_degrees):
                max_per_iter = -np.inf
                for (model_tag, warmstart_degree), dict_over_seeds in data.items():
                    
                    if not run_model_tags_func(model_tag):
                        continue
                    if warmstart_degree not in plot_ws_degrees:
                        continue
                    if seperate_degrees and warmstart_degree != ws_degree:
                        continue
                    tag = model_tag.split('model-')[-1] # todo?
                    
                    colour_confidence = colour = get_colour(model_tag=model_tag, warmstart_degree=warmstart_degree)
                    label = get_label(model_tag=model_tag, warmstart_degree=warmstart_degree)
                    linestyle = get_linestyle(warmstart_degree)
                        
                    all_seeds = list(dict_over_seeds.values())
                    if len(all_seeds):
                        try:
                            all_seeds = np.array(all_seeds).transpose((1,0,2))
                            all_seeds = all_seeds.reshape((len(all_seeds), -1))
                            
                            if plot_samples is not None:
                                order = order_of_convergence(all_seeds[:, :])
                                ax.plot(x, all_seeds[:, order[plot_samples]], lw=.5, color=colour, linestyle=linestyle)
                            
                            mask = np.isnan(all_seeds)
                            mask[0] = False
                            isnan_mean = mask.mean(1)
                            # all_seeds = np.where(mask, mins, all_seeds)
                            all_seeds[mask] = np.nanmin(all_seeds)
                            # all_seeds[mask] = 0.
                            
                            # q_low, med, q_high = np.nanpercentile(all_seeds, [5, 50, 95], axis=1)
                            
                            # all_seeds = np.nan_to_num(all_seeds, nan=tol)
                            lower, upper = 5, 95
                            q_low, med, q_high = np.percentile(all_seeds, [upper, 50, lower], axis=1)
                            
                            # don't plot median anymore where confidence bars are zero. 
                            make_nan = q_low == q_high
                            make_nan = isnan_mean*100 > (50 + 0.44) # give a tiny bit of leeway (for one specific plot)
                            med[make_nan] = np.nan
                            # plot conficence bars for one longer
                            make_nan[np.argmax(make_nan)] = False
                            # but only if at least 5% of samples still had a residual
                            make_nan[isnan_mean*100>upper] = True
                            make_nan = isnan_mean*100>upper
                            q_low[make_nan], q_high[make_nan] = np.nan, np.nan
                            
                            
                            if 1 and overlay_configs:
                                # if 'jac' not in label and 'No Reg.' not in label:
                                #     continue
                                # if '0.32' not in label:
                                #     continue
                                if tol!=0 and 1:
                                    colour_confidence = 'black'
                                else:
                                    label += ' (tol 0.0)'
                                print(med.round(4)[:4], label)

                            idx = np.argmax(all_seeds <= 1e-2, axis=0)
                            found_no_converged = (all_seeds > 1e-2) | np.isnan(all_seeds)
                            found_no_converged_for_sample = np.all(found_no_converged, axis=0)
                            idx[found_no_converged_for_sample] = 9  # none found -> 10
                            median_cost = np.median(idx) + 1
                            mean_cost = np.mean(idx) + 1
                            
                            # average costs over the different data's in differnet cols
                            if avg_cost_over_datas:
                                # store contribution to mean_cost
                                mean_cost_store[(model_tag, warmstart_degree)] = (
                                      mean_cost_store.get((model_tag, warmstart_degree), 0)
                                    + mean_cost / n_cols
                                )
                                
                                # retrieve
                                if i_col+1==n_cols:
                                    mean_cost = mean_cost_store[(model_tag, warmstart_degree)]
                                else:
                                    mean_cost = np.nan

                            # write mean cost 
                            rep = f'{mean_cost:6.2f} '
                            # rep = f'| {mean_cost:1.2f} {model_tag}|'
                            label = label.replace('%%', rep)

                            if plot_median:
                                ax.plot(x, med, lw=2, label=label, color=colour, linestyle=linestyle)
                            if plot_confidence: 
                                ax.fill_between(x, q_low, q_high, alpha=0.2, color=colour_confidence)
                            
                            q_high[0] = np.inf # if nan, it only mean there was no "0th" initial guess
                            max_per_iter_curr = np.nan_to_num(q_high if plot_confidence else med, nan=-np.inf) # other nans are below threshold
                            max_per_iter = np.maximum(max_per_iter, max_per_iter_curr)
                        except ValueError as e:
                            print(f"Skipping {label} {(model_tag, warmstart_degree)} due to {e}")
                            raise e
                            continue

                ax.set_title(f"{title} solver\n(MD at {temp}K. Up to tolercance {tol}.)")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Residual")
                l,u = ax.get_ylim()
                # ax.set_ylim(0, min(.1, u))
                # ax.set_xlim(1, 8)
                ax.set_yscale('log')
                
                # find index where all all below threshold
                if type(max_per_iter) is float:
                    continue
                indices_below_threshold = np.nonzero(max_per_iter <= plot_up_to)[0]
                
                first_idx = indices_below_threshold[0] if len(indices_below_threshold) else len(max_per_iter)-1
                ax.set_xlim(1, first_idx+1)
                ax.set_xticks(np.arange(1, first_idx+2))
                    
            
        ### LEGEND
        
        def pad_labels_and_title(labels, title):
            title = title + " | cost  | FMAE"
            def count_after_pipe(s):
                return len(s) - s.rfind("|") - 1 if "|" in s else 0

            max_tail = max(
                [count_after_pipe(title)] + [count_after_pipe(lbl) for lbl in labels]
            )
            target_head_length = max(
                [lbl.find("|") for lbl in labels] # only of lbls
            )

            def pad_right_to_tail(s, max_tail):
                tail = count_after_pipe(s)
                return s + " " * (max_tail - tail)
            def pad_before_pipe(s, target_pos):
                pos = s.find("|")
                if pos == -1:
                    return s
                return s[:pos] + " " * (target_pos - pos) + s[pos:]


            title = pad_right_to_tail(title, max_tail)
            labels = [pad_right_to_tail(
                        pad_before_pipe(l, target_head_length), 
                        max_tail) 
                      for l in labels]
            
            return labels, title
    
        def pad_and_ra(legend):
            # pad title and labels
            labels = [t.get_text() for t in legend.get_texts()]
            title = legend.get_title().get_text()
            labels, title = pad_labels_and_title(labels, title)
            for t, lbl in zip(legend.get_texts(), labels):
                t.set_text(lbl)
            legend.get_title().set_text(title)

            # right align
            legend._legend_box.align = "right"
            # (optional) also right-align each column/row packer inside the handle box
            for col in legend._legend_handle_box.get_children():
                col.align = "right"
                for row in col.get_children():
                    row.align = "right"

            # force a redraw so the packers recompute positions
            # legend.axes.figure.canvas.draw_idle()
            fig.canvas.draw_idle()
        
        legend_ax = axs[0][-1]
        
        if no_legend:
            pass
        elif not pull_preceding_label_into_legend_title:
            legend = legend_ax.legend(
                title='Warmstart strategy',
                prop={"family": "monospace"},
                title_fontproperties={"family": "monospace"},
            )
            pad_and_ra(legend)

        else:
            handles, labels = legend_ax.get_legend_handles_labels()
            
            ## sort labels by regularisation type (head == jac/trunc/...)
            grouped = {}
            for handle, label in zip(handles, labels):
                if label:
                    parts = label.split(pull_preceding_label_into_legend_title, 1)
                    head = parts[0]
                    tail = parts[1] if len(parts) > 1 else ""
                    grouped.setdefault(head, []).append((handle, tail))

            if grouped:
                base_x = 0.98
                base_x = 0.9
                base_y = 1.0
                base_y = .9
                y = base_y
                
                # print(grouped.items())
                for head, items in grouped.items():
                    def sort_by_leading_float_in_lbl(x):
                        lbl = x[1]
                        
                        if '|' not in lbl:
                            return -1e9
                        
                        num_str = lbl.split('|')[0]
                        if all(c==' ' for c in num_str):
                            return -1e9
                            
                        return float(num_str)
                    items = sorted(items, key=sort_by_leading_float_in_lbl)
                    group_handles = [h for h, _ in items]
                    group_labels = [l for _, l in items]
                    group_title = head.ljust(5).replace('%', ' ')
                    
                    # group_labels, group_title = pad_labels_and_title(group_labels, group_title)
                    
                    legend = fig.legend(
                        group_handles,
                        group_labels,
                        title=group_title,
                        prop={"family": "monospace"},
                        title_fontproperties={"family": "monospace"},
                        loc="upper left",
                        bbox_to_anchor=(base_x, y),
                        bbox_transform=fig.transFigure,
                    )
                    # legend.get_title().set_ha("right")
                    # legend.get_title().set_position((1, 0))
                    # legend.get_title().set_multialignment("right")
                    # legend._legend_box.align = "right"
                    
                    pad_and_ra(legend)


                    legend.set_in_layout(True)
                    # fig.canvas.draw_idle()
                    bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
                    height = bbox.height / fig.bbox.height
                    y -= height + .03
                
        for ax in axs.flatten():
            ax.axhline(red_line_at, color='red', linestyle='--', linewidth=1)

        if (not overlay_configs) and show:
            plt.show()
            
        figs.append(fig)
        axss.append(axs)
            
    return figs, axss, mean_cost_store


### Scatter plot

def _bboxes_overlap(a, b, pad=0.0):
    return (a.x0 - pad < b.x1 and a.x1 + pad > b.x0 and
            a.y0 - pad < b.y1 and a.y1 + pad > b.y0)
    
def _bbox_inside(b, outer, pad=0.0):
    return (b.x0 >= outer.x0 + pad and
            b.y0 >= outer.y0 + pad and
            b.x1 <= outer.x1 - pad and
            b.y1 <= outer.y1 - pad)

def place_labels_no_overlap(ax, xs, ys, labels, colors, *,
                            fontsize=8,
                            offsets_pts=((6,6),(6,-6),(-6,6),(-6,-6),(10,0),(-10,0),(0,10),(0,-10)),
                            point_avoid_px=2,
                            label_pad_px=1):
    """
    Greedy label placement: try a small set of point-relative offsets (in points),
    pick first that doesn't overlap previously placed labels and doesn't cover points too closely.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # precompute point positions in display (pixel) coords for point-avoid checks
    pts_disp = ax.transData.transform(np.c_[xs, ys])

    placed_bboxes = []
    texts = []

    for x, y, lab, color, pdisp in zip(xs, ys, labels, colors, pts_disp):
        txt = None
        for dx, dy in offsets_pts:
            t = ax.annotate(
                str(lab),
                xy=(x, y),
                xytext=(dx, dy),
                textcoords='offset points',
                ha='left' if dx >= 0 else 'right',
                va='bottom' if dy >= 0 else 'top',
                fontsize=fontsize,
                color=color,
            )
            bb = t.get_window_extent(renderer=renderer)
            ax_bb = ax.get_window_extent(renderer=renderer)

            if not _bbox_inside(bb, ax_bb, pad=1):
                t.remove()
                continue

            # avoid other labels
            if any(_bboxes_overlap(bb, obb, pad=label_pad_px) for obb in placed_bboxes):
                t.remove()
                continue

            # avoid sitting on top of any point (including other points)
            # (simple: ensure bbox isn't too close to any point display coord)
            cx0, cy0, cx1, cy1 = bb.x0, bb.y0, bb.x1, bb.y1
            inside = ((pts_disp[:,0] >= cx0 - point_avoid_px) & (pts_disp[:,0] <= cx1 + point_avoid_px) &
                      (pts_disp[:,1] >= cy0 - point_avoid_px) & (pts_disp[:,1] <= cy1 + point_avoid_px))
            # allow the label to be near its own point; disallow covering any other point
            if inside.any():
                # if it's only the current point, it's fine
                idxs = np.where(inside)[0]
                # current point index isn't directly known here; compare by pixel equality tolerance
                if not (len(idxs) == 1 and np.allclose(pts_disp[idxs[0]], pdisp, atol=1e-6)):
                    t.remove()
                    continue

            txt = t
            placed_bboxes.append(bb)
            texts.append(txt)
            break

        # fallback: keep last tried (or place with a larger offset)
        if txt is None:
            txt = ax.annotate(str(lab), xy=(x, y), xytext=(12, 12),
                              textcoords='offset points', fontsize=fontsize, 
                              color=color)
            placed_bboxes.append(txt.get_window_extent(renderer=renderer))
            texts.append(txt)

    return texts

def grouped_legend(ax, legend_groups, loc=None, title=None):
    handles, labels = [], []
    for typ, styles in legend_groups.items():
        styles_unique = list(dict.fromkeys(styles))  # unique (color, marker) in insertion order
        sub_handles = [
            Line2D([0], [0], marker=m, linestyle='None', color=c, markersize=7)
            for c, m in styles_unique
        ]
        handles.append(tuple(sub_handles))
        labels.append(typ)

    ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=-0.9)},
        title=title or "type",
        loc=loc
    )
    
TEST_COLUMN, TAG_COLUMN = 'latest__test_forces_MAE', 'latest_tag'
TEST_COLUMN, TAG_COLUMN = 'best_manual__val_forces_MAE', 'best_manual_tag'
TEST_COLUMN, TAG_COLUMN = 'best_manual__test_forces_MAE', 'best_manual_tag'

def scatter_plot(
    scatter_df,
    legend_loc=None,
    legend_title=None,
    x='cost',
    y=TEST_COLUMN,
    ):
    fig, ax = plt.subplots()
    
    _label_map = lambda k: {
        'cost': 'Avg. Iterations to reach a 0.01 residual',
        TEST_COLUMN: 'F-MAE',
    }.get(k, k)

    xs, ys, labs, colors = [], [], [], []
    legend_groups_raw = defaultdict(list)

    for _, row in scatter_df.iterrows():
        xx = row[x]
        yy = row[y]
        c = row['color']
        m = row.get('marker', 'o')

        ax.scatter(xx, yy, color=c, marker=m)

        typ, *other = row['label'].split(':')
        strength = other[0] if other else typ

        try:
            sort_key = float(strength)
        except (ValueError, TypeError):
            sort_key = float('-inf')

        xs.append(xx); ys.append(yy); labs.append(strength); colors.append(c)
        legend_groups_raw[typ].append((sort_key, c, m))

    legend_groups = {}
    for typ in sorted(legend_groups_raw):
        items = sorted(legend_groups_raw[typ], key=lambda it: it[0])
        seen = []
        for _, c, m in items:
            if (c, m) not in seen:
                seen.append((c, m))
        legend_groups[typ] = seen
    grouped_legend(ax, legend_groups, loc=legend_loc, title=legend_title)


    # draw once so we have a renderer for bbox calculations
    fig.canvas.draw()
    place_labels_no_overlap(ax, xs, ys, labs, colors, fontsize=8)

    ax.set_xlabel(_label_map(x))
    ax.set_ylabel(_label_map(y))
    
    return fig, ax


def create_scatter_df(runs_df, mean_cost_store, get_colour, get_label):
    rows = []
    for (mt, deg), mean_cost in mean_cost_store.items():
        row = runs_df.query(f'{TAG_COLUMN} == @mt').iloc[0].copy()
        row['deg'] = deg
        row['cost'] = mean_cost
        color = get_colour(model_tag=mt, warmstart_degree=deg, runs_df=runs_df)
        label = get_label(model_tag=mt, warmstart_degree=deg, runs_df=runs_df)
        row['color'] = color
        row['label'] = label.split(' ')[0]
        rows.append(row)

    return pd.DataFrame(rows)


## helpers

import numpy as np


def get_ws_label(warmstart_degree, brackets=True, warmstart_str=" warmstart"):
    if warmstart_degree < 0:
        if brackets:
            return ""
        return "No" + warmstart_str
    
    if warmstart_degree == 0:    
        l = f"const."
    elif warmstart_degree == 1:    
        l = f"linear"
    else:    
        l = f"deg={warmstart_degree}"
    
    if brackets:
        return f" ({l})"
    return l + warmstart_str

def get_linestyle(warmstart_degree):
    try:
        linestyle = {
            -1: '-.',
            0: ':',
            1: '-',
            2: '--',
            3: (0, (3, 1, 1, 1)),
            4: (0, (5, 1)),
        }[warmstart_degree]
    except Exception as e:
        linestyle = '-'
    return linestyle

def get_colour_by_degree(warmstart_degree, all_degrees=[-1,0,1,2,3,4,5], model_tag=None, runs_df=None):
    
    unique_vals = sorted(all_degrees)
    if warmstart_degree in unique_vals:
        return plt.cm.viridis(unique_vals.index(warmstart_degree) / (len(unique_vals)-.999)) # avoid div by zero
    
    print('Unknown degree:', warmstart_degree)
    return 'black'

def get_label_by_degree(model_tag, warmstart_degree, runs_df, **kwargs):
    label = get_ws_label(warmstart_degree, brackets=False, **kwargs) or 'No warmstart'
        
    # add fmae info
    try:
        row = runs_df.query(f"{TAG_COLUMN}.str.contains({model_tag.split('/')[-1].split(':')[0]!r})")
        if len(row)==0:
            print(model_tag)
        if len(row)>1:
            print(row)
        fmae = row[TEST_COLUMN].item() 
        label += f' |%%| {fmae:>2.2f}'
    except Exception as e:
        print(e)
        print()
        label += f' |%%| fmae not found'
    
    return label



def _extract_stats(runs_df, model_tag):
    
    row = runs_df.query(f"{TAG_COLUMN}.str.contains({model_tag.split('/')[-1].split(':')[0]!r})")
    if len(row)!=1:
        print(model_tag)
        print(row)
        assert 0, f'too many hits: {model_tag}'

    f, t, j = row['fpc_strength'].item(), row['trunc_strength'].item(), row['jac_reg_strength'].item()
    assert sum([np.isnan(f), np.isnan(t), np.isnan(j)]) >= 2, f"More than one of fpc, trunc, jac_reg is non-zero for {model_tag}"
    
    if f>0:
        fg = row['fpc_gamma'].item()
    else:
        fg = np.nan
    
    return (f, fg), t, j, row[TEST_COLUMN].item() 
   

def _get_colour_dict(runs_df,
    columns = ['jac_reg_strength', 'trunc_strength', 'fpc_strength', 'fpc_strength_0.4'],
    cmaps = ['Greens', 'Reds', 'Blues', 'Purples'],
    ):
    colour_dicts = {}
    for col, colour_map in zip(columns, cmaps):
        if col == 'fpc_strength_0.4':
            mask = runs_df['fpc_gamma'] == 0.4
            source = runs_df.loc[mask, 'fpc_strength']
        elif col == 'fpc_strength':
            mask = runs_df['fpc_gamma'] != 0.4
            source = runs_df.loc[mask, 'fpc_strength']
        else:
            source = runs_df[col]
        unique_vals = sorted(source.dropna().unique())
        colours = [tuple(c) for c in plt.get_cmap(colour_map)(np.linspace(0.3, 0.9, len(unique_vals)))]
        colour_dicts[col] = dict(zip(unique_vals, colours))
    return colour_dicts
     

def get_colour_by_regularisation(model_tag, runs_df, warmstart_degree=None):
    
    (f, fg), t, j, _ = _extract_stats(runs_df, model_tag)
    colour_dicts = _get_colour_dict(runs_df)

    if j > 0:
        return colour_dicts["jac_reg_strength"][j]
    if f > 0:
        lbl = "fpc_strength"
        if fg==0.4:
            lbl += '_0.4'
        return colour_dicts[lbl][f]
    if t > 0:
        return colour_dicts["trunc_strength"][t]
    return 'black'


def get_label_by_regularisation(model_tag, warmstart_degree, runs_df, add_performance=True):
    
    (f, fg), t, j, fmae = _extract_stats(runs_df, model_tag)
    
    def format_sci(val):
        return f"{val:.0e}"

    def format_j(val):
        s = f"{val:.3f}"
        zeros = len(s) - len(s.rstrip("0"))
        s_trim = s.rstrip("0")
        if s_trim.endswith("."):
            s_trim = s_trim[:-1]
        if "." not in s_trim:
            s_trim += "."
        return s_trim + (" " * zeros)

    if j > 0:
        label = f"jac:{format_j(j).rjust(5)}"
    elif f > 0:
        lbl = 'fpc'
        if fg==.4:
            lbl += '(0.4)'
        label = f"{lbl}:{format_sci(f).rjust(5)}"
    elif t > 0:
        label = f"trunc:{format_sci(t).rjust(5)}"
    else:
        label = "No Reg.:" + ' '*5
       
    if add_performance:
        # add fmae performance, placeholder
        label += f' |%%| {fmae:>2.2f}'
        
    # denote which warmstart
    label += get_ws_label(warmstart_degree)
        
    return label