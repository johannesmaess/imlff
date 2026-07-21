import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from imlff.data.info import molecule_to_n_atoms


# ── helpers ──────────────────────────────────────────────────────────────────

def upscale_deq_costs(
    df,
    deq_type="deq_tol",
    cost_col="cost_mean",
    acc_col="F_mae",
    group_cols=("molecule_name", "Rep", "n_z"),
    bins_per_unit=50,
):
    """
    Discretise DEQ cost_mean values onto a fixed grid, then upscale across
    molecules.

    Grid: bins_per_unit bins per unit, aligned so integers are bin upper
    bounds (bin_width = 1/bins_per_unit, upper bounds at k/bins_per_unit).

    Within each bin the representative cost_mean = min in the bin (leftmost
    actual value), and the representative acc = min in the bin (best).

    After binning, for each (Rep, n_z) group, forward-fill missing bins
    for molecules that don't cover the full union of bin cost_means.
    """
    bin_width = 1.0 / bins_per_unit
    all_group = [c for c in group_cols if c in df.columns]
    cross_group = [c for c in all_group if c != "molecule_name"]

    deq_mask = df["type"] == deq_type
    deq_df = df[deq_mask].copy()
    non_deq_df = df[~deq_mask]

    if deq_df.empty:
        return df

    # Assign each row to its bin (upper bound = (bin_idx+1)*bin_width)
    deq_df["_bin"] = np.floor(deq_df[cost_col] / bin_width).astype(int)

    # Within each (group + bin): representative cost_mean = min, acc = min
    agg_keys = all_group + ["_bin"]
    other_cols = [
        c for c in deq_df.columns if c not in agg_keys + [cost_col, acc_col]
    ]
    deq_binned = (
        deq_df.groupby(agg_keys, observed=True, as_index=False)
        .agg(**{
            cost_col: (cost_col, "min"),
            acc_col:  (acc_col,  "min"),
            **{c: (c, "first") for c in other_cols},
        })
        .drop(columns=["_bin"])
    )

    # Forward-fill missing bins across molecules within each cross_group
    def _augment_group(gdf):
        all_costs = sorted(gdf[cost_col].dropna().unique())
        parts = []
        for _, mol_df in gdf.groupby("molecule_name", observed=True):
            existing = set(mol_df[cost_col].dropna().values)
            missing = [c for c in all_costs if c not in existing]
            if missing:
                mol_sorted = mol_df.sort_values(cost_col).reset_index(drop=True)
                new_rows = []
                for cost in missing:
                    prev = mol_sorted[mol_sorted[cost_col] <= cost]
                    if prev.empty:
                        continue
                    row = prev.iloc[-1].copy()
                    row[cost_col] = cost
                    new_rows.append(row)
                if new_rows:
                    mol_df = pd.concat(
                        [mol_df, pd.DataFrame(new_rows)], ignore_index=True
                    )
            parts.append(mol_df)
        return pd.concat(parts, ignore_index=True) if parts else gdf

    if cross_group:
        augmented = deq_binned.groupby(
            cross_group, observed=True, group_keys=False
        ).apply(_augment_group)
    else:
        augmented = _augment_group(deq_binned)

    return pd.concat([non_deq_df, augmented], ignore_index=True)

def _ylabel_from_normalize(normalize):
    if normalize == "relative":
        return r"$(F_\mathrm{exp} - F_\mathrm{implicit})\;/\;F_\mathrm{implicit}$"
    if normalize == "relative_symmetric":
        return r"$\dfrac{F_\mathrm{exp} - F_\mathrm{implicit}}{\min(F_\mathrm{exp}, F_\mathrm{implicit})}$"
        return r"$(F_\mathrm{exp} - F_\mathrm{implicit})\;/\;min(F_\mathrm{exp}, F_\mathrm{implicit})$"
    return r"$\Delta F_\mathrm{mae}$  (explicit − implicit)"


def _signed_percent_formatter(normalize):
    if "relative" in normalize:
        return mticker.FuncFormatter(lambda x, _: f"{x:+.0%}")
    return None


# ── 1. Compute delta curves ─────────────────────────────────────────────────

def compute_delta_curves(
    df,
    explicit_types=("explicit", "explicit-tied"),
    deq_type="deq_tol",
    cost_col="cost_mean",
    acc_col="F_mae",
    group_cols=("molecule_name", "Rep", "n_z"),
    extra_cols=(),
    normalize="delta",
    eps=1e-8,
    include_dominated=True,
    cost_pad_before=0.1,
    verbose=False,
    cut_at_exp_max_cost=True,
):
    """
    Per group compute  Δ = F_exp − F_deq  on a common cost grid.
    Positive Δ  ⟹  explicit is worse (deq wins).

    include_dominated: if True, keep dominated explicit costs in the grid
                       (they don't affect the delta, just provide bin edges).
    cost_pad_before:   if > 0, add an extra evaluation point this distance
                       before each explicit cost sample. Useful for capturing
                       the step just before explicit improves.
                       These points are marked _is_sampled=False.
    """
    all_group = [c for c in group_cols if c in df.columns]

    records = []

    for group_key, gdf in df.groupby(all_group):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        gdict = dict(zip(all_group, group_key))

        deq = (
            gdf.loc[gdf["type"] == deq_type, [cost_col, acc_col]]
            .sort_values(cost_col)
            .drop_duplicates(subset=[cost_col], keep="last")
            .reset_index(drop=True)
        )
        if deq.empty:
            if verbose:
                print(f"  SKIP {gdict}: no rows with type='{deq_type}'")
                avail = gdf["type"].unique().tolist()
                print(f"    available types: {avail}")
            continue

        for exp_type in explicit_types:
            exp = (
                gdf.loc[gdf["type"] == exp_type, [cost_col, acc_col]]
                .sort_values(cost_col)
                .reset_index(drop=True)
            )
            if exp.empty:
                if verbose:
                    print(f"  SKIP {gdict}: no rows with type='{exp_type}'")
                continue

            # keep raw costs before cummin
            exp_raw_costs = exp[cost_col].values.copy()

            # cummin envelope
            exp = exp.sort_values(cost_col)
            exp[acc_col] = exp[acc_col].cummin()
            exp = exp.drop_duplicates(subset=[acc_col], keep="first")

            # common cost grid
            cost_lo = max(deq[cost_col].min(), exp[cost_col].min())
            cost_hi = exp_raw_costs.max() if cut_at_exp_max_cost else np.inf
            cut_mask = deq[cost_col].values > cost_hi
            if any(cut_mask):
                cut_costs = deq[cost_col].values[cut_mask]
                cut_acc = deq[acc_col].values[cut_mask]
                present_acc = deq[acc_col].values[~cut_mask]
                print(f'Warning: Not showing some implicit configurations, bc we have no explicit model above {cost_hi} layers ({group_key}): {cut_costs.min():.2f} - {cut_costs.max():.2f}, improving {acc_col} from {present_acc.min():.4f} to {cut_acc.min():.4f}')

            # sampled points: deq + envelope
            sampled = np.union1d(deq[cost_col].values, exp[cost_col].values)
            if include_dominated:
                sampled = np.union1d(sampled, exp_raw_costs)

            # padded points just before each explicit cost
            padded = np.array([])
            if cost_pad_before > 0:
                padded = exp_raw_costs - cost_pad_before
                padded = padded[padded >= cost_lo]

            all_costs = np.sort(np.union1d(sampled, padded))
            all_costs = all_costs[(all_costs >= cost_lo) & (all_costs <= cost_hi)]
            if len(all_costs) == 0:
                if verbose:
                    print(f"  SKIP {gdict} vs {exp_type}: no cost overlap")
                continue

            grid = pd.DataFrame({cost_col: all_costs})

            # mark which points are real samples vs padded
            sampled_set = set(np.round(sampled, decimals=10))
            grid["_is_sampled"] = grid[cost_col].apply(
                lambda x: round(x, 10) in sampled_set
            )

            deq_g = pd.merge_asof(
                grid[[cost_col]], deq.rename(columns={acc_col: "deq_acc"}),
                on=cost_col, direction="backward",
            )
            exp_g = pd.merge_asof(
                grid[[cost_col]], exp.rename(columns={acc_col: "exp_acc"}),
                on=cost_col, direction="backward",
            )

            out = grid.copy()
            out["deq_acc"] = deq_g["deq_acc"].values
            out["exp_acc"] = exp_g["exp_acc"].values

            raw = out["exp_acc"] - out["deq_acc"]
            if normalize == "relative":
                out["delta"] = raw / (out["deq_acc"].abs() + eps)
            elif normalize == "relative_symmetric":
                out["delta"] = raw / (out[["deq_acc", "exp_acc"]].abs().min(axis=1) + eps)
            else:
                out["delta"] = raw

            out["explicit_type"] = exp_type
            for k, v in gdict.items():
                out[k] = v

            if verbose:
                n_sampled = out["_is_sampled"].sum()
                n_padded = len(out) - n_sampled
                print(f"  OK {gdict} vs {exp_type}: {len(out)} grid points "
                      f"({n_sampled} sampled, {n_padded} padded), "
                      f"delta range [{out['delta'].min():.4f}, {out['delta'].max():.4f}]")

            records.append(out)

    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)

    # join back extra_cols from original df via group_cols
    extra_cols = [c for c in extra_cols if c in df.columns and c not in result.columns]
    if extra_cols:
        lookup_cols = [c for c in group_cols if c in df.columns] + list(extra_cols)
        lookup = df[lookup_cols].drop_duplicates(subset=[c for c in group_cols if c in df.columns])
        result = result.merge(lookup, on=[c for c in group_cols if c in df.columns], how="left")

    return result


# ── 2. Bin evaluation ───────────────────────────────────────────────────────

def _evaluate_at_bins(ss, cost_col, edges, model_id_cols, extra_value_cols=()):
    """
    For every model × every bin, pick costliest sample with cost ≤ bin
    upper edge.  Every model contributes to every reachable bin.

    Also tracks whether the bin's upper edge is a real sampled point.
    extra_value_cols: additional columns from ss to carry through (e.g.
                      "deq_acc", "exp_acc" for aggregate_then_delta mode).
    """
    bin_grid = pd.DataFrame({
        "_upper": edges,
        "_bin_plot_at": edges,
        "_edge_is_sampled": [True] * len(edges),  # default, overwritten below
    })

    # check which edges are sampled vs padded
    if "_is_sampled" in ss.columns:
        sampled_costs = set(np.round(
            ss.loc[ss["_is_sampled"], cost_col].values, decimals=10
        ))
        bin_grid["_edge_is_sampled"] = bin_grid["_upper"].apply(
            lambda x: round(x, 10) in sampled_costs
        )

    value_cols = ["delta"] + [c for c in extra_value_cols if c in ss.columns]
    parts = []
    for keys, mdf in ss.groupby(model_id_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mdf = mdf.sort_values(cost_col).reset_index(drop=True)

        merged = pd.merge_asof(
            bin_grid[["_upper"]].rename(columns={"_upper": cost_col}),
            mdf[[cost_col] + value_cols],
            on=cost_col, direction="backward",
        )
        merged = merged.dropna(subset=["delta"])
        merged["_bin_plot_at"] = bin_grid.loc[merged.index, "_bin_plot_at"].values
        merged["_edge_is_sampled"] = bin_grid.loc[merged.index, "_edge_is_sampled"].values
        for col_name, val in zip(model_id_cols, keys):
            merged[col_name] = val
        parts.append(merged[["_bin_plot_at", "_edge_is_sampled"] + value_cols + list(model_id_cols)])

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── 3. Drawing helper ───────────────────────────────────────────────────────

def _draw_binned(
    ax, ss, cost_col, band, color, ls, label,
    band_alpha, plot_individual, individual_alpha,
    individual_group_col, model_id_cols, verbose=False,
    show_median=True,
    aggregate_then_delta=False, normalize="delta", eps=1e-8,
):
    cmin, cmax = ss[cost_col].min(), ss[cost_col].max()
    span = cmax - cmin
    if span <= 0:
        return

    # bin edges at actual sampled costs
    all_costs = np.sort(ss[cost_col].unique())

    if verbose:
        print(f"  _draw_binned: {len(all_costs)} unique costs in [{cmin:.2f}, {cmax:.2f}], "
              f"{len(all_costs)} costs in [{all_costs[0]:.2f}, {all_costs[-1]:.2f}]")

    if len(all_costs) < 2:
        return

    id_cols = [c for c in model_id_cols if c in ss.columns]
    if not id_cols:
        id_cols = [individual_group_col]

    extra_cols = ("deq_acc", "exp_acc") if aggregate_then_delta else ()
    binned = _evaluate_at_bins(ss, cost_col, all_costs, id_cols, extra_value_cols=extra_cols)
    if binned.empty:
        return

    if verbose:
        print(f'{binned[["molecule_name", "_bin_plot_at"]][binned._bin_plot_at>4]=}')
        print(binned[binned._bin_plot_at>4])

    grp = binned.groupby("_bin_plot_at", observed=True)

    if aggregate_then_delta:
        # Central: aggregate deq_acc and exp_acc first, then normalize
        agg_fn = "median" if show_median else "mean"
        agg_deq = grp["deq_acc"].agg(agg_fn)
        agg_exp = grp["exp_acc"].agg(agg_fn)
        raw_abs = agg_exp - agg_deq
        if normalize == "relative":
            raw = raw_abs / (agg_deq.abs() + eps)
        elif normalize == "relative_symmetric":
            raw = raw_abs / (pd.concat([agg_deq.abs(), agg_exp.abs()], axis=1).min(axis=1) + eps)
        else:
            raw = raw_abs

        # Band: IQR/CI of per-molecule delta values.
        # delta is already in the same units as raw (normalized by compute_delta_curves),
        # so shift aligns the band center with raw without any unit mismatch.
        dgrp = grp["delta"]
        if show_median:
            delta_med = dgrp.median()
            shift = raw - delta_med
            lo = dgrp.quantile(band[0] / 100) + shift
            hi = dgrp.quantile(band[1] / 100) + shift
        else:
            sem = dgrp.sem()
            lo = raw - 1.96 * sem
            hi = raw + 1.96 * sem

        med = raw
    else:
        dgrp = grp["delta"]
        if show_median:
            med = dgrp.median()
            lo = dgrp.quantile(band[0] / 100)
            hi = dgrp.quantile(band[1] / 100)
        else:
            # mean ± confidence interval (95% CI via 1.96 * SEM)
            med = dgrp.mean()
            sem = dgrp.sem()
            z   = 1.96
            lo  = med - z * sem
            hi  = med + z * sem

    valid = med.notna()
    c  = med.index[valid].values
    m  = med.values[valid.values]
    ql = lo.values[valid.values]
    qh = hi.values[valid.values]
    if len(c) == 0:
        return

    # line without markers (covers all bins including padded)
    ax.plot(c, m, color=color, ls=ls, lw=1.5, label=label)
    ax.fill_between(c, ql, qh, alpha=band_alpha, color=color)

    # scatter markers only on bins whose edge is a real sampled cost
    sampled_mask = c % 1 == 0

    if sampled_mask.any():
        ax.scatter(c[sampled_mask], m[sampled_mask],
                   color=color, s=10, zorder=5)

    if plot_individual and individual_group_col in binned.columns:
        for _, mdf in binned.groupby(individual_group_col, observed=True):
            g = mdf.groupby("_bin_plot_at", observed=True)["delta"].median()
            v = g.notna()
            if v.any():
                ax.plot(
                    g.index[v].values, g.values[v.values],
                    color=color, ls=ls, alpha=individual_alpha, lw=0.6,
                )


# ── 4. Facet logic helpers ──────────────────────────────────────────────────

def _resolve_row_col(deltas, row, col, col_order):
    """
    Build compound _row key if row is a list, resolve orderings.
    Returns (facet_row_key, r_order, c_order, deltas).
    """
    # ── columns ──
    if col_order is None and col == "Rep":
        preferred = ["PaiNN", "SO3net", "SchNet"]
        present = list(deltas[col].dropna().unique())
        c_order = [r for r in preferred if r in present]
        c_order += sorted(set(present) - set(c_order))
    elif col_order is not None:
        c_order = col_order
    elif col is not None:
        c_order = sorted(deltas[col].dropna().unique(), key=str)
    else:
        c_order = [None]

    # ── rows ──
    if row is None:
        return None, [None], c_order, deltas

    if isinstance(row, (list, tuple)):
        # compound row label
        parts = list(row)
        deltas = deltas.copy()
        deltas["_row"] = deltas[parts[0]].astype(str)
        for p in parts[1:]:
            deltas["_row"] = deltas["_row"] + "  |  " + p + "=" + deltas[p].astype(str)
        facet_row = "_row"

        # order: by n_atoms (if molecule_name involved), then other dims
        if "n_atoms" in deltas.columns and "molecule_name" in parts:
            mol_order = (
                deltas.drop_duplicates("molecule_name")
                .sort_values("n_atoms")["molecule_name"].tolist()
            )
            def _sort_key(s):
                mol = s.split("  |")[0].strip()
                return (mol_order.index(mol) if mol in mol_order else 999, s)
            r_order = sorted(deltas[facet_row].dropna().unique(), key=_sort_key)
        else:
            r_order = sorted(deltas[facet_row].dropna().unique(), key=str)
    else:
        facet_row = row
        if row == "n_z":
            r_order = sorted(deltas["n_z"].dropna().unique())
        elif "n_atoms" in deltas.columns and row == "molecule_name":
            r_order = (
                deltas.drop_duplicates("molecule_name")
                .sort_values("n_atoms")["molecule_name"].tolist()
            )
        else:
            r_order = sorted(deltas[facet_row].dropna().unique(), key=str)

    return facet_row, r_order, c_order, deltas


# ── 5. Main ─────────────────────────────────────────────────────────────────

def plot_cost_delta(
    df,
    # ── data ──
    explicit_types=("explicit", "explicit-tied"),
    deq_type="deq_tol",
    cost_col="cost_mean",
    acc_col="F_mae",
    group_cols=("molecule_name", "Rep", "n_z"),
    # ── normalization ──
    normalize="delta",              # "delta" | "relative" | "relative_symmetric"
    eps=1e-8,
    # ── grid control ──
    include_dominated=True,
    cost_pad_before=0.1,
    bins_per_unit=50,
    # ── aggregation ──
    aggregate_over=None,            # None | "dataset" | "n_z" | "all"
    band=(25, 75),
    model_id_cols=("molecule_name", "n_z"),
    # ── faceting (fully independent of aggregate_over) ──
    col="Rep",
    col_order=None,
    row=None,            # str, list of str, or None
    row_order=None,                 # explicit order; None = auto
    split=None,                     # one figure per value
    # ── cosmetics ──
    figsize_per_ax=(4, 3),
    ylim=None,
    xpad=0.,
    hline0=True,
    hline0_kwargs=None,
    band_alpha=0.2,
    individual_alpha=0.25,
    plot_individual=False,
    legend=True,                    # True | False | "in ax"
    xlabel_desc='Layer calls',
    ylabel_desc=None,
    series_styles=None,
    title_pos=None,
    show_median=True,
    aggregate_then_delta=False,
    nz_in_label=False,
    sharey=True,
    verbose=False,
):
    """
    Plot  ΔF_mae = explicit − deq  vs cost.
    Positive ⟹ explicit is worse (deq wins).

    aggregate_over: what to pool for median/band
      None      → per-curve step plots (no binning)
      "dataset" → pool molecules
      "n_z"     → pool n_z values
      "all"     → pool both

    row, col, split: control faceting independently.
      row can be a list → compound row labels.
      Validation: cannot facet by a dimension you pooled.
      
      
      ++++++++
      Note on confidence bands
      ++++++++
      
      compute_delta_curves stores in binned["delta"]:

        normalize="relative" → delta_i = (exp_i − deq_i) / deq_i per molecule (already relative)
        deq_acc, exp_acc → always raw absolute values
        aggregate_then_delta=False

        Median:


        med = median( (exp_i − deq_i) / deq_i )   ← denominator varies per molecule
        lo/hi = q25/q75 of same distribution       ← band centered at med ✓
        Mean:


        med = mean( (exp_i − deq_i) / deq_i )     ← denominator varies per molecule
        lo/hi = med ± 1.96 · SEM(delta_i)         ← centered at med ✓
        aggregate_then_delta=True

        Median:


        raw = (median(exp_i) − median(deq_i)) / median(deq_i)   ← single shared denominator

        delta_med = median( (exp_i − deq_i) / deq_i )           ← per-molecule denominator
        shift = raw − delta_med
        lo/hi = q25/q75( delta_i ) + shift                       ← IQR width preserved, centered at raw ✓
        Mean:


        raw = (mean(exp_i) − mean(deq_i)) / mean(deq_i)         ← single shared denominator
            = mean(exp_i − deq_i) / mean(deq_i)                 ← NOT equal to mean(delta_i) !
                                                                (different denominators)
        sem = SEM( (exp_i − deq_i) / deq_i )                    ← per-molecule denominator
        lo/hi = raw ± 1.96 · sem                                 ← centered at raw ✓
        Key difference between the two modes for relative:

        aggregate_then_delta=False normalises before aggregating — each molecule contributes (exp_i − deq_i) / deq_i, then you take the median/mean of those ratios.

        aggregate_then_delta=True aggregates before normalising — the center uses median(deq_i) or mean(deq_i) as a single shared denominator, then the band (IQR/CI from per-molecule ratios) is shifted to that center.

        For the mean case specifically, raw = mean(exp_i − deq_i) / mean(deq_i) whereas the band's SEM comes from (exp_i − deq_i) / deq_i — the denominators differ. This is intentional: the center is "relative difference of the typical molecule", but the band width still reflects cross-molecule spread. Consistent only in the sense that both are dimensionless ratios.

    """
    if hline0_kwargs is None:
        hline0_kwargs = dict(color="turquoise", ls="--", lw=3.5)
    if series_styles is None:
        series_styles = [
            dict(type="explicit",      linestyle="-",  label="Explicit", color='blue'),
            dict(type="explicit-tied", linestyle="-.", label="Explicit (tied)", color='orange'),
        ]

    # ── validate: can't facet by pooled dimension ───────────────────────────
    pool_molecules = aggregate_over in ("dataset", "all")
    pool_nz        = aggregate_over in ("n_z", "all")

    facet_dims = set()
    for x in ([row] if isinstance(row, str) else (row or [])):
        facet_dims.add(x)
    if col:
        facet_dims.add(col)
    if split:
        facet_dims.add(split)

    if pool_molecules and "molecule_name" in facet_dims:
        raise ValueError(
            f"Cannot facet by 'molecule_name' when aggregate_over='{aggregate_over}' "
            f"(molecules are pooled). Use split= or remove from row=."
        )
    if pool_nz and "n_z" in facet_dims:
        raise ValueError(
            f"Cannot facet by 'n_z' when aggregate_over='{aggregate_over}' "
            f"(n_z is pooled). Use split= or remove from row=."
        )

    # ── gather extra_cols: everything used for faceting/splitting ────────────
    extra = set()
    if split and split not in group_cols:
        extra.add(split)
    for x in ([row] if isinstance(row, str) else (row or [])):
        if x not in group_cols:
            extra.add(x)
    if col and col not in group_cols:
        extra.add(col)
    extra.add('Dataset')
    extra_cols = [c for c in extra if c in df.columns]

    # ── upscale DEQ cost grid across molecules ───────────────────────────────
    df = upscale_deq_costs(
        df,
        deq_type=deq_type,
        cost_col=cost_col,
        acc_col=acc_col,
        group_cols=group_cols,
        bins_per_unit=bins_per_unit,
    )

    # ── compute deltas ──────────────────────────────────────────────────────
    if verbose:
        print(f"group_cols={list(group_cols)}, extra_cols={extra_cols}")
        print(f"df types: {df['type'].unique().tolist()}")
        print(f"deq_type='{deq_type}', explicit_types={explicit_types}")
        print(f"df shape: {df.shape}")
    deltas = compute_delta_curves(
        df,
        explicit_types=explicit_types,
        deq_type=deq_type,
        cost_col=cost_col,
        acc_col=acc_col,
        group_cols=group_cols,
        extra_cols=extra_cols,
        normalize=normalize,
        eps=eps,
        include_dominated=include_dominated,
        cost_pad_before=cost_pad_before,
        verbose=verbose,
    )
    if deltas.empty:
        print("No delta data produced.")
        if not verbose:
            print("  Hint: rerun with verbose=True to see per-group diagnostics.")
        return None, None

    # propagate n_atoms for ordering
    if "n_atoms" not in deltas.columns:
        if "n_atoms" in df.columns:
            mapping = df.drop_duplicates("molecule_name")[["molecule_name", "n_atoms"]]
            deltas = deltas.merge(mapping, on="molecule_name", how="left")
        elif "molecule_name" in deltas.columns:
            def _safe_n_atoms(m):
                try:
                    return molecule_to_n_atoms(m)
                except (KeyError, TypeError):
                    return None
            deltas["n_atoms"] = deltas["molecule_name"].map(_safe_n_atoms)

    # ── hue: n_z as colour when it's not pooled and not a facet row ─────────
    row_dims = set([row] if isinstance(row, str) else (row or []))
    use_hue = (not pool_nz) and ("n_z" not in row_dims) and (row != "n_z")

    # ── split handling ──────────────────────────────────────────────────────
    split_vals = (
        [None] if split is None
        else sorted(deltas[split].dropna().unique())
    )

    # ── resolve row/col ordering ────────────────────────────────────────────
    facet_row, r_order_auto, c_order, deltas = _resolve_row_col(
        deltas, row, col, col_order,
    )
    r_order = row_order if row_order is not None else r_order_auto
    
    # ── xlims and ax widths ────────────────────────────────────────────
    cost_mean_series = df[df.type!='deq_tol'].groupby('Rep').cost_mean
    xlims = cost_mean_series.agg([min,max])
    width_ratios = cost_mean_series.agg(lambda series: math.ceil(series.max() - series.min()))

    xlims['min'] -= xpad
    xlims['max'] += xpad
    width_ratios += 2 * xpad
    
    xlims = xlims.values
    width_ratios = width_ratios.tolist()

    figs = []
    for split_val in split_vals:
        dd = deltas if split is None else deltas[deltas[split] == split_val]

        nrows, ncols = len(r_order), len(c_order)
        print(nrows, ncols)
        width = figsize_per_ax[0] * ncols
        width *= np.mean(width_ratios) / 6
        fig, axes = plt.subplots(
            nrows, ncols,
            sharex="col", sharey=sharey,
            figsize=(width, figsize_per_ax[1] * nrows),
            gridspec_kw=dict(width_ratios=width_ratios, wspace=0.1, hspace=0.05),
            squeeze=False,
        )

        if use_hue:
            nz_vals = sorted(dd["n_z"].dropna().unique())
            cmap = plt.get_cmap("tab10")
            nz_colors = {nz: cmap(k % 10) for k, nz in enumerate(nz_vals)}
        else:
            nz_vals = [None]
            nz_colors = {}

        do_aggregate = aggregate_over is not None

        # ── draw ────────────────────────────────────────────────────────────
        for i, rv in enumerate(r_order):
            for j, cv in enumerate(c_order):
                ax = axes[i, j]
                sub = dd
                if facet_row and rv is not None:
                    sub = sub[sub[facet_row] == rv]
                if col and cv is not None:
                    sub = sub[sub[col] == cv]
                if sub.empty:
                    ax.set_axis_off()
                    continue

                for style in series_styles:
                    ls = style.get("linestyle", "-")
                    cl = style.get("color")
                    series_label = style.get("label")

                    for nz_val in nz_vals:
                        ss = sub
                        
                        if 'type' in style:
                            ss = ss[ss["explicit_type"] == style['type']]
                        if 'Dataset' in style:
                            ss = ss[ss["Dataset"] == style['Dataset']]
                            
                        if use_hue and nz_val is not None:
                            ss = ss[ss["n_z"] == nz_val]
                        if ss.empty:
                            continue

                        # explicit series color wins; else hue color; else None
                        color = cl or nz_colors.get(nz_val)
                        lbl = series_label
                        if use_hue and nz_val is not None and nz_in_label:
                            nz_lbl = f"n_z={nz_val}"
                            if lbl: 
                                lbl = f'{lbl} | {nz_lbl}'
                            else:
                                lbl = nz_lbl

                        if do_aggregate:
                            _draw_binned(
                                ax, ss,
                                cost_col=cost_col,
                                band=band,
                                color=color, ls=ls, label=lbl,
                                band_alpha=band_alpha,
                                plot_individual=plot_individual,
                                individual_alpha=individual_alpha,
                                individual_group_col="molecule_name",
                                model_id_cols=model_id_cols,
                                show_median=show_median,
                                aggregate_then_delta=aggregate_then_delta,
                                normalize=normalize,
                                eps=eps,
                                verbose=verbose,
                            )
                        else:
                            ss_sorted = ss.sort_values(cost_col)
                            ax.step(
                                ss_sorted[cost_col], ss_sorted["delta"],
                                where="post",
                                color=color, ls=ls, lw=1.5,
                                label=lbl,
                            )

                # ── chrome ──
                if hline0:
                    ax.axhline(0, **hline0_kwargs)
                if i == 0 and col:
                    if title_pos is None:
                        ax.set_title(str(cv))
                else:
                    if title_pos is None:
                        ax.set_title(str(cv))
                    else:
                        assert isinstance(title_pos, tuple)
                        ax.text(title_pos[0], title_pos[1], str(cv),
                                transform=ax.transAxes,
                                fontsize=14, fontweight='bold',
                                ha='left', va='bottom',
                                clip_on=False, zorder=10,
                                )
                if j == 0:
                    ax.set_ylabel(ylabel_desc or _ylabel_from_normalize(normalize))
                    if facet_row and rv is not None and nrows > 1:
                        ax.text(
                            -0.25, 0.5, str(rv),
                            transform=ax.transAxes,
                            rotation=90, va="center", ha="right",
                            fontweight="bold",
                        )
                if i == nrows - 1:
                    desc = xlabel_desc if type(xlabel_desc) is str else xlabel_desc[j] 
                    ax.set_xlabel(desc or cost_col)
                    
                if type(ylim) is tuple:
                    ax.set_ylim(*ylim)

                fmt = _signed_percent_formatter(normalize)
                if fmt is not None:
                    ax.yaxis.set_major_formatter(fmt)
                    
                    
        # set maximum line (not fillbetween) as ylim
        if ylim == 'lines':
            ymax = max(
                max(line.get_ydata())
                for ax in axes.flatten()
                for line in ax.get_lines()
            ) * 1.1

            axes.flatten()[0].set_ylim(top=ymax)


        # ── legend ──
        if "in ax" == legend:
            for a in axes.flatten():
                a.legend()
        elif legend:
            handles, labels = [], []
            for a in fig.axes:
                h, l = a.get_legend_handles_labels()
                handles += h
                labels += l
            seen = set()
            uniq = [(h, l) for h, l in zip(handles, labels)
                    if not (l in seen or seen.add(l))]
            if uniq:
                on = fig
                if type(legend) is tuple: # ax index
                    on = axes[*legend]
                    
                on.legend(
                    [h for h, _ in uniq], [l for _, l in uniq],
                    # bbox_to_anchor=(.8, 1),
                    loc='upper right',
                    frameon=False,
                )

        if split is not None:
            fig.suptitle(str(split_val), y=1.02)

        # plt.tight_layout()
        fig.tight_layout()
        figs.append((fig, axes))
        
        # xticks, xlim
        for ax in axes.flatten():
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for ax_row in axes:
            for ax, xlim in zip(ax_row, xlims):
                ax.set_xlim(*xlim)
            
    if len(figs) == 1:
        return figs[0]
    return figs