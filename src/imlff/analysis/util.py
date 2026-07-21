import operator
import wandb
import numpy as np
     
# For each group of duplicate latest_tag, prefer rows with valid latest__test_forces_MAE
def dedup_by_latest_tag(df, metric_test, column_tag='latest_manual_tag'):
    if metric_test in df.columns:
        # check that all force errors are same on duplicate entries
        has_fmae = df[metric_test].notna()
        fmae_grouped = df[has_fmae].groupby(column_tag)[metric_test]
        fmae_consistency = fmae_grouped.apply(lambda x: np.ptp(x.dropna()) < 1e-6)
        assert (fmae_consistency).all(), (
            f"Conflicting FMAE values for same latest_tag:\n"
            f"{fmae_consistency[fmae_consistency > 1]}\n"
            f"{fmae_grouped.unique()[~fmae_consistency]}"
        )
        
        df = df.sort_values(metric_test, na_position="last")
    
    return df.drop_duplicates(subset=column_tag, keep="first")

def pick_best_by_val_forces(best_runs_per_group, val_metric='val_forces_MAE (better)'):
    group_cols = [
        "AtomisticRepresentationClass",
        "molecule",
        'n_z', 
        "type",
        "n_schnets",
        "shared_schnets",
        "n_interactions",
        "shared_interactions",
        "norm_o",
        "norm_s",
        "norm_c",
        "EquiNormClass",
        "InvNormClass",
        "MergedNormClass",
        "shared_filters",
    ]

    return (
        best_runs_per_group
        .loc[best_runs_per_group.groupby(group_cols, dropna=False)[val_metric].idxmin()]
        .reset_index(drop=True)
    )