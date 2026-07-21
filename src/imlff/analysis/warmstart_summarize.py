import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, field
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from imlff.common.util import proj_path
from imlff.data.info import molecule_name_typed, molecule_to_n_atoms, molecule_to_atomic_numbers, molecule_to_dataset
from imlff.md.util import warmstart_config_str
from imlff.scripts.warmstart_analysis import analyze_warmstart

from matplotlib.lines import Line2D


@dataclass
class WarmstartLoadConfig:
    # cohort selection
    runs_file_tag: str
    which_tag: str = "best_manual"
    selected_query: list = field(default_factory=list)  # joined with ' and '; comment lines out in nb
    not_run_ids: tuple = ()

    # warmstart sweep
    warmstart_configs: list = field(default_factory=lambda: [dict(z=1, g_bar=1)])
    default_forw_tol: float = 0.01
    default_back_tol: float = 0.01
    back_iter_on_g_bar: bool = True
    saving_postfix: str = ""

    # behavior
    should_collect_extra_data: bool = False
    only_mean: bool = True
    overweigh_back: float = 1.0

    @property
    def column_tag(self): return f"{self.which_tag}_tag"
    @property
    def metric_val(self): return f"{self.which_tag}__val_forces_MAE"
    @property
    def metric_test(self): return f"{self.which_tag}__test_forces_MAE"
    @property
    def df_deq_path(self):
        sub = self.runs_file_tag.rsplit("/", 1)[0]
        return proj_path(
            f"result_summary/runs/{sub}/warmstart_df_deq{self.saving_postfix}.csv",
            verbose=False,
        )
    @property
    def df_explicit_path(self):
        sub = self.runs_file_tag.rsplit("/", 1)[0]
        return proj_path(
            f"result_summary/runs/{sub}/warmstart_df_explicit{self.saving_postfix}.csv",
            verbose=False,
        )
    @property
    def df_all_path(self):
        sub = self.runs_file_tag.rsplit("/", 1)[0]
        return proj_path(
            f"result_summary/runs/{sub}/warmstart_df_all{self.saving_postfix}.csv",
            verbose=False,
        )


def load_best_runs(cfg: WarmstartLoadConfig) -> pd.DataFrame:
    df = pd.read_csv(proj_path(f"result_summary/runs/{cfg.runs_file_tag}.csv", verbose=False))
    df = df.query(f"not {cfg.column_tag}.isna()").query(f"not run_id in {list(cfg.not_run_ids)}")
    df["n_atoms"] = df.molecule.map(lambda mol: len(molecule_to_atomic_numbers[mol]))
    df["AtomisticRepresentationClass"] = pd.Categorical(
        df["AtomisticRepresentationClass"],
        categories=["SchNet", "PaiNN", "SO3net"],
        ordered=True,
    )
    df = df.sort_values(["AtomisticRepresentationClass", "n_atoms"]).reset_index(drop=True)
    return df


def select_cohort(best_runs: pd.DataFrame, cfg: WarmstartLoadConfig) -> pd.DataFrame:
    filters = [f for f in cfg.selected_query if f and f.strip()]
    if not filters:
        return best_runs.reset_index(drop=True)
    return best_runs.query(" and ".join(f"({f})" for f in filters)).reset_index(drop=True)


def load_or_build_df_deq(
    selected: pd.DataFrame,
    cfg: WarmstartLoadConfig,
    dic_arrays: dict | None = None,
    *,
    reload: bool = False,
    save=False,
):
    if dic_arrays is None or reload:
        dic_arrays = {}

    dic_arrays = populate_dic_arrays(
        dic_arrays,
        selected,
        cfg.warmstart_configs,
        default_forw_tol=cfg.default_forw_tol,
        default_back_tol=cfg.default_back_tol,
        back_iter_on_g_bar=cfg.back_iter_on_g_bar,
        should_collect_extra_data=cfg.should_collect_extra_data,
        which_tag=cfg.which_tag,
    )

    path = cfg.df_deq_path
    if save == "force" or (save and not os.path.exists(path)):
        df_deq = dic_arrays_to_df_deq(
            dic_arrays,
            selected,
            cfg.warmstart_configs,
            overweigh_back=cfg.overweigh_back,
            only_mean=cfg.only_mean,
            which_tag=cfg.which_tag,
        )
        df_deq.to_csv(path)
    elif os.path.exists(path) and not save:
        df_deq = pd.read_csv(path)
    else:
        df_deq = dic_arrays_to_df_deq(
            dic_arrays,
            selected,
            cfg.warmstart_configs,
            overweigh_back=cfg.overweigh_back,
            only_mean=cfg.only_mean,
            which_tag=cfg.which_tag,
        )

    return df_deq, dic_arrays


def prepare_explicit_cohort(selected: pd.DataFrame) -> pd.DataFrame:
    """Cohort prep mirroring `select_cohort`, but for the explicit branch.

    PaiNN's tied explicit models are stored under type ``explicit-tt``; here we
    drop any pre-existing ``explicit-tied`` PaiNN rows and relabel the
    ``explicit-tt`` ones to ``explicit-tied`` so all architectures use a single
    consistent type name downstream.
    """
    out = selected.copy()

    # Remove PaiNN explicit-tied rows (rebuilt from explicit-tt below)
    out = out[~((out.AtomisticRepresentationClass == "PaiNN") &
                (out.type == "explicit-tied"))]

    # Rename PaiNN explicit-tt to explicit-tied
    mask = (out.AtomisticRepresentationClass == "PaiNN") & (out.type == "explicit-tt")
    out.loc[mask, "type"] = "explicit-tied"
    return out


def _build_df_explicit(selected_take_tt: pd.DataFrame, cfg: WarmstartLoadConfig) -> pd.DataFrame:
    df_explicit = build_df_other_explicit(selected_take_tt, which_tag=cfg.which_tag)
    df_explicit["Dataset"] = df_explicit.molecule_name.apply(molecule_to_dataset)
    return df_explicit


def load_or_build_df_explicit(
    selected_take_tt: pd.DataFrame,
    cfg: WarmstartLoadConfig,
    *,
    reload: bool = False,
    save=False,
) -> pd.DataFrame:
    """Load `df_explicit` from disk or (re)build it, mirroring `load_or_build_df_deq`.

    save semantics (same as the DEQ loader):
      - ``"force"``                -> always rebuild and overwrite the csv
      - truthy and no csv yet      -> build and save
      - csv exists and not saving  -> read from csv
      - otherwise                  -> build without saving
    `reload=True` forces a rebuild regardless of an existing csv.
    """
    path = cfg.df_explicit_path

    if save == "force" or reload or (save and not os.path.exists(path)):
        df_explicit = _build_df_explicit(selected_take_tt, cfg)
        if save:
            df_explicit.to_csv(path)
    elif os.path.exists(path) and not save:
        df_explicit = pd.read_csv(path)
    else:
        df_explicit = _build_df_explicit(selected_take_tt, cfg)

    return df_explicit


def populate_dic_arrays_explicit(
    dic_arrays_explicit,
    selected_take_tt,
    *,
    which_tag="better",
    reload=False,
):
    """Build a `dic_arrays`-like dict for explicit models.

    Mirrors `populate_dic_arrays`, but explicit models have no
    (forw_iter, back_iter) sweep, so each value is a 1D array over samples
    instead of the (n_iter_combos, n_samples) DEQ arrays.

    key   : (molecule, Rep, model_tag)        # no warmstart tuple
    value : {'type': str, 'F_mae': (n_samples,), 'E_mae': (n_samples,)}
    """
    if dic_arrays_explicit is None or reload:
        dic_arrays_explicit = {}

    for _, row in tqdm(list(selected_take_tt.iterrows())):
        type_str = row["type"]
        if "explicit" not in type_str:
            continue

        model_tag = row[f"{which_tag}_tag"]
        if "model-" not in model_tag:
            print(f'Warning: {model_tag} does not contain "model-" ({row.molecule}) - skipping.')
            continue

        Rep = row["AtomisticRepresentationClass"]
        molecule = row["molecule"]
        key = (molecule, Rep, model_tag)

        if key in dic_arrays_explicit and not reload:
            continue

        _, default_path, _ = analyze_warmstart(
            "qcml-group/dev-v10/" + model_tag, return_paths=True,
        )

        try:
            F_mae = np.load(default_path + "F_mae_next.npy")
            E_mae = np.load(default_path + "E_mae_next.npy")
        except Exception:
            print(f'Warning: {model_tag} missing F_mae_next/E_mae_next ({row.molecule}) - skipping.')
            continue

        dic_arrays_explicit[key] = {"type": type_str, "F_mae": F_mae, "E_mae": E_mae}

    return dic_arrays_explicit


DENSITY_PLOT_TYPES = ("deq-merged", "explicit", "explicit-tied", "explicit-tt")

DENSITY_COLORS = {
    "deq": "tab:cyan",
    "deq-merged": "tab:green",
    "explicit": "tab:blue",
    "explicit-tied": "tab:orange",
    "explicit-tt": "tab:red",
}


def max_iters_grid(maxi, mini=1, bmaxi=None, bmini=None, along_diagonal=1000000):
    if bmini is None: bmini = mini
    if bmaxi is None: bmaxi = maxi
    f = range(mini, maxi+1)
    b = range(bmini, bmaxi+1)
    max_iters = list(product(f,b))
    max_iters = [(x,y) for (x,y) in max_iters if x-along_diagonal<=y<=x+along_diagonal]
    max_iters = sorted(max_iters, key=lambda x: (x[1], x[0]))
    return max_iters

def is_warm(warmstart_config):
    if type(warmstart_config) is bool:
        return warmstart_config 
    return any(v>=0 for v in warmstart_config.values())

def get_iters(warmstart_config, model_tag=None):
    if 'g51tassj' in model_tag:
        return max_iters_grid(maxi=16)
    if is_warm(warmstart_config):
        return max_iters_grid(maxi=8, bmaxi=6)
    else:
        return max_iters_grid(maxi=16)
    
def get_tols(warmstart_config, model_tag=None):
    if is_warm(warmstart_config):
        forw_tols = back_tols = np.logspace(-3, -1, 9)
        forw_tols = back_tols = np.logspace(-4, 1, 46)
    else:
        forw_tols = back_tols = np.logspace(-4, 1, 46)

    return forw_tols, back_tols
    
def populate_dic_arrays(
    dic_arrays,
    selected,
    warmstart_configs,
    *,
    default_forw_tol,
    default_back_tol,
    back_iter_on_g_bar,
    should_collect_extra_data=False,
    only_for_model_tag_substr=None,   # e.g. "g51tassj" or None
    which_tag='better',
    types=('deq','deq-merged'),                   # which model types sweep like deq, e.g. ('deq', 'deq-merged')
):

    for warmstarted in warmstart_configs:
        warmstarted_full = dict(z=-1, g=-1, g_bar=-1) | warmstarted
        collect_extra_data = bool(warmstarted) and bool(should_collect_extra_data)

        for idx, row in tqdm(list(selected.iterrows())[::-1]):
            model_tag = row[f'{which_tag}_tag']

            if only_for_model_tag_substr and only_for_model_tag_substr not in model_tag:
                continue

            max_iters = get_iters(warmstarted_full, model_tag)

            type_str = row['type']
            if type_str not in types:
                continue

            Rep = row['AtomisticRepresentationClass']
            molecule = row['molecule']
            dic_arrays_key = (molecule, Rep, model_tag, tuple(warmstarted.items()))
            
            # look at subdirectories in dir at default_path -> tells us which iters we need to have loaded.
            results_path_general, default_path_general, _ = analyze_warmstart(
                'qcml-group/dev-v10/' + model_tag,
                forw_max_iter=9876,
                back_max_iter=9876,
                default_forw_tol=default_forw_tol,
                default_back_tol=default_back_tol,
                back_iter_on_g_bar=back_iter_on_g_bar,
                return_paths=True,
                **warmstarted_full,
            )
            
            dir_beginning = warmstart_config_str(**warmstarted)
            found_max_iters = []

            pat = re.compile(r"_mi(\d+)(?:_bmi(\d+))?_iter_g_bar$")

            search_pat = Path(default_path_general if is_warm(warmstarted) else results_path_general)
            if not search_pat.exists():
                print(f"Warning: Directory does not exist: {search_pat}")
                print(f"Skipping this configuration.")
                continue  # or return, depending on your loop structure
            for p in sorted(search_pat.iterdir(), key=lambda x: x.name):
                if not p.is_dir() or not p.name.startswith(dir_beginning):
                    continue
                m = pat.search(p.name)
                if m:
                    mi = int(m.group(1))
                    bmi = mi if m.group(2) is None else int(m.group(2))
                    found_max_iters.append((mi, bmi))
            found_max_iters = sorted(found_max_iters, key=lambda x: (x[1], x[0]))

            n_usual_iters = len(max_iters)
            n_found_iters = len(found_max_iters)
            if n_usual_iters > n_found_iters:
                print(f'Less iters than usual ({n_found_iters} rather than {n_usual_iters}):', dic_arrays_key)
                print(search_pat)
            if n_usual_iters < n_found_iters:
                print(f'More iters than usual ({n_found_iters} rather than {n_usual_iters}):', dic_arrays_key)

            deq_dict_arrays = dic_arrays.get(dic_arrays_key)
            needs_load = (
                deq_dict_arrays is None
                or ('forw_iter' not in deq_dict_arrays)
                or (len(deq_dict_arrays['forw_iter']) != n_found_iters)
                # or n_usual_iters > n_found_iters # force reload of thisspecial case?
            )
            if not needs_load or n_found_iters==0:
                continue

            print('Not yet loaded. Now Loading:', dic_arrays_key)

            deq_dict_lists = defaultdict(list)
            not_found = []

            for forw, back in max_iters:
                _, default_path, warm_path = analyze_warmstart(
                    'qcml-group/dev-v10/' + model_tag,
                    forw_max_iter=forw,
                    back_max_iter=back,
                    default_forw_tol=default_forw_tol,
                    default_back_tol=default_back_tol,
                    back_iter_on_g_bar=back_iter_on_g_bar,
                    return_paths=True,
                    **warmstarted_full,
                )

                try:
                    F_mae = np.load(warm_path + 'F_mae_next.npy')
                    
                    # pri(f'{(forw+overweigh_back*back)/(1+overweigh_back):.1f} {forw:>2} {back:>2}', F_mae[:50])

                    deq_dict_lists['forw_iter'].append(forw)
                    deq_dict_lists['back_iter'].append(back)
                    deq_dict_lists['F_mae'].append(F_mae)
                    deq_dict_lists['E_mae'].append(np.load(warm_path + 'E_mae_next.npy'))

                    back_label = 'g_bar' if back_iter_on_g_bar else 'g'
                    load_calc_diffs = collect_extra_data * ['F_pred_next', 'E_pred_next']
                    load_store = [
                        'forw_res_next',
                        'back_res_next',
                    ] + collect_extra_data * [
                        'E_mae_next',
                        'z_star_diff_next_warm',
                        'g_star_diff_next_warm_per_atom',
                        'g_star_diff_next_warm_per_atom_deg',
                        'z_star_diff_next_warm',
                        'g_star_diff_next_warm_per_atom',
                        'g_star_diff_next_warm_per_atom_deg',
                        'g_bar_star_diff_next_warm',
                        'g_bar_star_diff_next_warm_per_atom',
                        'g_bar_star_diff_next_warm_per_atom_deg',
                        'z_res_next',
                        'z_res_abs_next',
                        back_label + '_res_next',
                        back_label + '_res_abs_next',
                        'forw_res_next_per_atom',
                        'back_res_next_per_atom',
                        'z_res_abs_next_per_atom_deg',
                        back_label + '_res_abs_next_per_atom_deg',
                    ]

                    for org_key in load_store:
                        short_key = org_key.replace('_next', '').replace('_warm', '')
                        try:
                            d = np.load(warm_path + org_key + '.npy')
                        except Exception as e:
                            print(f'{forw:>2} {back:>2}: {org_key}. Error in load_store.')
                            # print(e)
                            d = None
                        deq_dict_lists[short_key].append(d)

                    for org_key in load_calc_diffs:
                        try:
                            pred_converged = np.load(default_path + org_key + '.npy')
                            pred_restricted = np.load(warm_path + org_key + '.npy')
                            pred_diff = pred_restricted - pred_converged

                            mae = lambda diff, per_atom=False: np.mean(np.abs(diff), axis=tuple(range(1 + per_atom, diff.ndim)))
                            pred_diff_mae = mae(pred_diff, per_atom=False)
                            pred_diff_mae_per_atom = mae(pred_diff, per_atom=True)
                            pred_diff_mae_rel = pred_diff_mae / mae(pred_converged, per_atom=False)
                            pred_diff_mae_per_atom_rel = pred_diff_mae_per_atom / mae(pred_converged, per_atom=True)

                            short_key = org_key.replace('_next', '')
                            deq_dict_lists[short_key + '_diff_mae'].append(pred_diff_mae)
                            deq_dict_lists[short_key + '_diff_mae_per_atom'].append(pred_diff_mae_per_atom)
                            deq_dict_lists[short_key + '_diff_mae_rel'].append(pred_diff_mae_rel)
                            deq_dict_lists[short_key + '_diff_mae_per_atom_rel'].append(pred_diff_mae_per_atom_rel)
                        except Exception as e:
                            print(f'{forw:>2} {back:>2}: {org_key}. Error in load_calc_diffs.')
                            print(e)

                except Exception as e:
                    # print(e)
                    not_found.append((forw, back))

            if not_found:
                print(f'Not found for {model_tag}:', not_found)

            if not deq_dict_lists:
                print(f'{model_tag} {type_str} {row["n_z"]} {row["n_interactions"]} not found')
                continue

            deq_dict_arrays = {k: np.stack(v) for k, v in deq_dict_lists.items()}
            deq_dict_arrays['type'] = type_str
            dic_arrays[dic_arrays_key] = deq_dict_arrays

    return dic_arrays


def dic_arrays_to_df_deq(
    dic_arrays,
    selected,
    warmstart_configs,
    *,
    overweigh_back=1.,
    only_mean=True,
    get_tols = get_tols,
    which_tag = 'better',
):

    # for quick lookup of converged metrics per (molecule, Rep, tag)
    conv = {}
    for _, row in selected.iterrows():
        if 'deq' not in row['type']:
            continue
        conv[row[f'{which_tag}_tag']] = (
            row[f'{which_tag}__test_forces_MAE'],
            row[f'{which_tag}__test_energy_MAE'],
            row['n_z'],
            row['n_interactions'],
            row.Dataset,
        )

    df_list = []

    for warmstarted in warmstart_configs:
        warmstarted_full = dict(z=-1, g=-1, g_bar=-1) | warmstarted
        for (molecule, Rep, model_tag, warm_items), deq_dict_arrays in dic_arrays.items():
            if warm_items != tuple(warmstarted.items()):
                continue

            if model_tag not in conv:
                # dic_arrays might contain stuff from a different "selected" than current
                continue
            F_mae_conv, E_mae_conv, n_z, n_interactions, dataset = conv[model_tag]

            type_str = deq_dict_arrays['type']

            forw_res = deq_dict_arrays['forw_res']
            back_res = deq_dict_arrays['back_res']
            forw_iter = deq_dict_arrays['forw_iter']
            back_iter = deq_dict_arrays['back_iter']
            F_mae = deq_dict_arrays['F_mae']
            E_mae = deq_dict_arrays['E_mae']

            forw_iter_unique, forw_iter_unique_indices = np.unique(forw_iter, return_index=True)

            if 1 != (m := (np.diff(forw_res, axis=0) < 1e12).mean()):
                print(f'Warning: forw_res is decreasing in only {m*10:.6f}% of {forw_res.shape[0]} iterations')
            if 1 != (m := (np.diff(back_res, axis=0) < 1e12).mean()):
                print(f'Warning: back_res is decreasing in only {m*10:.6f}% of {back_res.shape[0]} iterations')
            assert np.all(np.diff(forw_iter_unique) >= 0),  f'{(molecule, Rep, model_tag, warm_items)} {forw_iter_unique}'
            assert np.all(np.diff(back_iter) >= 0),         f'{(molecule, Rep, model_tag, warm_items)} {back_iter}'

            forw_tols, back_tols = get_tols(warmstarted_full, model_tag)
            
            for forw_tol in forw_tols:
                forw_surpassed = forw_res[forw_iter_unique_indices] <= forw_tol
                if not forw_surpassed.any(axis=0).all():
                    continue

                forw_first_below_index = np.argmax(forw_surpassed, axis=0)
                forw_first_below_iter = forw_iter_unique[forw_first_below_index]

                forw_iter_matching_mask = forw_iter[:, None] == forw_first_below_iter[None, :]
                back_res_masked = np.where(forw_iter_matching_mask, back_res, np.inf)

                for back_tol in back_tols:
                    if warmstarted and back_tol < forw_tol:
                        continue

                    both_surpassed = back_res_masked <= back_tol
                    if not both_surpassed.any(axis=0).all():
                        continue

                    both_first_below_index = np.argmax(both_surpassed, axis=0)
                    back_first_below_iter = back_iter[both_first_below_index]

                    both_surpassed_F_mae = F_mae[both_first_below_index, np.arange(F_mae.shape[1])]
                    both_surpassed_E_mae = E_mae[both_first_below_index, np.arange(E_mae.shape[1])]

                    df_list.append({
                        'n_z': n_z,
                        'n_interactions': n_interactions,
                        'forw_tol': forw_tol,
                        'back_tol': back_tol,
                        'forw_iter_mean': forw_first_below_iter.mean(),
                        'back_iter_mean': back_first_below_iter.mean(),
                        'cost_mean': (
                            forw_first_below_iter.mean()/(1+overweigh_back)
                            + overweigh_back/(1+overweigh_back)*back_first_below_iter.mean()
                        ),
                        'F_mae_mean': both_surpassed_F_mae.mean(),
                        'forw_iter': forw_first_below_iter if not only_mean else [forw_first_below_iter.mean()],
                        'back_iter': back_first_below_iter if not only_mean else [back_first_below_iter.mean()],
                        'F_mae': both_surpassed_F_mae if not only_mean else [both_surpassed_F_mae.mean()],
                        'type': type_str,
                        'Rep': Rep,
                        'Dataset': dataset,
                        'molecule_name': molecule,
                        'model_tag': model_tag.split('/')[-1],
                        'z': warmstarted_full['z'],
                        'g': warmstarted_full['g'],
                        'g_bar': warmstarted_full['g_bar'],
                        'F_mae_converged': F_mae_conv,
                        'E_mae_converged': E_mae_conv,
                    })

    return pd.DataFrame(df_list)


def build_df_other_explicit(selected, *, only_mean=True, which_tag='better'):
    """
    Collect per-sample (or mean) F_mae_next for explicit models in `selected`
    and return a DataFrame with the requested columns.
    """
    df_list_other = []

    for _, row in tqdm(selected.iterrows()):
        model_tag = 'qcml-group/dev-v10/' + row[f'{which_tag}_tag']  # you overwrite best_tag anyway

        if 'model-' not in model_tag:
            print(f'Warning: {model_tag} does not contain "model-" ({row.molecule}) - skipping.')
            continue

        type_str = row['type']
        if 'explicit' not in type_str:
            continue

        Rep = row['AtomisticRepresentationClass']
        molecule_name = row['molecule']
        n_z = row['n_z']
        n_interactions = row['n_interactions']

        _, default_path, _ = analyze_warmstart(model_tag, return_paths=True)
        print(default_path)

        try:
            F_mae_next = np.load(default_path + 'F_mae_next.npy')
        except Exception:
            print(f'Warning: {model_tag} does not contain "F_mae_next" ({row.molecule}) - skipping.')
            continue

        if only_mean:
            F_mae_next = F_mae_next.mean()[None]

        cost_mean = n_interactions  # matches your current code

        for i in range(len(F_mae_next)):
            df_list_other.append((
                model_tag.split('/')[-1],
                type_str,
                n_z,
                n_interactions,
                cost_mean,
                i,
                F_mae_next[i],
                Rep,
                molecule_name,
            ))

    # keep (unchanged)
    if False:
        ## DEQ with maximum iterations
        for (n_z, n_interactions, forw, back), F_mae_next in deq_dict_F.items():
            if only_mean:
                F_mae_next = F_mae_next.mean()[None]
            Rep = molecule_name = 'uncategorized'
            if len(other) > 0: Rep = other[0]
            if len(other) > 1: molecule_name = other[1]
            for i in range(len(F_mae_next)):
                df_list_other.append((model_tag.split('/')[-1], 'deq_mi', n_z, n_interactions, (forw+3*back)/4, i, F_mae_next[i], Rep, molecule_name))
                
    return pd.DataFrame(
        df_list_other,
        columns=['model_tag', 'type', 'n_z', 'n_interactions', 'cost_mean', 'sample', 'F_mae', 'Rep', 'molecule_name'],
    )

def get_loosest_tols_with_tolerable_fmae(
        df_deq,
        # fmae_tolerance = 1.01,
        # fmae_tolerance = 1.05,
        fmae_tolerance = 1.02,
        # fmae_tolerance = 1.025,
        # power = 1,
        power = 'min',
    ):

    check_all_lower_tols_have_tolerable_fmae = False,

    rows_loosest_with_tolerable_fmae = []
    for (Rep, Dataset, molecule_name, model_tag, n_z, n_interactions), group in df_deq.groupby(['Rep', 'Dataset', 'molecule_name', 'model_tag', 'n_z', 'n_interactions']):
        
        # check only those rows that have tolerable fmae
        min_fmae = group.F_mae_mean.min()
        target_fmae = fmae_tolerance * min_fmae
        group = group[group.F_mae_mean < target_fmae].reset_index(drop=True)
        
        # try to find one with high tolerances
        tols = group[['forw_tol', 'back_tol']].to_numpy()
        if power == 'min':
            group['tols_weighted'] = np.min(tols, axis=1)
        else:
            group['tols_weighted'] = (tols ** power).sum(1)
        
        group = group.sort_values(['tols_weighted', 'forw_tol', 'back_tol'], ascending=False).reset_index(drop=True)
        
        
        if False:
            # show all
            for _, row in group.iterrows():
                rows_loosest_with_tolerable_fmae.append(row)

        if not check_all_lower_tols_have_tolerable_fmae:
            row = group.loc[0]
            row = row.copy()
            row.molecule_name = row.molecule_name# + ' - best'
        else:
            for _, row in group.iterrows():
                # check if all tighter tolerance pairs have a tolerable fmae, too.
                fmae_of_lower_tols = group.F_mae_mean[(group.forw_tol < row.forw_tol) & (group.back_tol < row.back_tol)]
                if (fmae_of_lower_tols < target_fmae).all():
                    break
        
        rows_loosest_with_tolerable_fmae.append(row)
    return pd.DataFrame(rows_loosest_with_tolerable_fmae)


def split_legends(ax, group, *, arch2marker, mol2color):    
    # if seaborn didn't make a legend (e.g. only `color=` was used), create ours anyway
    if ax.legend_ is not None:
        ax.legend_.remove()
    
    # present items in THIS plot
    present_arch = group["Architecture"].dropna().drop_duplicates().tolist()
    if "N Atoms" in group.columns:
        present_mols = (group[["Molecule", "N Atoms"]]
                        .drop_duplicates()
                        .sort_values("N Atoms")["Molecule"]
                        .tolist())
    else:
        present_mols = group["Molecule"].dropna().drop_duplicates().tolist()
    
    # --- Architecture legend ---
    if present_arch:
        arch_handles = [
            Line2D([0], [0],
                   marker=arch2marker[a],
                   linestyle="None",
                   markersize=8,
                   markerfacecolor="none",
                   markeredgecolor="black",
                   label=a)
            for a in present_arch
        ]
        leg_arch = ax.legend(
            handles=arch_handles,
            title="Architecture",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(present_arch),
            frameon=False,
        )
        ax.add_artist(leg_arch)
    
    # --- Molecule legend (split by Dataset) ---
    if present_mols:
        # Group molecules by dataset
        mol_dataset_map = group[["Molecule", "Dataset"]].drop_duplicates().set_index("Molecule")["Dataset"].to_dict()
        datasets = group["Dataset"].dropna().drop_duplicates().tolist()
        print(mol_dataset_map)
        # Prepare dataset legends with molecule counts
        dataset_legends = []
        for dataset in datasets:
            dataset_mols = [m for m in present_mols if mol_dataset_map.get(m) == dataset]
            if dataset_mols:
                dataset_legends.append((dataset, dataset_mols))
        
        # Sort by number of entries (descending) so larger goes to the right
        dataset_legends.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Create legends: larger on right (1.0), smaller on left (~0.7)
        for i, (dataset, dataset_mols) in enumerate(dataset_legends):
            mol_handles = [
                Line2D([0], [0],
                       marker="o",
                       linestyle="None",
                       markersize=8,
                       markerfacecolor=mol2color[m],
                       markeredgecolor=mol2color[m],
                       label=m)
                for m in dataset_mols
            ]
            
            # Larger legend (index 0) goes to upper right, smaller (index 1) to its left
            if i == 0:
                bbox_x = 1.0
                loc="upper right"
            else:
                bbox_x = 0.7
                loc="upper left"
            
            leg_mol = ax.legend(
                handles=mol_handles,
                title=dataset,
                loc=loc,
                # bbox_to_anchor=(bbox_x, 1.0),
                frameon=False,
            )
            if i < len(dataset_legends) - 1:
                ax.add_artist(leg_mol)
                
                

arch2marker = {
    "SchNet": "o",   # circle
    "PaiNN": "s",    # square (much clearer than *)
    "SO3net": "D",   # diamond
}           
                
                
def plot_scatter_tols_for_tolerable_fmae(
    df_loosest_with_tolerable_fmae,
    back_iter_on_g_bar=True,
    
    # save = True,
    save = False,

    # group_by = ['Architecture', 'Molecule'],
    # group_by = 'Architecture',
    # group_by = 'Molecule',
    group_by = 'nothing',

    # palette = sns.color_palette("tab20", 12),
    # palette = sns.color_palette("husl", 12),
    palette = sns.color_palette("tab10") + sns.color_palette("tab20")[10:12],
    marker_size='iter_mean',
    
    marker_size_legend=False,
    iter_in_molecule_legend=True,
    put_on_ax = None,
    show = True
):

    if True: # preprocess data
        cols_org = [
            'forw_tol', 
            'back_tol', 
            'F_mae_mean', 
            'Rep', 
            'molecule_name', 
            'Dataset',
            'forw_iter_mean',
            'back_iter_mean',
            ]
        cols_new = [
            'Forward tolerance',
            'Backward tolerance',
            'F-MAE',
            'Architecture',
            'Molecule',
            'Dataset',
            'forw_iter_mean',
            'back_iter_mean',
        ]

        df_loosest_with_tolerable_fmae = df_loosest_with_tolerable_fmae.copy()[cols_org]
        df_loosest_with_tolerable_fmae.columns = cols_new
        df_loosest_with_tolerable_fmae['nothing'] = True
        df_loosest_with_tolerable_fmae['N Atoms'] = df_loosest_with_tolerable_fmae['Molecule'].apply(molecule_to_n_atoms)
        df_loosest_with_tolerable_fmae['Molecule'] = df_loosest_with_tolerable_fmae['Molecule'].apply(molecule_name_typed)
        df_loosest_with_tolerable_fmae['iter_mean'] = (df_loosest_with_tolerable_fmae['forw_iter_mean'] + df_loosest_with_tolerable_fmae['back_iter_mean'])/2
        
        if iter_in_molecule_legend and group_by == 'Architecture':
            df_loosest_with_tolerable_fmae['Molecule'] += df_loosest_with_tolerable_fmae['iter_mean'].apply('  ({:.1f})'.format)
        if iter_in_molecule_legend and group_by == 'Molecule':
            df_loosest_with_tolerable_fmae['Architecture'] += df_loosest_with_tolerable_fmae['iter_mean'].apply(' ({:.1f})'.format)


        if group_by=='nothing':
            # add some noise to points ssuch that they dont overlap too much
            np.random.seed(42)
            for col in [
                'Forward tolerance',
                'Backward tolerance',
            ]:
                sigma = 0.05
                df_loosest_with_tolerable_fmae[col] = np.exp(np.log(df_loosest_with_tolerable_fmae[col]) + np.random.normal(0.0, sigma, size=len(df_loosest_with_tolerable_fmae)))


        # 1) fixed Molecule order, sorted by N Atoms
        mol_order = (
            df_loosest_with_tolerable_fmae
            .drop_duplicates("Molecule")
            .sort_values("N Atoms")["Molecule"]
            .tolist()
        )

        # 2) fixed Molecule -> color mapping (order respected)
        palette = sns.color_palette("tab20", len(mol_order))
        mol2color = dict(zip(mol_order, palette))


    if group_by!='nothing':
        group_iterator = df_loosest_with_tolerable_fmae.groupby(group_by)
    else:
        group_iterator = df_loosest_with_tolerable_fmae.iterrows()
    for group_keys_or_nothing, group in group_iterator:

        group = group.to_frame().T if isinstance(group, pd.Series) else group
        
        # --- hue kwargs with fixed mapping + local legend ---
        present_mols = (
            group[["Molecule", "N Atoms"]]
            .drop_duplicates()
            .sort_values(by="N Atoms")["Molecule"]
            .tolist()
        )
        mol_palette = {m: mol2color[m] for m in present_mols}
        if "Molecule" in group_by:
            m, = present_mols
            color_kwargs = dict(color=mol_palette[m])
        else:
            color_kwargs = dict(hue="Molecule", hue_order=present_mols, palette=mol_palette)

        # --- style kwargs with fixed mapping + local legend ---
        present_arch = (
            group["Architecture"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        if "Architecture" in group_by:
            a, = present_arch
            style_kwargs = dict(marker=arch2marker[a])
        else:
            local_arch2marker = {a: arch2marker[a.split(' (')[0]] for a in present_arch}
            style_kwargs = dict(style="Architecture", markers=local_arch2marker, style_order=present_arch)

        ax = sns.scatterplot(
            data=group.sort_values("N Atoms"),
            x="Forward tolerance",
            y="Backward tolerance",
            size=marker_size,
            ax = put_on_ax,
            **color_kwargs,
            **style_kwargs,
        )
        handles, labels = ax.get_legend_handles_labels()
        print(len(present_mols), len(present_arch), len(handles), labels)
        if not marker_size_legend:
            # drop size entries (they come last)
            n_until_sizes = len(present_mols) + len(present_arch)
            if group_by=='Architecture':
                handles = handles[:len(present_mols)]
                labels = labels[:len(present_mols)]
            if group_by=='Molecule':
                handles = handles[-len(present_arch):]
                labels = labels[-len(present_arch):]
        ax.legend(handles, labels, frameon=False)
        if group_by in ('nothing', 'Architecture'):
            split_legends(ax, group, arch2marker=arch2marker, mol2color=mol2color)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.002, 5.)
        plt.ylim(0.002, 5.)
        fmt = mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_box_aspect(1)
        lineuntil = .5 + .25 * (group_by == 'Molecule')
        ax.plot(
            [0, lineuntil], [0, lineuntil],
            transform=ax.transAxes,
            color="0.6",
            linestyle="--",
            linewidth=1.0,
        )
        if group_by != "nothing":
            ax.text(
                0.02, 0.98,
                str(group_keys_or_nothing),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=13,
                # fontweight="bold",
            )

        
        if save:
            if group_by=='nothing':
                postfix = ''
            elif type(group_by) is str:
                postfix = '__' + group_keys_or_nothing
            else:
                postfix = '__' + '_'.join(group_keys_or_nothing)
            if back_iter_on_g_bar:
                postfix +='__back_on_g_bar'
            fig_dir = '/home/johannes/imlff-paper/figures'
            pat = f'{fig_dir}/scatter__ftol_btol_for_tolerable_Fmae{postfix}.pdf'
            plt.savefig(pat, bbox_inches="tight")
            print(pat)
        
        if show:
            plt.show()
        
# put FMAE on a grid of ftol, btol
def gather_vals(
        df_deq,
        ftols,
        btols,
        value_key="F_mae_mean",
    ):
    dic_vals = {}

    for (molecule, Rep, model_tag, z, g_bar), df_model in tqdm(df_deq.groupby(['molecule_name', 'Rep', 'model_tag', 'z', 'g_bar'])):
        vals = np.full((len(ftols), len(btols)), np.nan, dtype=float)
        z, g_bar = int(z), int(g_bar)

        for i, ft in enumerate(ftols):
            for j, bt in enumerate(btols):
                m = (df_model.forw_tol.eq(ft) & df_model.back_tol.eq(bt))
                rows = df_model.loc[m, value_key]
                # rows = df_model.loc[m, "forw_iter_mean"]
                # rows = df_model.loc[m, "back_iter_mean"]
                assert len(rows) <= 1, f"More than one match for forw_tol={ft}, back_tol={bt}"
                if len(rows) == 1:
                    vals[i, j] = float(rows.iloc[0])

        dic_vals[(
            molecule,
            Rep,
            # model_tag
            z, g_bar
            )] = vals
    return dic_vals

    
# to add little scatter at tols that first go below tolerable fmae
def add_threshold_borders_scatter(Z, x, y, thr, ax=None, s=10, continue_south_and_west=1, **scatter_kwargs):
    """
    Z: (ny, nx) 2D array (may contain NaN as 'non-values')
    x: (nx,) x-coordinates for columns of Z
    y: (ny,) y-coordinates for rows of Z
    thr: scalar threshold; we find borders of region where Z < thr
    Excludes:
      - borders that lie on the outer boundary of the array
      - borders adjacent to non-values (NaN) on either side
    Adds scatter points at midpoints of the grid-edges where a border occurs.
    Returns (xb, yb) arrays of border-point coordinates.
    """
    Z = np.asarray(Z)
    x = np.asarray(x)
    y = np.asarray(y)

    ny, nx = Z.shape
    assert x.shape == (nx,)
    assert y.shape == (ny,)
    
    inside = (Z <= thr) & np.isfinite(Z)

    border = np.zeros_like(inside, dtype=bool)

    border[1:-1, 1:-1] = inside[1:-1, 1:-1] & (
        (np.isfinite(Z[1:-1, :-2]) & ~inside[1:-1, :-2]) |  # left
        (np.isfinite(Z[1:-1, 2: ]) & ~inside[1:-1, 2: ]) |  # right
        (np.isfinite(Z[:-2, 1:-1]) & ~inside[:-2, 1:-1]) | # down
        (np.isfinite(Z[2: , 1:-1]) & ~inside[2: , 1:-1])   # up
    )

    if ax is None:
        ax = plt.gca()
    ii, jj = np.nonzero(border)
    
    if continue_south_and_west:
        # west - norther most point
        idx = np.argmin(ii*1000+jj)
        print(idx)
        wn_i = ii[idx]
        wn_j = jj[idx]
        W_i = list(range(0,wn_i+1))
        W_j = [wn_j] * len(W_i)

        # south - easter most point
        idx = np.argmin(jj*1000-ii)
        print(idx)
        se_i = ii[idx]
        se_j = jj[idx]
        S_j = list(range(0,se_j+1))
        S_i = [se_i] * len(S_j)
        
        ii = list(ii) + W_i + S_i
        jj = list(jj) + W_j + S_j

    xb = x[jj]
    yb = y[ii]
    ax.scatter(xb, yb, s=s, **scatter_kwargs)
    return ii, jj
