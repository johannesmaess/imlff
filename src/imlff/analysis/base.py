import wandb
import pandas as pd
import numpy as np
import copy
import re
from tqdm import tqdm
from datetime import datetime, timedelta
from imlff.data.info import molecule_to_n_atoms

def fetch_runs(project_name, created_within=None, created_after=None, entity='qcml-group', tags=None, summary=None, **more_config_filters):
    filters = dict()
    
    # Add generic filters
    for name, value in more_config_filters.items():
        if value is not None:
            prefixes = 'config.', 'summary_metrics.'
            has_prefix = any(name.startswith(prefix) for prefix in prefixes)
            is_known = name in ['created_at', 'heartbeat_at']
            if not (has_prefix or is_known):
                name = prefixes[0] + name
            filters[name] = value
            
    if tags is not None:
        filters["tags"] = tags
        
    # Add time-based filters
    if created_within:
        now = datetime.now()
        x_hours_ago = now - timedelta(hours=created_within)
        filters["created_at"] = {"$gte": x_hours_ago.isoformat()}
    elif created_after:
        patterns = ["%y%m%d-%H%M%S", "%y%m%d-%H%M", "%y%m%d-%H", "%y%m%d"]
        created_after_dt = None
        for pattern in patterns:
            try:
                created_after_dt = datetime.strptime(created_after, pattern)
                break
            except ValueError:
                pass
        assert created_after_dt is not None, f"Could not parse {created_after} with any of the patterns {patterns}"
        filters["created_at"] = {"$gte": created_after_dt.isoformat()}

    # Fetch runs from WandB API
    api = wandb.Api()
    print(f"Fetching runs from {project_name} with filters {filters}")
    runs = api.runs(f'{entity}/{project_name}', filters=filters)
    
    print(f"Found {len(runs)} runs.")
    if not len(runs):
        raise ValueError("No runs found. Please check your filters.")
    
    if summary:
        runs = [run for run in runs if all((run.summary.get(k) == v) for k, v in summary.items())]
        print(f"Filtered down to {len(runs)} runs.")
    
    return runs

def fetch_runs_list(max_runs=None, summary_black_list=[
    'train_.*', 
    '.*_warm_z0_.*', 
    # '.*_res_.*'
    ], **kwargs):
    runs = fetch_runs(**kwargs)
    runs = [r for r in runs]
    if max_runs is not None:
        if type(max_runs) is int:
            assert max_runs > 0, "max_runs should be a positive integer"
            runs = runs[:max_runs]
        elif type(max_runs) is slice:
            print(type(max_runs), max_runs)
            runs = runs[max_runs]
            
        elif type(max_runs) is list:
            pass
        else:
            assert False, "max_runs should be a positive integer, a slice, or None"
    print('after filter:', len(runs), flush=True)

    def process_run(run):
        if dic := run.summary._json_dict:
            if '_wandb' in dic:
                dic.update(dic['_wandb'])
                del dic['_wandb']
            for key in list(dic.keys()):
                if any([re.match(pattern, key) for pattern in summary_black_list]):
                    del dic[key]
            dic.update(run.config)
            dic['name'] = run.name
            dic['run_id'] = run.id
            dic['state'] = run.state
            dic['run'] = run
            return dic

    runs = [process_run(run) for run in tqdm(runs)]
    
    runs = [run for run in runs if run is not None]
    return runs

def fetch_runs_df(runs_list=None, fix=True, **kwargs):
    runs_df = pd.DataFrame(runs_list or fetch_runs_list(**kwargs))
    if fix: runs_df = fix_wandb_df(runs_df, verbose=False)
    return runs_df

def fix_wandb_df(runs_df, verbose=True):
    if verbose: print('filling nan with 1: n_schnets')
    runs_df.n_schnets = pd.to_numeric(runs_df.n_schnets, errors='coerce').fillna(1).astype(int)
    if verbose: print('filling nan with 1: n_interactions')
    runs_df.n_interactions = pd.to_numeric(runs_df.n_interactions, errors='coerce').fillna(1).astype(int)
    
    ## Shared Interactions ##
    runs_df['shared_interactions'] = runs_df.shared_interactions.fillna(False)
    
    ## Shared Schnets ##
    runs_df['shared_schnets'] = runs_df.shared_schnets.fillna(False)
    
    ## Shared Filters ##
    if 'shared_filters' not in runs_df.columns:
        runs_df['shared_filters'] = np.nan
    # On PaiNN it is True if it was not logged
    runs_df.loc[
        runs_df.shared_filters.isna()
        & (runs_df.AtomisticRepresentationClass=='PaiNN'),
        'shared_filters'
    ] = False 
    
    # Otherwise it is equal to shared_interactions
    runs_df.loc[
        runs_df.shared_filters.isna()
        & runs_df.shared_interactions,
        'shared_filters'
    ] = True
    runs_df.loc[
        runs_df.shared_filters.isna()
        & ~runs_df.shared_interactions,
        'shared_filters'
    ] = False
    
    # 'type' is only a shorthand used for the plots (that's why it's not saved in WandB)
    def make_type(row):
        if row.use_deq == True or row.shared_schnets == 'deq':
            if row.get('norm_s'): #  or row.norm_o or row.norm_i:
                typ = 'deq-norm-s'
            elif row.get('norm_o'):
                typ = 'deq'
            elif row.get('norm_i'):
                typ = 'deq-norm-i'
            else:
                return 'deq-wo-norm'
            mnc = row.get('MergedNormClass')
            # NaN is truthy in Python; '' is the post-fill sentinel (line below).
            # Only a real class name means a merged-norm run.
            if pd.notna(mnc) and mnc != '':
                typ += '-merged'
            return typ
        postfix = ''
        if row.get('norm_s'):
            postfix += '-norm-s'
        if row.get('norm_o'):
            postfix += '-norm-o'
        if row.get('norm_i'):
            postfix += '-norm-i'
        if row.n_schnets != 1:
            assert row.n_interactions == 1, row.n_interactions
            postfix += '-ii'
        typ = 'explicit'
        if row.shared_interactions:
            if row.AtomisticRepresentationClass=='PaiNN' and row.shared_filters:
                typ = 'explicit-tt'
            else:
                typ = 'explicit-tied'
        return typ + postfix
    runs_df['type'] = runs_df.apply(make_type, axis=1)
    
    assert np.all(runs_df.query('type == "deq"').n_interactions == 1)
    
    for col in runs_df.columns:
        if 'val_' in col:
            if verbose: print('filling nan with inf:', col)
            # runs_df[col].fillna(float('inf'), inplace=True)
            runs_df[col] = runs_df[col].fillna(float('inf'))
        elif 'NormClass' in col:
            if verbose: print('filling nan with empty string:', col)
            runs_df[col] = runs_df[col].fillna('')
            
    runs_df['n_atoms'] = runs_df.molecule.apply(molecule_to_n_atoms)
    return runs_df.copy()

    api = wandb.Api()
    
    runs_df['best_tag'] = None
    runs_df['latest_tag'] = None

    n = len(runs_df)
    for i, (idx, row) in enumerate(runs_df.iterrows()):
        run_id = row['run_id']
        run = api.run(f"qcml-group/dev-v10/{run_id}")
        artifacts = run.logged_artifacts()
        print(f"\n({i+1}/{n}) Artifacts for run {run_id}:")
        # artifact_info[run_id] = []
        latest_manual_v = -1
        latest_manual = None
        for artifact in artifacts:
            if 'model-' not in artifact.name:
                continue
            aliases = artifact.aliases
            print(f"  - {artifact.name} (aliases: {aliases})")
            
            if 'best' in aliases:
                runs_df.at[idx, 'best_tag'] = artifact.name
            if 'latest' in aliases:
                runs_df.at[idx, 'latest_tag'] = artifact.name
                artifact.aliases.append(today_tag+'-latest')
                artifact.save()
            
            v = int(artifact.name.split(':v')[1])
            if v > latest_manual_v:
                latest_manual_v = v
                latest_manual = artifact
        else:
            if runs_df.at[idx, 'latest_tag'] is None and latest_manual is not None:
                runs_df.at[idx, 'latest_tag'] = latest_manual.name
                latest_manual.aliases.append(today_tag+'-latest')
                latest_manual.save()