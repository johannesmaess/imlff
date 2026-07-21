import argparse
import os
import sys
import traceback

import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from imlff.common.util import proj_path
from imlff.analysis.warmstart_run import analyze_warmstart, clear_warmstart_run_caches, warmstart_exists
from imlff.analysis.warmstart_summarize import max_iters_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Warmstart analysis runner (Codex variant).")
    parser.add_argument("array_idx", nargs="?", type=int, default=None, help="Optional model index to run.")
    parser.add_argument("array_stride", nargs="?", type=int, default=None, help="Stride for array job splitting. Defaults to inf (run only array_idx).")

    parser.add_argument("--per-batch", action="store_true", help="Run one batch at a time and optionally aggregate.")
    parser.add_argument("--reverse-max-iters", action="store_true", help="Reverse the max-iter grid order.")
    parser.add_argument("--reverse", action="store_true", help="Reverse the dataframe order.")
    parser.add_argument("--bsz", type=int, default=50, help="Batch size.")
    parser.add_argument(
        "--initial-per-batch-bsz",
        type=int,
        default=None,
        help="Starting batch size when falling back to per-batch mode after OOM. Defaults to max(1, bsz//2).",
    )
    parser.add_argument(
        "--try-only-one-bsz",
        action="store_true",
        help="If set, do not try smaller batch sizes on OOM (skip model instead).",
    )
    parser.add_argument("--default-forw-tol", type=float, default=0.01, help="Default forward tol for baseline run.")
    parser.add_argument(
        "--default-back-tol",
        type=float,
        default=None,
        help="Default backward tol for baseline run. Defaults to --default-forw-tol.",
    )
    parser.add_argument(
        "--max-iters",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Opverride max-iter grid for all models. Provide 1 to 5 ints as: "
            "(maxi) | (maxi bmaxi) | (maxi bmaxi mini) | (maxi bmaxi mini bmini) | (maxi bmaxi mini bmini along_diagonal). "
            "Uses max_iters_grid(maxi, mini=1, bmaxi=None, bmini=None, along_diagonal=1000000)."
        ),
    )
    parser.add_argument(
        "--runs-file",
        type=str,
        default=None,
        help="CSV basename under result_summary/ (without .csv). If set, loads selected from result_summary/<runs_file>.csv without applying filters.",
    )
    parser.add_argument(
        "--which-tag",
        type=str,
        default='best_manual',
        help="Use {latest, best, better}_tag? better should be set in df as the better of the two by val_forces.",
    )
    parser.add_argument(
        "--idx-mi",
        action="store_true",
        help=(
            "Use array_idx as a global counter over all (ws_degree, model, max_iter) combinations "
            "instead of slicing the model dataframe. Skips row-level array_idx selection and instead "
            "runs only the single (k, i, j) combination whose running counter equals array_idx."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check which models are missing F_mae_next.npy for all (ws_degree, max_iter) combinations "
            "and write a sub-CSV of incomplete rows. No GPU work is done. "
            "Output path: result_summary/runs/<tag>/complete_warmstart_warm_tmp.csv (with --tag) "
            "or result_summary/complete_warmstart_analysis_warm_tmp.csv (without --tag)."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for --check output path: result_summary/runs/<tag>/complete_warmstart_warm_tmp.csv.",
    )
    parser.add_argument(
        "--ws-degree",
        nargs=2,
        type=int,
        default=(1, 1),
        metavar=("Z", "G_BAR"),
        help=(
            "Warmstart degree (z, g_bar) to run/check. Default (1, 1). "
            "Use '-1 -1' for the no-warmstart (cold) baseline (g is always -1). "
            "Applies to both the run loop and --check."
        ),
    )

    args = parser.parse_args()

    per_batch = args.per_batch
    reverse_max_iters = args.reverse_max_iters
    reverse = args.reverse
    bsz = args.bsz
    initial_per_batch_bsz = args.initial_per_batch_bsz if args.initial_per_batch_bsz is not None else max(1, bsz // 2)
    try_only_one_bsz = args.try_only_one_bsz
    default_forw_tol = args.default_forw_tol
    default_back_tol = args.default_back_tol if args.default_back_tol is not None else default_forw_tol
    array_idx = args.array_idx
    array_stride = args.array_stride
    which_tag = args.which_tag
    idx_mi = args.idx_mi
    check = args.check
    ws_degree = tuple(args.ws_degree)

    # Cap analysis to 20k conformations (see analyze_warmstart default limit=20000).
    limit = 20_000

    cli_max_iters = None
    if args.max_iters is not None:
        if not (1 <= len(args.max_iters) <= 5):
            raise ValueError("--max-iters expects 1 to 5 integers.")
        cli_max_iters = max_iters_grid(*args.max_iters)

    def _clear_exception_refs(exc: BaseException) -> None:
        # Important for CUDA OOM retries: the exception traceback can hold references
        # to large CUDA tensors (inputs/outputs/model) via frame locals.
        try:
            traceback.clear_frames(exc.__traceback__)
        except Exception:
            pass

    def _default_max_iters(row, model_tag, warmstart_degree):
        # Non-DEQ models: no grid; only run once.
        # Match analyze_warmstart_core's is_deq check (model_key-based) to avoid
        # passing (None, None) into the DEQ branch when row.type and model_key disagree.
        if 'mi' not in str(row.model_key):
            return [(None, None)]

        # Default DEQ sweep.
        mis = max_iters_grid(maxi=8, bmaxi=6)

        # Special-case: no warmstart uses a wider grid by default.
        if warmstart_degree == (-1, -1):
            # mis = max_iters_grid(maxi=16, along_diagonal=3)
            mis = max_iters_grid(maxi=16)

        # Known-outlier run tags may need more iterations.
        if "g51tassj" in model_tag:
            mis = max_iters_grid(maxi=16)

        return mis

    print(
        "CLI",
        dict(
            array_idx=array_idx,
            array_stride=array_stride,
            per_batch=per_batch,
            bsz=bsz,
            initial_per_batch_bsz=initial_per_batch_bsz,
            try_only_one_bsz=try_only_one_bsz,
            reverse_max_iters=reverse_max_iters,
            reverse=reverse,
            default_forw_tol=default_forw_tol,
            default_back_tol=default_back_tol,
            limit=limit,
            max_iters=("custom" if cli_max_iters is not None else "default"),
            runs_file=args.runs_file,
            tag=args.tag,
            idx_mi=idx_mi,
            ws_degree=ws_degree,
        ),
    )

    if args.runs_file is not None:
        selected = pd.read_csv(proj_path(f"result_summary/{args.runs_file}.csv", verbose=False))
    elif args.tag is not None:
        incomplete_path = proj_path(f'result_summary/runs/{args.tag}/complete_warmstart_warm_tmp.csv')
        full_path       = proj_path(f'result_summary/runs/{args.tag}/runs_best_per_group.csv')
        if array_idx is not None and os.path.exists(incomplete_path):
            print(f'Loading incomplete CSV: {incomplete_path}')
            selected = pd.read_csv(incomplete_path)
        else:
            print(f'Loading full CSV: {full_path}')
            selected = pd.read_csv(full_path)
    else:
        selected = pd.read_csv('result_summary/complete_warmstart_analysis_warm_tmp.csv')
        # selected = pd.read_csv('result_summary/complete_warmstart_analysis_cold_tmp.csv')
        # selected = pd.read_csv('result_summary/complete_warmstart_analysis_explicit_tmp.csv')
    
    selected["AtomisticRepresentationClass"] = pd.Categorical(selected["AtomisticRepresentationClass"], categories=["SchNet", "PaiNN", "SO3net"], ordered=True)
    selected = selected.sort_values("AtomisticRepresentationClass").reset_index(drop=True)

    
    print(selected)
    print('n models:', len(selected))

    if check:
        ws_degrees_check = [ws_degree]
        incomplete_pos = set()
        for k, warmstart_degree in enumerate(ws_degrees_check):
            for i, (_, row) in enumerate(selected.iterrows()):
                model_tag = row[f'{which_tag}_tag']
                model_tag = 'qcml-group/dev-v10/' + model_tag
                max_iters = cli_max_iters if cli_max_iters is not None else _default_max_iters(row, model_tag, warmstart_degree)
                all_exist = all(
                    warmstart_exists(
                        model_tag,
                        row.to_dict(),
                        forw_max_iter=forw,
                        back_max_iter=back,
                        default_forw_tol=default_forw_tol,
                        default_back_tol=default_back_tol,
                        back_iter_on_g_bar=True,
                        z=warmstart_degree[0],
                        g=-1,
                        g_bar=warmstart_degree[1],
                        limit=limit,
                        results_dir='results',
                    )
                    for forw, back in max_iters
                )
                if not all_exist:
                    incomplete_pos.add(i)
        incomplete_df = selected.iloc[sorted(incomplete_pos)]
        if args.tag:
            out_path = proj_path(f'result_summary/runs/{args.tag}/complete_warmstart_warm_tmp.csv')
        else:
            out_path = proj_path('result_summary/complete_warmstart_analysis_warm_tmp.csv')
        incomplete_df.to_csv(out_path, index=False)
        print(f'Missing: {len(incomplete_df)} / {len(selected)} models')
        print(f'Saved: {out_path}')
        sys.exit(0)

    # reverse df order:
    if reverse:
        selected = selected[::-1].reset_index(drop=True)
        
    if not idx_mi:
        if array_idx is not None:
            if array_stride is not None:
                selected = selected.iloc[array_idx::array_stride].reset_index(drop=True)
            else:
                selected = selected.iloc[[array_idx]].reset_index(drop=True)
        else:
            selected = selected[::-1].reset_index(drop=True)

    global_counter = 0

    ws_degrees = [ws_degree]
    K = len(ws_degrees)
    for k, warmstart_degree in enumerate(ws_degrees):

        I = len(selected)
        for i, (_, row) in enumerate(selected.iterrows()):
            model_tag = row[f'{which_tag}_tag']
            model_tag = 'qcml-group/dev-v10/' + model_tag
            
            
            print('\n'*5)
            print('='*80)
            print(f'({k+1}/{K} ws_degrees) ({i+1}/{I} models) --  Running for {model_tag}')    
            print('='*80)
            
            max_iters = cli_max_iters if cli_max_iters is not None else _default_max_iters(row, model_tag, warmstart_degree)

            if reverse_max_iters:
                max_iters = max_iters[::-1]
            
            try_bsz = bsz
            try_per_batch = per_batch
            arrived_at_j_without_failing = 0
            
            while try_bsz > 0:
                # try with decreasing batch sizes
                try:
                    # mis = max_iters[array_idx or 0::(1 if array_idx is None else 10)]
                    mis = max_iters
                    J = len(mis)
                    for j, (forw, back) in enumerate(mis):

                        if idx_mi:
                            current_count = global_counter
                            global_counter += 1
                            if array_idx is not None:
                                if array_stride is not None:
                                    if current_count % array_stride != array_idx:
                                        continue
                                elif current_count != array_idx:
                                    continue

                        # don't replay if you failed through a too large batch size:
                        if arrived_at_j_without_failing > j:
                            continue
                        arrived_at_j_without_failing = j
                        
                        print()
                        print()
                        print(f'({k+1}/{K} ws_degrees) ({i+1}/{I} models) ({j+1}/{J} iters) Running forw:', forw, 'back:', back)
                        
                        analyze_warmstart(
                            model_tag,
                            config=row.to_dict(),
                            forw_max_iter=forw,
                            back_max_iter=back,
                            # measure_restricted=True,
                            default_forw_tol=default_forw_tol,
                            default_back_tol=default_back_tol,
                            bsz=try_bsz,
                            per_batch=try_per_batch,
                            limit=limit,
                            
                            # warmstart ? -1 means no.
                            g=-1,
                            z    =warmstart_degree[0],
                            g_bar=warmstart_degree[1],
                            
                            back_iter_on_g_bar = True,
                            measure_default = 'only_next',
                            
                            # partition='val',
                            save_atomwise_degree_residuals=False,
                            save_atomwise_residuals=False,
                            
                            results_dir='results' if torch.cuda.is_available() else  'results-local'
                        )
                    break # successfully computed, leave while loop!
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:

                    if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                        raise  # unrelated RuntimeError – let it propagate
                        
                    if try_only_one_bsz:
                        print('Configured to only try one batch size, but went OOM. Skipping this model.')
                        break
                    
                    # clean caches
                    _clear_exception_refs(e)
                    del e
                    clear_warmstart_run_caches()
                    
                    try_bsz //= 2
                    
                    if try_bsz:
                        print(f'Received OutOfMemoryError, trying with smaller batch size: {try_bsz}' + try_per_batch * ' (running one batch at a time)')
                    elif per_batch is False:
                        
                        print(f'Received OutOfMemoryError, while running just one sample at a time (though all batches results are in memory). Skipping this run.')
                    
                        try_per_batch = True
                        try_bsz = initial_per_batch_bsz
                        print(f'Try computation per batch. Bsz: {try_bsz}')
                    else:
                        print(f'Received OutOfMemoryError, while running just one sample at a time. Skipping this model.')
                        break
                    
                except Exception as e:
                    print("*"*80)
                    print('Received exception:', e)
                    traceback.print_exc()
                    print('Skipping this run.')
                    print("*"*80)
                    _clear_exception_refs(e)
                    del e
                    
                    break
