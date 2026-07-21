import gc
import traceback

import torch


def cleanup_after_oom(exc: BaseException) -> None:
    """Aggressively free GPU memory after an OOM error.

    Exception tracebacks frequently retain references to large CUDA tensors
    via frame locals — clear those first before emptying the cache.
    """
    try:
        traceback.clear_frames(exc.__traceback__)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_gpu(*objects) -> None:
    """Delete objects and free GPU memory. Call after each model inference."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_with_oom_retry(fn, data_config: dict):
    """Call fn(data_config) retrying with halved batch_size on OOM.

    fn receives the (mutable) data_config dict and should create its own
    trainer and data module from it. It is responsible for cleaning up the
    trainer on both success and failure (e.g. via try/finally).
    data_config['batch_size'] is mutated in-place on each retry.
    """
    while True:
        try:
            return fn(data_config)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'CUDA out of memory' not in str(e):
                raise
            cleanup_after_oom(e)
            del e
            data_config['batch_size'] //= 2
            if data_config['batch_size'] == 0:
                raise RuntimeError('OOM even at batch_size=1')
            print(f'WARNING: OOM — retrying with batch_size={data_config["batch_size"]}')
