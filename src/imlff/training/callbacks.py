import time, datetime
import pytorch_lightning as pl
import wandb

from imlff.training.util import delete_artifacts

class TimeLimitCallback(pl.Callback):
    def __init__(self, deadline):
        self.deadline = deadline  # Absolute time to stop training
        
        # Set the same callback for both training and validation
        self.on_train_batch_start = self.on_validation_batch_start

    def on_validation_batch_start(self, trainer, *_):
        current_time = time.time()
        if current_time >= self.deadline:
            trainer.should_stop = True
            formatted_time = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Stopping training at {formatted_time} (deadline reached).")
            
class CustomBestMetricTracker(pl.Callback):
    def __init__(self, monitor="val_loss", mode='min', measure=[
        'val_energy_MAE', 'val_forces_MAE',
        # new training routine:
        'val_s001_warm_z0_g_bar0_forw_iter_0.01',
        'val_s001_warm_z0_g_bar0_back_iter_0.005',
        'val_s002_warm_z1_g_bar1_forw_iter_0.01',
        'val_s002_warm_z1_g_bar1_back_iter_0.005',
        # old training routine:
        'val_forces_MAE_01', 'val_forces_MAE_next_warm_01',
        'val_forces_MAE_02', 'val_forces_MAE_next_warm_02',
        ]):
        self.mode = mode
        self.monitor = monitor
        self.best = {metric: None for metric in measure}
        self.best[monitor] = float('inf') if mode == 'min' else -float('inf')
        self.last = self.best.copy()
        self.best_epoch = 'NA'
        self.last_epoch = 'NA'
    
    def on_validation_end(self, trainer, pl_module):
        self.last = {
            metric: trainer.callback_metrics.get(metric, None)    
            for metric in self.best.keys()
        }
        self.last_epoch = trainer.current_epoch
        
        if self.last[self.monitor] is None:
            print(f"Warning: monitor {self.monitor} not found in callback metrics.")
            return
        
        if (self.mode == 'min' and self.last[self.monitor] < self.best[self.monitor]) or \
            (self.mode == 'max' and self.last[self.monitor] > self.best[self.monitor]):
            self.best = self.last
            self.best_epoch = self.last_epoch
            self.log_to_wandb()
            
    def log_to_wandb(self):
        metrics = self.get_metrics('best', with_epoch=True, with_labels=True)
        wandb.log(metrics)

    def get_metrics(self, which, with_epoch=True, with_labels=False):
        if which == 'best': metrics, epoch = self.best, self.best_epoch
        if which == 'last': metrics, epoch = self.last, self.last_epoch
        
        ret = metrics.copy()
        if with_epoch: ret['epoch'] = epoch
        
        if with_labels:
            if which == 'best': lbl = f' (at min {self.monitor})'
            if which == 'last': lbl = ' (last)'
            ret = {key+lbl: value for key, value in ret.items()}
        
        return ret
            
    def get_metrics_string(self, which, with_epoch=True):
        if which == 'best': metrics, epoch = self.best, self.best_epoch
        if which == 'last': metrics, epoch = self.last, self.last_epoch
        form_val = lambda val: f"{val:.4f}" if val not in [None, float('inf'), float('-inf')] else 'NA'
        metrics_str = '_'.join([f"{key}={form_val(value)}" for key, value in metrics.items()])
        if with_epoch:
            if epoch != 'NA': metrics_str = f"e{epoch:04d}_" + metrics_str
            else:             metrics_str = f"eNA_"          + metrics_str
        return metrics_str

class CleanWandbArtifactsCallback(pl.Callback):
    def __init__(self, dry_run=False, safe_mode=False):
        self.dry_run = dry_run
        self.safe_mode = safe_mode

    def _get_wandb_info(self, pl_module):
        """
        Helper function to retrieve W&B project name and run name from the logger.
        Returns a tuple of (wandb_project, run_name) or (None, None) if unavailable.
        """
        if not hasattr(pl_module, 'logger'):
            return None, None, None, None

        logger = pl_module.logger
        return (
            getattr(logger, '_wandb_entity', None),
            getattr(logger, '_project_name', None),
            getattr(logger, '_checkpoint_name', None),
            getattr(logger, '_clean_at_n_epoch', None),
        )

    def on_train_epoch_end(self, trainer, pl_module):
        wandb_entity, project_name, run_name, clean_at_n_epoch = self._get_wandb_info(pl_module)
        if project_name and run_name and clean_at_n_epoch and trainer.current_epoch % clean_at_n_epoch == 0:
            delete_artifacts(wandb_entity, project_name, run_name, self.dry_run, self.safe_mode)

    def on_fit_end(self, _, pl_module):
        print('\033[31mEnd of training. Final cleaning of wandb artifacts.\033[0m')
        wandb_entity, project_name, run_name, _ = self._get_wandb_info(pl_module)
        if project_name and run_name:
            delete_artifacts(wandb_entity, project_name, run_name, self.dry_run, self.safe_mode)