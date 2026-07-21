import time
from typing import Dict, List

import wandb

from schnetpack.md.simulator import Simulator
from schnetpack.md.simulation_hooks.callback_hooks import TensorBoardLogger


class WandBWriter:
    """
    Emulates tensorboardX.SummaryWriter for logging to Weights & Biases.
    """
    def add_scalars(self, mol_name: str, logger_dict: Dict, step: int):
        """
        Log the scalar data associated with the target property.
        An individual entry is created for every replica and molecule.

        Args:
            mol_name (str): Base name of the property group to log.
            logger_dict (dict): Dictionary holding the scalar data to be logged.
            step (int): Current simulation step.
"""
        if set(logger_dict.keys()) == {'centroid', 'r01'}:
            # if only one replica is used, the centroid is the same as the replica.
            del logger_dict['centroid']
            
        logger_dict = {f"{mol_name}/{k}": v for k, v in logger_dict.items()}
        wandb.log(logger_dict, step=step)

class WandBLoggerMD(TensorBoardLogger):
    """
    Logging scalar information of the system replicas and molecules collected during the simulation to WandB.
    Reuses most code from the TensorBoardLogger.

    Args:
        properties (list): List of properties to log.
        every_n_steps (int): Frequency with which data is logged to TensorBoard.
        **wandb_kwargs: Additional keyword arguments for wandb.init.
    """

    def __init__(self, properties: List = ['energy', 'temperature'], every_n_steps: int = 100, hyperparams=None, log_properties=None, **wandb_kwargs):
        super(WandBLoggerMD, self).__init__(log_file='', properties=properties, every_n_steps=every_n_steps)
        
        # capture and extend wandb_kwargs
        self.hyperparams = hyperparams
        self.wandb_kwargs = dict(entity="qcml-group", project="md-dev") # defaults
        self.wandb_kwargs.update(wandb_kwargs)
        
        md_name = self.wandb_kwargs.get('name', 'MD')
        mol_name = self.hyperparams.get('molecule_config', {}).get('molecule_name', '')
        model_config_description = self.hyperparams.get('model_config_description', '')
        self.wandb_kwargs['name'] = f"{mol_name}/{md_name}/{model_config_description}"
        
        # close the tensorboardX.SummaryWriter
        # (that was initialized in the BasicTensorBoardLogger init)
        self.writer.close()
        del self.writer

        # create the wandb writer
        self.writer = WandBWriter()
        self.log_properties=log_properties
        
    def on_simulation_start(self, simulator: Simulator):
        super(WandBLoggerMD, self).on_simulation_start(simulator)
        self.run = wandb.init(**self.wandb_kwargs, config=self.hyperparams)
        print(f'Initialized WandB run {self.wandb_kwargs.get("name")} with:\n{self.wandb_kwargs}\nand hyperparams:\n{self.hyperparams}')
        self.total_time = None
        self.time_per_step = None

    def log_message(self, simulator: Simulator):
        if msg := getattr(simulator, 'log_message', None):
            wandb.log(msg, step=simulator.step)
            delattr(simulator, 'log_message')

    def on_simulation_end(self, simulator: Simulator):
        self.log_message(simulator)
        if self.run is not None:
            self.run.finish()
            self.run = None
        
    def on_step_finalize(self, simulator: Simulator):
        super(WandBLoggerMD, self).on_step_finalize(simulator)

        # log termination info to wandb
        self.log_message(simulator)

        # if not self.log_properties: return
        
        results = getattr(simulator.calculator, 'results', None)
        if not results: return
        
        log = {
            p: results[p]
            for p in self.log_properties
            if p in results
        }
        
        ## LOG FROM BACK
        # for dir in ['forw', 'back']:
        #     for m in [1,2,3]: # how far from the back to look
        #         if len(results[f'{dir}_res']) > m+1:
        #             key = sorted(list(results[f'{dir}_res'].keys()))[-m]
        #             val = results[f'{dir}_res'][key]
        #         else:
        #             val = None
        #         log[f'{dir}_res_m{m}'] = val
              
        ## LOG FROM FRONT  
        for dir in ['forw', 'back']:
            if f'{dir}_res' not in results:
                continue
            
            for m in [0,1,2,3,4]: # which res to log
                if m < len(results[f'{dir}_res']):
                    key = sorted(list(results[f'{dir}_res'].keys()))[m]
                    val = results[f'{dir}_res'][key].max() # max over all atoms in molecule
                else:
                    val = None
                log[f'{dir}_res_{m}'] = val
            
        # log residuals to wandb
        if log:
            wandb.log(log, step=simulator.step)