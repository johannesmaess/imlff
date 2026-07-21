import os
import string
import time
import torch
import wandb
import random
import logging
from filelock import FileLock
from contextlib import nullcontext

from functools import lru_cache, partial

from pathlib import Path

from ase import Atoms
from ase.db import connect
from ase.optimize.lbfgs import LBFGS
from ase.io import write, read

from schnetpack import properties
from schnetpack.transform.neighborlist import ASENeighborList
from schnetpack.md.system import System
from schnetpack.md.simulator import Simulator
from schnetpack.md import simulation_hooks
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.initial_conditions import UniformInit
from schnetpack.md.simulation_hooks import callback_hooks, LangevinThermostat
from schnetpack.md.calculators.schnetpack_calculator import SchNetPackCalculator
from schnetpack.interfaces.ase_interface import SpkCalculator

from imlff.loading import load_model_from_wandb, turn_iter_into_deq
from imlff.md.calculator import make_stateful_calculator
from imlff.md.stateful_extrapolator import StatefulAdamsBashforth
from imlff.common.util import proj_path
from imlff.model.util import solving_str
from imlff.data.info import molecule_to_dataset
from imlff.model.solvers import fixed_point_iteration
from imlff.model.deq_inference import to_inference

# Hooks

class NaNCheckHook(simulation_hooks.SimulationHook):
    def on_step_end(self, simulator: Simulator):
        if torch.isnan(simulator.system.positions).any():
            raise ValueError("NaN detected in atomic positions.")

class BondBreakCheckHook(simulation_hooks.SimulationHook):
    def __init__(self, *args, bonds_and_stats=None, k_termination=6, r_termination=1., m_termination=.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert bonds_and_stats is not None
        pairs = torch.tensor(bonds_and_stats['bonds'])
        self.bond_indices_i = pairs[:, 0]
        self.bond_indices_j = pairs[:, 1]
        self.r0s = torch.tensor(bonds_and_stats['r0s'])
        self.sigmas = torch.tensor(bonds_and_stats['sigmas'])
        
        self.k_termination = k_termination
        self.r_termination = r_termination
        self.m_termination = m_termination
        
        self.ks_log = tuple(range(3, 21))
        self.rs_log = .3, .4, .5, .6, .7, .8, .9, 1.
        self.ds_log = .3, .4, .5, .6, .7, .8, .9, 1.
        self.ms_log = .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.
        
        self.pair_indices_i = []
        self.pair_indices_j = []
        n_atoms = pairs.max() # assume there is no unbounded atom.
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                self.pair_indices_i.append(i)
                self.pair_indices_j.append(j)
        
        self.first_above_k = {}
        self.first_below_r = {}
        self.first_above_r = {}
        self.first_below_d = {}
        self.first_above_d = {}
        self.first_below_m = {}
        
        self.termination_dict = {}
        
        
    def calculate_bond_stats(self, R):
        # divergences is the difference by which the atom pair deviates from their equilibrium distance
        # relative_divergences is normalized (divided) by the equilibrium distance
        # ks is the multiple that this relative distance takes over the expected standard deviation (std is calculated by fitting an harmonic oscillator to MD statistics of the bonded atom pair)
        
        dists = torch.norm(R[self.bond_indices_i] - R[self.bond_indices_j], dim=1).cpu()
        divergences = dists - self.r0s
        
        relative_divergences = divergences / self.r0s
        ks = divergences.abs() / self.sigmas
        
        return ks, relative_divergences, divergences
    
    def calculate_min_distance(self, R):
        return torch.norm(R[self.pair_indices_i] - R[self.pair_indices_j], dim=1).min()
        
    def gather_errors(self, R, simulator_step):
        
        # assume r0s, sigmas are on same scale as input system unit (with which the calculator was initialized):
        
        ks, rel, div = self.calculate_bond_stats(R)
        k = ks.max().item()
        r_min = rel.min().item()
        r_max = rel.max().item()
        d_min = div.min().item()
        d_max = div.max().item()
        m = self.calculate_min_distance(R).item()
        
        curr_termination_dict = {}
        
        for kmax in sorted(self.ks_log):
            if k > kmax:
                if kmax not in self.first_above_k:
                    self.first_above_k[kmax] = simulator_step
                    print(f"First step with k > {kmax}: {simulator_step}")
                    curr_termination_dict[f'k.first_step_above.{kmax}'] = simulator_step
                    self.termination_dict[f'k.first_step_above.{kmax}'] = simulator_step
            else:
                break
        
        for rmax in sorted(self.rs_log):
            found = False
            
            if r_min < -rmax:
                found = True
                if rmax not in self.first_below_r:
                    print(f'First relative bond divergence shorther than {rmax}:', simulator_step, r_min)
                    self.first_below_r[rmax] = simulator_step
                    curr_termination_dict[f'r.first_step_below.{str(rmax).replace(".", "_")}'] = simulator_step
                    self.termination_dict[f'r.first_step_below.{str(rmax).replace(".", "_")}'] = simulator_step
            
            if r_max > rmax:
                found = True
                if rmax not in self.first_above_r:
                    print(f'First relative bond divergence longer than {rmax}:', simulator_step, r_max)
                    self.first_above_r[rmax] = simulator_step
                    curr_termination_dict[f'r.first_step_above.{str(rmax).replace(".", "_")}'] = simulator_step
                    self.termination_dict[f'r.first_step_above.{str(rmax).replace(".", "_")}'] = simulator_step
            
            if not found:
                break
        
        for dmax in sorted(self.ds_log):
            found = False
            
            if d_min < -dmax:
                found = True
                if dmax not in self.first_below_r:
                    print(f'First absolute bond divergence shorther with d {d_min} < -{dmax}:', simulator_step, d_min)
                    self.first_below_r[dmax] = simulator_step
                    curr_termination_dict[f'd.first_step_below.{str(dmax).replace(".", "_")}'] = simulator_step
                    self.termination_dict[f'd.first_step_below.{str(dmax).replace(".", "_")}'] = simulator_step
            
            if d_max > dmax:
                found = True
                if dmax not in self.first_above_r:
                    print(f'First absolute bond divergence longer with d {d_max} > {dmax}:', simulator_step, d_max)
                    self.first_above_r[dmax] = simulator_step
                    curr_termination_dict[f'd.first_step_above.{str(dmax).replace(".", "_")}'] = simulator_step
                    self.termination_dict[f'd.first_step_above.{str(dmax).replace(".", "_")}'] = simulator_step
            
            if not found:
                break
        
        
        for mmin in sorted(self.ms_log, reverse=True):
            if m < mmin:
                if mmin not in self.first_below_m:
                    self.first_below_m[mmin] = simulator_step
                    print(f"First step with m < {mmin}: {simulator_step}")
                    curr_termination_dict[f'm.first_step_below.{str(mmin).replace(".", "_")}'] = simulator_step
                    self.termination_dict[f'm.first_step_below.{str(mmin).replace(".", "_")}'] = simulator_step
            else:
                break
            
        msg_dict = {'termination': curr_termination_dict}
        error_msg = ''
        
        if self.k_termination is not None and k > self.k_termination:
            error_msg = f"Bond breaking detected at step {simulator_step} with k {k} > {self.k_termination}."
            msg_dict = {
                'state': 'md_error',
                'termination': {
                    'type': 'k',
                    'reason': f"k>{self.k_termination}",
                    'detail': error_msg,
                    **curr_termination_dict
                }
            }
            
        
        if self.r_termination is not None and r_min > self.r_termination:
            error_msg = f"Bond breaking detected at step {simulator_step} with r {r_min} > {self.r_termination}. Min: {r_min}, Max: {r_max}."
            msg_dict = {
                'state': 'md_error',
                'termination': {
                    'type': 'r',
                    'reason': f"r<-{self.r_termination}",
                    'detail': error_msg,
                    **curr_termination_dict
                }
            }
            
        
        if self.r_termination is not None and r_max > self.r_termination:
            error_msg = f"Bond breaking detected at step {simulator_step} with r {r_max} > {self.r_termination}. Min: {r_min}, Max: {r_max}."
            msg_dict = {
                'state': 'md_error',
                'termination': {
                    'type': 'r',
                    'reason': f"r>{self.r_termination}",
                    'detail': error_msg,
                    **curr_termination_dict
                }
            }
        
        if self.m_termination is not None and m < self.m_termination:
            error_msg = f"Bond breaking detected at step {simulator_step} with m {m} > {self.m_termination}."
            msg_dict = {
                'state': 'md_error',
                'termination': {
                    'type': 'm',
                    'reason': f"m<{self.m_termination}",
                    'detail': error_msg,
                    **curr_termination_dict
                }
            }
        
        return msg_dict, error_msg
        
    def on_step_end(self, simulator: Simulator):
        R = simulator.system.positions[0] / simulator.calculator.position_conversion 
        
        msg_dict, error_msg = self.gather_errors(R, simulator.step)
        simulator.log_message = msg_dict
        if error_msg:
            raise ValueError(error_msg)
        
    
    def get_config_dict(self):
        return dict(
            ks_log=self.ks_log,
            rs_log=self.rs_log,
            ds_log=self.ds_log,
            ms_log=self.ms_log,
        )

# Setup functions

def setup_system(atoms, temperature=300, Init=UniformInit):
    assert temperature >= 0
    # Create system instance and load molecule
    md_system = System()
    md_system.properties={}
    md_system.load_molecules(
        atoms,
        position_unit_input="Angstrom"
    )
    
    if Init is None:
        print('Note: Initializing form UniformInit, not MaxwellBoltzmannInit.')

    # Set up the initializer
    md_initializer = Init(
        temperature,
        remove_center_of_mass=True,
        remove_translation=True,
        remove_rotation=True,
    )

    # Initialize the system momenta
    md_initializer.initialize_system(md_system)
    
    return md_system

def setup_calculator(model, md=True, warmstart_configs=None, log_properties=['forw_iterations', 'back_iterations'], StatefulClass=StatefulAdamsBashforth, **kwargs):

    # pick Representation, Class based on if it's a deq
    try:
        # deq
        rep = model.representation.f
        CalculatorClass = make_stateful_calculator(StatefulClass=StatefulClass, md=md)

        CalculatorClass = partial(
            CalculatorClass,
            log_properties=log_properties,
            warmstart_configs=warmstart_configs,
        )

    except AttributeError:
        # access iterator
        rep = model.representation.stack_schnets[0]
        CalculatorClass = SchNetPackCalculator if md else SpkCalculator
    
    # set cutoff and buffer region
    cutoff = rep.cutoff_fn.cutoff.item()  # Angstrom (units used in model)
    cutoff_shell = 2.0  # Angstrom

    # initialize neighbor list for MD using the ASENeighborlist as basis
    if md:
        neighbor_list = NeighborListMD(
            cutoff,
            cutoff_shell,
            ASENeighborList,
        )
    else:
        neighbor_list = ASENeighborList(cutoff=cutoff)

    md_calculator = CalculatorClass(
        model_file=model,
        energy_key="energy",
        force_key="forces",
        energy_unit="kcal/mol",
        position_unit="Angstrom",
        neighbor_list=neighbor_list,
        **kwargs
    )
    return md_calculator

def setup_hdf5_logger(log_file=None, overwrite=True, log_properties=None, log_molecule=True):
    assert log_file is not None, "Need to provide path to log file."
    
    log_file += '.hdf5'
        
    # create log file directory if it does not exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
        
    # del log file if it exists, or continue on it, depending on user input:
    if os.path.exists(log_file):
        if overwrite in ('append', False):
            print(f"Appending to\n{log_file}")
        elif overwrite=='yes' \
        or (overwrite and input(f"Delete {log_file}? [y/N]") == 'y'):
            os.remove(log_file)
        else:
            print(f"Keeping\n{log_file}\nand returning without setting up logger.")
            return None

    # Size of the buffer
    buffer_size = 100

    # Set up data streams
    data_streams = log_molecule * [
        # store positions, momenta and the energy
        callback_hooks.MoleculeStream(store_velocities=True),
    ] + [
         # if None: let PropertyStream figure it our automatically, based on first steps results
        callback_hooks.PropertyStream(
            target_properties=None if log_properties is None else log_properties + [properties.energy, properties.forces]
        ),
    ]

    # Create the file logger
    file_logger = callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=1,  # logging frequency
        precision=32,  # floating point precision used in hdf5 database
    )
    
    return file_logger
    
def load_checkpoint(simulator, log_file, overwrite=False):
    if log_file is None: return
    log_file += '.chk'
    
    if overwrite in ('append', False):
        assert os.path.exists(log_file), f"Checkpoint {log_file} does not exist."
        checkpoint = torch.load(log_file)
        simulator.restart_simulation(checkpoint)
        # todo:
        assert False, "Currently not working, new positions are not written to loaded checkpoint file."
            
    elif overwrite:
        if os.path.exists(log_file):
            print(f"Overwriting checkpoint at {log_file}")
            os.remove(log_file)
    
    
## Tags

def short_name(name):
    return name.replace('_combPlus_InvUnitL2Feat_EquiUnitL2Feat', '')

# lru chached:
@lru_cache(maxsize=100)
def get_short_name(tag):
    api = wandb.Api()
    run = api.artifact(tag).logged_by()
    return short_name(run.name)

def warmstart_config_str(**warmstart_configs):
    possible_keys = ['z', 'g', 'g_bar']
    for k in warmstart_configs:
        if k not in possible_keys:
            raise ValueError(f"Invalid key in warmstart_configs: {k}")
    if not warmstart_configs or all(deg < 0 for deg in warmstart_configs.values()):
        return 'cold'
    s = 'warm'
    for k in possible_keys:
        deg = warmstart_configs.get(k)
        if deg is None or deg < 0:
            continue
        s += f'_{k}{deg}'
    return s

def load_update_desc_model(model_tag, model=None, turn_into_deq=False, device=None, inference_mode=False, **deq_config):
    """
    If not model is passed: loads the model from the artifact with the given tag.
    Updates the model with the given deq_config.
    Returns the model and a description of the model.
    """
    
    if model is None:
        model = load_model_from_wandb(model_tag, map_location='cpu')
        if device:
            model = model.to(device)
        
    if turn_into_deq:
        turn_iter_into_deq(model)
        
    if model.representation.__class__.__name__ == 'AtomisticDEQ':
        deq = model.representation.deq
        
        # note: in some very old models deq is a function that just calls model.representation
        # (backward compatibility hack: if type(deq) == type(lambda x: x): deq = model.representation)
        assert isinstance(deq, torch.nn.Module)
        
        
        # extract warm start config
        warmstart_configs = deq_config.pop('warmstart_configs', None)
            
        # update deq config
        deq.__dict__.update(deq_config)
        
        if inference_mode:
            assert deq_config.get('implicit_diff', True), "can't use inference mode without implicit_diff"
            deq = to_inference(deq)
            model.representation.deq = deq
            
        md_solving = solving_str(deq.implicit_diff, deq.forw_solver, deq.back_solver, deq.forw_tol, deq.back_tol, deq.forw_max_iter, deq.back_max_iter)
        if warmstart_configs is not None:
            md_solving += '_' + warmstart_config_str(**warmstart_configs)
    else:
        md_solving = 'normal'
    model_desc = f"{md_solving}__{get_short_name(model_tag)}__{model_tag.replace('/', '_')}"
    return model.eval(), model_desc

def desc_md(timestep, integrator=VelocityVerlet, temperature=None, initial_temperature=None, Init=UniformInit, langevin_time_constant=None):
    if not (langevin_time_constant or initial_temperature == temperature):
        "If no Langevin thermostat is used, and initial and target temperature are unequal, we assume that the potential energy makes up for the difference e.g. through a relaxation."
    name = integrator.__name__
    if name=='VelocityVerlet': name = 'VV'
    desc = f"MD__{name}"
    if temperature is not None:
        desc += f"_T{temperature:04d}K"
    desc += f"_dt{timestep:1.2f}fs"
    if temperature != initial_temperature:
        desc += f"_iT{initial_temperature:04d}K"
    if Init != UniformInit:
        desc += f'_{Init.__name__}'
    if langevin_time_constant:
        desc += f"_Langevin{langevin_time_constant:04d}fs"
    return desc

def desc_molecule(molecule_name, molecule_idx, fmax, stepsmax, model_config_relax=None, desc_equil=False, equil_steps=0, equil_md_config=None):
    a = f"{molecule_name}_{molecule_idx}"
    if model_config_relax is None:
        return a
    a += f"_fmax{fmax:.5f}_stepsmax{stepsmax:05d}"
    if not (desc_equil and equil_steps>0 and equil_md_config):
        return a
    a += f"__Equil_steps{equil_steps:05d}_{desc_md(**equil_md_config)}"
    return a


def get_atoms(molecule_name, molecule_idx, verbose=False):
    db_path = proj_path(f'data/{molecule_to_dataset(molecule_name)}/{molecule_name}.db', verbose=verbose)
    with connect(db_path) as db:
        row = db.get(molecule_idx)
    atoms = row.toatoms()
    atoms.info = row.data
    return atoms
        

def relax_atoms(atoms, model_config_relax, fmax, stepsmax, model, **_):
    # check if there are any other keys present
    has_config = model_config_relax.keys() > {'model_tag'}
    
    if has_config:
        model.representation.deq._tmp_config = model_config_relax

    atoms.calc = setup_calculator(model, md=False)
    
    optimizer = LBFGS(atoms)
    print('starting to relax...')
    optimizer.run(fmax=fmax, steps=stepsmax)
    print('relax done.')
    
    if has_config:
        model.representation.deq._tmp_config = None
    return atoms

def equilibrate_atoms(atoms, model_config_relax, model, equil_steps, equil_md_config, Init=UniformInit, hooks=None, **_):
    # check if there are any other keys present
    has_config = model_config_relax.keys() > {'model_tag'}

    if has_config:
        model.representation.deq._tmp_config = model_config_relax

    md_simulator, md_system = setup_md(model, atoms, equil_md_config, Init=Init, warmstart_configs=model_config_relax.get('warmstart_configs'), log_properties=[], hooks=hooks)
    md_simulator.simulate(equil_steps)
    
    if has_config:
        model.representation.deq._tmp_config = None
    return md_system

def get_next_path(base_path):
    base = Path(base_path)
    mm_dd = time.strftime("%m-%d")
    for _ in range(100):
        suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        candidate = base / f"{mm_dd}-{suffix}"
        if not candidate.exists():
            return candidate

def setup_md(model, atoms, md_config, log_properties=None, hooks=None, progress=True, device=None, StatefulClass=None, stateful_kwargs=None, Init=UniformInit, **calculator_kwargs):
    """
    md_config['initial_temperature']: Temperature which the atoms geometry likely had at generation (500K for MD17).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    hooks = hooks or []
    IntegratorClass = md_config.get('integrator', VelocityVerlet)

    if isinstance(atoms, Atoms) or (isinstance(atoms, list) and all(isinstance(a, Atoms) for a in atoms)):
        # if potential enery "contains" already a different amount than desired, scale initial kinetic energy up/down accordingly:
        init_with_temperature = 2*md_config['temperature']-md_config['initial_temperature']
        if init_with_temperature < 0:
            print(f"Warning: can not down-regulate to {md_config['temperature']} from {md_config['initial_temperature']=}. Initializing with 0K, thus the system contains ~{md_config['initial_temperature']/2}K of energy initially. A thermostat must be used to reach {md_config['temperature']}.")
            init_with_temperature = 0
        md_system = setup_system(atoms=atoms, temperature=init_with_temperature, Init=Init)
    else:
        assert isinstance(atoms, System) or isinstance(atoms, list) and all(isinstance(a, System) for a in atoms), \
            "Atoms must be an instance of ASE Atoms or SchNetPack System."
        md_system = atoms

    # omit the logging from md calculator init (removing psot processing etc)
    org_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)
    calculator_kwargs.setdefault('md', True)
    if StatefulClass is not None:
        calculator_kwargs['StatefulClass'] = StatefulClass
    if stateful_kwargs is not None:
        calculator_kwargs.update(stateful_kwargs)
    md_calculator = setup_calculator(model=model, **calculator_kwargs, log_properties=log_properties)
    logging.getLogger().setLevel(org_level)
    
    # setup MD simulator
    if ltc:=md_config.get('langevin_time_constant'):
        thermostat = LangevinThermostat(temperature_bath=md_config['temperature'], time_constant=ltc)
        hooks.append(thermostat)
        
    md_simulator = Simulator(
        md_system,
        IntegratorClass(time_step=md_config['timestep']),
        md_calculator,
        simulator_hooks=hooks,
        progress=progress
    )
    md_simulator = md_simulator.to(device)
    return md_simulator, md_system

def bring_to_thermal_equilibrium(atoms, model, equilibrate_strategy, target_temperature, initial_temperature, Init=UniformInit, **_):
    """
    Bring atoms to thermal equilibrium using the specified strategy.
    """

    if equilibrate_strategy == 1:
        time_constants = [
            50, 
            200, 
            1000
            ]
        model_config = dict(
            implicit_diff=True,
            forw_tol=.0001,
            back_tol=.0001,
            forw_solver=fixed_point_iteration,
            back_solver=fixed_point_iteration,
            forw_max_iter=10,
            back_max_iter=10,
            warmstart_configs=dict(z=1, g_bar=1),
        )  
    else:
        raise ValueError(f"Unknown thermal equilibrium strategy: {equilibrate_strategy}")
    
    
    md_config = dict(
        timestep=0.5,
        integrator=VelocityVerlet,
        temperature=target_temperature,
        initial_temperature=initial_temperature
    )
    md_simulator, md_system = setup_md(
        model, atoms, md_config,
        Init=Init,
        warmstart_configs=model_config.get('warmstart_configs'),
    )
    
    model.representation.deq._tmp_config = model_config
    for ltc in time_constants:
        thermostat = LangevinThermostat(temperature_bath=target_temperature, time_constant=ltc)
        md_simulator.simulator_hooks = torch.nn.ModuleList([thermostat])
        md_simulator.simulate(2*ltc)
    model.representation.deq._tmp_config = None
    
    return md_system

def load_system(path):
    system = System()
    system.nm_transform = system._nm_transformer(1)
    system.load_state_dict(torch.load(path))
    system.n_replicas = 1
    system.n_molecules = 1
    system.total_n_atoms = system.positions.shape[1]
    system.eval()
    # system.positions = system.positions.to(torch.float32)
    # system.momenta = system.momenta.to(torch.float32)
    
    # Ensure float32 precision
    for name, buf in system.named_buffers():
        if buf.dtype.is_floating_point:
            setattr(system, name, buf.to(torch.float32))

    # for name, param in system.named_parameters():
    #     if param.dtype.is_floating_point:
    #         setattr(system, name, torch.nn.Parameter(param.to(torch.float32)))
    
    return system
        
        
def load_relax_equil_atoms(model, relax_path, equil_path, equil_hooks=None, **molecule_config):
    
    relax = molecule_config.get('model_config_relax') is not None
    equil = relax and molecule_config.get('equil_steps', 0) > 0 and molecule_config.get('equil_md_config') is not None
    
    if not relax: relax_path = None
    if not equil: equil_path = None
    
    relax_lock = FileLock(relax_path + '.lock') if relax_path else nullcontext()
    equil_lock = FileLock(equil_path + '.lock') if equil_path else nullcontext()
    
    print('Aquiring Equilibration lock...')
    with equil_lock:
        # try to load equilibrated system
        if equil_path and os.path.exists(equil_path):
            system = load_system(equil_path)
            return system
        
        print('Aquiring Relaxation lock...')
        with relax_lock:
            # try to load relaxed atoms
            if relax_path and os.path.exists(relax_path):
                atoms = read(relax_path)
            else:
                # load unrelaxed atoms
                atoms = get_atoms(molecule_config['molecule_name'], molecule_config['molecule_idx'])
                if relax:
                    # relax atoms, save
                    print("Relaxing atoms...")
                    atoms = relax_atoms(atoms=atoms, model=model, **molecule_config)
                    if relax_path:
                        os.makedirs(os.path.dirname(relax_path), exist_ok=True)
                        write(relax_path, atoms)
                        
        if equil:
            # equilibrate atoms, save
            print("Equilibrating atoms...")
            system = equilibrate_atoms(atoms=atoms, model=model, hooks=equil_hooks, **molecule_config)
            if equil_path:
                os.makedirs(os.path.dirname(equil_path), exist_ok=True)
                torch.save(system.state_dict(), equil_path)
            return system
                
        return atoms
        
def load_system_equilibrated(model, equil_path, **molecule_config):
    if equil_path and os.path.exists(equil_path):
        system = load_system(equil_path)
    else:
        atoms = get_atoms(molecule_config['molecule_name'], molecule_config['molecule_idx'])

        system = bring_to_thermal_equilibrium(atoms=atoms, model=model, **molecule_config)
                
        if equil_path:
            os.makedirs(os.path.dirname(equil_path), exist_ok=True)
            torch.save(system.state_dict(), equil_path)
    return system
            
def setup_filesystem_hooks(log_file, overwrite, log_properties=['forw_iterations']):
    return [
        setup_hdf5_logger(log_file, overwrite, log_properties=log_properties),
        callback_hooks.Checkpoint(f"{log_file}.chk", every_n_steps=100),
    ]