import torch

from schnetpack.interfaces.ase_interface import SpkCalculator
from schnetpack.md.calculators.schnetpack_calculator import SchNetPackCalculator
from schnetpack.md import System

from imlff.md.stateful_extrapolator import StatefulAdamsBashforth

# ===========================================================================
# Mixing Stateful extrapolator into SchnetPack
# ===========================================================================

def make_stateful_calculator(md=True, StatefulClass=StatefulAdamsBashforth):
    if md:
        class _StatefulMDCalculator(SchNetPackCalculator, StatefulClass):

            results_key = 'results'

            def __init__(self, *args, log_properties=['forw_iterations', 'back_iterations'], add_to_batch=dict(), **kwargs):

                kwargs_spk = {
                    k: kwargs.pop(k) for k in (
                        'model_file',
                        'force_key',
                        'energy_unit',
                        'position_unit',
                        'neighbor_list',
                        'energy_key',
                        'stress_key',
                        # 'required_properties',
                        'property_conversion',
                        'script_model',
                    ) if k in kwargs
                }

                # call Stateful constructor
                StatefulClass.__init__(self, **kwargs)

                # call SchNetPackCalculator constructor
                SchNetPackCalculator.__init__(self, *args, required_properties=log_properties or [], **kwargs_spk)

                self.add_to_batch = add_to_batch # override forw_tol, back_tol, ...

            def _generate_input(self, system):
                base = {**super()._generate_input(system), **self.add_to_batch}
                return {**base, **self.get_state()}

            def calculate(self, system: System):
                inputs = self._generate_input(system)
                self.results = self.model(inputs)
                self._update_system(system)
                self.save_state()

            def _update_system(self, system: System):
                super()._log_required_properties(system)
                self._log_dic_properties(system)

                # setting force prediction
                super()._set_predicted_properties(system)

            def _log_dic_properties(self, system: System):
                # split dictonaries into scalars for hdf5 logging
                log_properties_split = [('forw_res', 10, torch.nan), ('back_res', 10, torch.nan)]

                # update the results
                for prop, up_to, default in log_properties_split:
                    if prop not in self.results:
                        continue
                    for i in range(1, up_to + 1):
                        val = self.results[prop].get(i)
                        if val is None:
                            val = torch.tensor([[default]])
                        val = val.mean().view(system.n_replicas, 1)
                        system.properties[f"{self.log_prefix}{prop}_{i:02d}"] = val

        return _StatefulMDCalculator

    else:
        class _StatefulCalculator(SpkCalculator, StatefulClass):

            results_key = 'model_results'

            def __init__(self, *args, log_properties=None, **kwargs):
                stateful_kwargs = {
                    k: kwargs.pop(k) for k in ('warmstart_configs', 'log_prefix') if k in kwargs
                }

                # call Stateful constructor
                StatefulClass.__init__(self, **stateful_kwargs)

                # call SpkCalculator constructor
                SpkCalculator.__init__(self, *args, **kwargs)

                # modify converter to include state as it is called within SpkCalculator.calculate
                self.converter_obj = self.converter
                self.converter = lambda atoms: dict(
                    **self.converter_obj(atoms),
                    **self.get_state()
                )

            def calculate(self, *args, **kwargs):
                super().calculate(*args, **kwargs)
                self.save_state()

        return _StatefulCalculator
