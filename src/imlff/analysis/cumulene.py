import numpy as np
import matplotlib.pyplot as plt

from imlff.analysis.cumulene_loading import (
    cached_test_loader,
    load_prep_model,
    get_energy_offset,
    get_molecule_train_mean,
    get_n_a,
    calc_E_pred,
    read_E_label,
    load_E_pred,
    save_E_pred,
    load_E_label,
    save_E_label,
    load_or_calc,
    load_or_calc_mae,
    load_or_read_out_label,
)

from matplotlib import colormaps

def style(n_lines=9, cmap='cividis'):
    plt.rcdefaults()
    cmap = colormaps.get_cmap(cmap)
    try:
        colors = cmap.colors
        every = len(colors) // n_lines
        colors = colors[::every]
    except AttributeError as e: # 'LinearSegmentedColormap' object has no attribute 'colors'
        if str(e) != "'LinearSegmentedColormap' object has no attribute 'colors'":
            raise e
        colors = [cmap(i / (n_lines - .9)) for i in range(n_lines)]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['grid.color'] = 'lightgray'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['legend.borderaxespad'] = 0.5
    plt.rcParams['legend.borderpad'] = 0.5
    plt.rcParams['legend.labelspacing'] = 0.5
    plt.rcParams['legend.handlelength'] = 1.5
    plt.rcParams['legend.handleheight'] = 0.5
    plt.rcParams['legend.columnspacing'] = 1.0
    plt.rcParams['axes.titlepad'] = 10
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['xtick.major.pad'] = 5
    plt.rcParams['ytick.major.pad'] = 5
    plt.rcParams['xtick.minor.pad'] = 5
    plt.rcParams['ytick.minor.pad'] = 5
    plt.rcParams['axes.xmargin'] = 0.05
    
def annotate(ax, ylabel, ticks='normal'):
    ax.set_xlabel('Torsion angle')
    ax.set_ylabel('rel. ' * (ylabel=='diff') + 'Energy (eV)' * bool(ylabel))
    
    ax.set_xticks([0, 499, 999])
    ts = [0, 45, 90]
    if ticks=='inverted':
        ts = ts[::-1]
    ts = [fr'{t}$^\circ$' for t in ts]
    if 'relative' in ticks:
        ts = ['+'+t for t in ts]
        if 'relaxed' in ticks:
            ts[0] = 'Relaxed'
    tl = ax.set_xticklabels(ts)
    if tl:
        tl[0].set_ha('left')
        tl[-1].set_ha('right')
        # tl[0].set_x(0.5)
        # tl[0].set_x(0.95)

def get_model_tag(Rep='SO3Net', type='imp', model_i=(1,2,3,4,5), cutoff=5., latest=False, norm='auto', train_forces=0.95, train_dft=False, jac_reg=0., verbose=False):
    print(f'get_model_tag called with params: {Rep=}, {type=}, {model_i=}, {cutoff=}, {latest=}, {norm=}, {train_forces=}, {train_dft=}, {jac_reg=}')
    ret = _get_model_tag(Rep=Rep, type=type, model_i=model_i, cutoff=cutoff, latest=latest, norm=norm, train_forces=train_forces, train_dft=train_dft, jac_reg=jac_reg,)
    print(f'get_model_tag called with params: {Rep=}, {type=}, {model_i=}, {cutoff=}, {latest=}, {norm=}, {train_forces=}, {train_dft=}, {jac_reg=} --> {ret}')
    return ret

def _get_model_tag(Rep='SO3Net', type='imp', model_i=(1,2,3,4,5), cutoff=5., latest=False, norm='auto', train_forces=0.95, train_dft=False, jac_reg=0.):
    
    if train_forces==0.:
        if type=='imp':
            assert norm in [True, 'auto']
        else:
            assert norm in [False, 'auto']
        assert Rep == 'SO3Net'
        
        assert model_i==(1,2,3,4,5,6,7,8)
        if train_dft:
            if cutoff == 5.:
                print('Warning: DFT data was bad')
                if type=='imp':
                    return 'qcml-group/dev-v10/model-msgwukz2:v339'
                elif type=='exp-3':
                    return 'qcml-group/dev-v10/model-9wgyhup3:v459'
        else:
            if cutoff == 5.:
                if type=='imp':
                    if jac_reg==0.32:
                        print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5,6,7,8), cutoff=5., kl0u9s9l)')
                        return 'qcml-group/dev-v10/model-kl0u9s9l:latest' # latest

                    
            if cutoff == 12.:
                if type=='imp':
                    pass
    
    if train_forces == 0.95:
        if train_dft:
            print('Warning: DFT data was bad')
            assert model_i==(1,2,3,4,5,6,7,8)
            assert Rep == 'SO3Net'
            if type=='imp':
                assert norm in [True, 'auto']
                return 'qcml-group/dev-v10/model-agid5lrv:v131'
            else:
                assert norm in [False, 'auto']
                return 'qcml-group/dev-v10/model-bu6ulg15:v99'
        else:
            if Rep == 'PaiNN':
                if type=='imp':
                    if cutoff == 5.:
                        if model_i==1: return 'qcml-group/dev-v10-foo/model-isgi42be:v2' # c1 local
                        if model_i==8: return 'qcml-group/dev-v10/model-punteq4o:v3' # c8 40 epoch
            if Rep == 'SO3Net':
                if type=='imp' and norm in [False]:
                    if cutoff == 5.:
                        if model_i==1: 
                            return 'qcml-group/dev-v10/model-u6gomooo:v166'
                        if model_i==8: 
                            return 'qcml-group/dev-v10/model-bo3mhfdq:v59'
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-dv8u80bc:v149' # latest
                            return 'qcml-group/dev-v10/model-dv8u80bc:v19' # best
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-h5syv45h:v1' # also interesting!
                    if cutoff == 1.75:
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-uc5zoib2:v40' # latest
                        if model_i==8: return 'qcml-group/dev-v10/model-j22tvlj9:v266'
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-zt6mshh3:v40' # latest
                            return 'qcml-group/dev-v10/model-zt6mshh3:v19' # best
                    if cutoff == 12.:
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-o3um7gs9:v61' # latest
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-900skidk:v119' # latest
                            return 'qcml-group/dev-v10/model-900skidk:v19' # best
                
                if type=='imp' and norm in ['auto', True]:
                    if cutoff == 5.:
                        if model_i==1: 
                            return ''
                        if model_i==8: 
                            return ''
                        if model_i==(1,3,6,8):
                            if latest: return '' # latest
                            return '' # best
                        if model_i==(1,2,3,4,5):
                            if jac_reg==0.32:
                                print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5), cutoff=5., 6ikwowuc)')
                                return 'qcml-group/dev-v10/model-6ikwowuc:latest' # latest
                            return 'qcml-group/dev-v10/model-c0waqlxc:v87'
                            return 'qcml-group/dev-v10/model-c0waqlxc:v54'
                            return 'qcml-group/dev-v10/model-c0waqlxc:v3'
                            if latest: return '' # latest
                            return '' # best
                        if model_i==(1,2,3,4,5,6,7,8):
                            if jac_reg==0.032:
                                print('Info: returning model with jac_reg=0.032 (model_i=(1,2,3,4,5,6,7,8), cutoff=5., ztvrp5g0)')
                                return 'qcml-group/dev-v10/model-ztvrp5g0:latest' # latest
                            if jac_reg==0.32:
                                print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5,6,7,8), cutoff=5., l87qqzve)')
                                return 'qcml-group/dev-v10/model-l87qqzve:latest' # latest
                            return 'qcml-group/dev-v10/model-ynuotx5u:latest'
                    if cutoff == 1.75:
                        if model_i==(1,2,3,4,5,6,7,8):
                            return '' # latest
                        if model_i==(1,2,3,4,5):
                            if jac_reg==0.32:
                                print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5), cutoff=1.75, n6hbbe5x)')
                                return 'qcml-group/dev-v10/model-n6hbbe5x:latest' # latest
                            if latest: return '' # latest
                            return '' # best
                        if model_i==(1,3,6,8):
                            if latest: return '' # latest
                            return '' # best
                    if cutoff == 12.:
                        if model_i==(1,2,3,4,5,6,7,8):
                            if jac_reg==0.32:
                                print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5,6,7,8), cutoff=12., 6lzpgt69)')
                                return 'qcml-group/dev-v10/model-6lzpgt69:latest' # latest
                        if model_i==(1,2,3,4,5):
                            if jac_reg==0.32:
                                print('Info: returning model with jac_reg=0.32 (model_i=(1,2,3,4,5), cutoff=12., 18mj6p3w)')
                                return 'qcml-group/dev-v10/model-18mj6p3w:latest' # latest
                            if latest: return '' # latest
                            return '' # best
                        if model_i==(1,3,6,8):
                            if latest: return '' # latest
                            return '' # best
                if type == 'exp-6' and norm in ['auto', False]:
                    if cutoff == 5.:
                        if model_i==(1,2,3,4,5):
                            return 'qcml-group/dev-v10/model-d475qr8b:v19' # best
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-xyociq3d:v59' # best
                if type == 'exp-3' and norm in ['auto', False]:
                    if cutoff == 5.:
                        if model_i==1: 
                            print('did not check run: m0vsdsa4')
                            return 'qcml-group/dev-v10/model-m0vsdsa4:v19'
                        if model_i==8: 
                            print('did not check run: b6kllp9b')
                            return 'qcml-group/dev-v10/model-b6kllp9b:v120'
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-erq83elv:v142' # latest
                            return 'qcml-group/dev-v10/model-erq83elv:v59' # best
                        if model_i==(1,3,6,8):
                            if latest: return 'qcml-group/dev-v10/model-wemwx3rq:v445' # latest
                            return 'qcml-group/dev-v10/model-wemwx3rq:v59' # best
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-zgltzvyp:best'
                    if cutoff == 1.75:
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-y4458g3d:v241' # latest
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-p8p8tgx9:v271' # latest
                            return 'qcml-group/dev-v10/model-p8p8tgx9:v159' # best
                        if model_i==(1,3,6,8):
                            if latest: return 'qcml-group/dev-v10/model-1s5ptcbl:v439' # latest
                            return 'qcml-group/dev-v10/model-1s5ptcbl:v99' # best
                    if cutoff == 12.:
                        if model_i==(1,2,3,4,5,6,7,8):
                            return 'qcml-group/dev-v10/model-jjktmo8d:v111' # latest
                        if model_i==(1,2,3,4,5):
                            if latest: return 'qcml-group/dev-v10/model-b5znar15:v350' # latest
                            return 'qcml-group/dev-v10/model-b5znar15:v59' # best
                            return 'qcml-group/dev-v10/model-b5znar15:v9'
                        if model_i==(1,3,6,8):
                            if latest: return 'qcml-group/dev-v10/model-9h0hyjmo:v445' # latest
                            return 'qcml-group/dev-v10/model-9h0hyjmo:v59' # best
                
## Plotting ##

def get_Z(molecule_i):
    return ['C', 'H', 'H'] + ['C'] * (molecule_i - 1) + ['H', 'H', 'C']

def get_indices_sel(Z):
    # swap H and C indices for better readability:
    indices = np.arange(len(Z))
    indices[[0,1]] = [1, 0]
    # exclude second half, and second H, as they are symmetric
    half_of_chain = (len(Z)+1) // 2
    # half_of_chain += 1
    sel_mask = indices < half_of_chain
    sel_mask[2] = False
    
    return indices[sel_mask]

def plot_E_per_molecule(
    models=None,
    molecule_is=(1,2,3,4,5,6,7,8),
    substract_start=True,
    plot_label=True,
    tied=True,
    cmap='cividis',
    recalc=False,
    axs=None,
    legend=True,
    train_dft=False,
    add_offsets=True,
    linestyle_kwargs=None,
    color_kwargs=None,
    **model_kwargs
    ):
    defaults = dict(
        Rep='SO3Net',
        type='imp',
        cutoff=5.,
        model_i=(1,2,3,4,5),
        norm='auto',
    )
    defaults.update(model_kwargs)
    if not models:
        models = [
            dict(type='exp-3'),
            dict(type='imp'),
        ]
    def p(model):
        m = defaults.copy()
        m.update(model)
        typ = model.get('type') or model.get('Rep')
        if typ in ['HF', 'DFT']:
            m['type'] = m['Rep'] = typ
        return m
    models = [p(model) for model in models]

    # global per-atom mean that AddOffsets added back at inference (None for non-NN baselines)
    offsets = [None if model['Rep'] in ('HF', 'DFT') else get_energy_offset(**model)
               for model in models] if add_offsets else [None] * len(models)

    style(len(models)+2, cmap=cmap)
    n = len(molecule_is)
    if axs is None:
        fig, axs = plt.subplots(1, n, figsize=(n * 4, 4), sharey=False)
    else:
        assert len(axs) == n, f'axs has {len(axs)} axes, but expected {n}'
    
    if tied:
        for ax in axs[1:]:
            ax.sharey(axs[0])
            # make ticks invisible:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    for j, (molecule_i, ax) in enumerate(zip(molecule_is, axs)):

        if substract_start and add_offsets:
            # offsets put all curves on the label's absolute scale, so subtract the
            # label's start from every curve to keep their relative shifts visible
            start = load_or_read_out_label(molecule_i, dft=train_dft)[0]
            s = lambda E: E - start
        else:
            s = lambda E: E - substract_start * E[0]

        is_ood = lambda model_config: molecule_i not in model_config['model_i'] and model_config['Rep'] != 'HF'
        
        ylabel = True
        ticks = 'normal'
        if molecule_i%2==0:
            ticks = 'inverted'
        if substract_start:
            ylabel = 'diff'
        if j>0:
            ylabel = False
        annotate(ax=ax, ylabel=ylabel, ticks=ticks)
        
        
        if plot_label:
            E_label = load_or_read_out_label(molecule_i, dft=train_dft)
            E_label = s(E_label)
            ax.plot(E_label, label='GFN2-xTB\n(label)', linestyle='-', zorder=-10, color='black')
            
        ax.set_title(fr'$C_{{{molecule_i+1}}}H_4$' + is_ood(models[-1]) * ' (O.O.D.)')
        
        for i, model in enumerate(models):
            print('BEFORE load_or_cals:')
            print(f'molecule_i={molecule_i}, recalc={recalc}, {model}')
            E, E_per_atom = load_or_calc(molecule_i=molecule_i, recalc=recalc, **model)
            if offsets[i] is not None:
                # training removed this molecule's own mean; AddOffsets restored only the
                # global average — swap the global mean for the per-molecule one
                mean_i = get_molecule_train_mean(molecule_i, dft=train_dft)
                E = E - (offsets[i] - mean_i) * len(get_Z(molecule_i))

            _linestyle_kwargs = {
                'imp': {'linestyle': '--', 'zorder': 10},
                'exp-3': {'linestyle': ':', 'zorder': 20},
                'exp': {'linestyle': '-.', 'zorder': 20},
            }
            if linestyle_kwargs is not None:
                _linestyle_kwargs.update(linestyle_kwargs)

            _color_kwargs = {
                1.75: {'color': '#d62728'},
                5:    {'color': '#1f77b4'},
                12:   {'color': '#bcbd22'},
            }
            if color_kwargs is not None:
                _color_kwargs.update(color_kwargs)

            mtype = model['type']
            if 'imp' in mtype:
                ls_kw = _linestyle_kwargs['imp']
            elif mtype == 'exp-3':
                ls_kw = _linestyle_kwargs['exp-3']
            else:
                ls_kw = _linestyle_kwargs['exp']

            cut_val = model["cutoff"]
            color_kw = _color_kwargs.get(cut_val, {})

            label = model['type']
            if ('imp' in label) or ('exp' in label):
                if 'imp' in label:
                    label = 'Implicit'
                elif 'exp' in label:
                    depth = label.split('-')[-1]
                    label = 'Explicit'
                    if depth != '3':
                        label += f'-{depth}'
                cut = str(cut_val).rstrip('0').rstrip('.')
                label += f' ({cut}Å)'
            E = s(E)
            ax.plot(E, label=label, **ls_kw, **color_kw)
        
    if legend: axs[0].legend()

def mae_per_molecule(
    models=None,
    molecule_is=(1, 2, 3, 4, 5, 6, 7, 8),
    recalc=False,
    train_dft=False,
    add_offsets=True,
    **model_kwargs,
):
    """Per-molecule energy & force MAE for each model in `models`.

    Mirrors plot_E_per_molecule's model normalization and energy-offset handling.
    Returns a long-form pandas DataFrame with columns:
    Rep, type, cutoff, jac_reg, molecule_i, energy_MAE, forces_MAE.
    HF/DFT baselines are skipped (no force pipeline). Units match the dataset:
    eV for energy, eV/Ang for forces (the cumulene DB stores eV, no conversion applied).
    """
    import pandas as pd

    defaults = dict(
        Rep='SO3Net',
        type='imp',
        cutoff=5.,
        model_i=(1, 2, 3, 4, 5),
        norm='auto',
    )
    defaults.update(model_kwargs)
    if not models:
        models = [dict(type='exp-3'), dict(type='imp')]

    def p(model):
        m = defaults.copy()
        m.update(model)
        typ = model.get('type') or model.get('Rep')
        if typ in ['HF', 'DFT']:
            m['type'] = m['Rep'] = typ
        return m

    models = [p(model) for model in models]

    rows = []
    for model in models:
        if model['Rep'] in ('HF', 'DFT'):
            print(f"Skipping {model['Rep']} (no force MAE pipeline)")
            continue
        for molecule_i in molecule_is:
            e_mae, f_mae = load_or_calc_mae(
                molecule_i=molecule_i, recalc=recalc, train_dft=train_dft,
                add_offsets=add_offsets, **model,
            )
            rows.append(dict(
                Rep=model['Rep'], type=model['type'], cutoff=model['cutoff'],
                jac_reg=model.get('jac_reg'), molecule_i=molecule_i,
                energy_MAE=e_mae, forces_MAE=f_mae,
            ))
    return pd.DataFrame(rows)


def plot_E_per_atom(model_i=(1,2,3,4,5,6,7,8),
                    molecule_is=(1,2,3,4,5,6,7,8),
                    substract_start=True,
                    cmap='cividis',
                    recalc=False,
                    Rep='SO3Net',
                    axs = None,
                    legend = False,
                    molecular=False,
                    sharey=True,
                    linestyle = None,
                    color = None,
                    **model):
    style(len(molecule_is), cmap=cmap)
    n = len(get_indices_sel(get_Z(max(molecule_is)))) + bool(molecular)
    if axs is not None:
        assert len(axs) == n, f'axs has received {len(axs)} axes, but expected {n}'
    else:
        fig, axs = plt.subplots(1, n, figsize=(n * 4, 4), sharey=False)
    axs = axs if molecular else [None, *axs]
        
    if sharey:
        for ax in axs[2:]:
            ax.sharey(axs[1])
    
    if molecular:
        pos = axs[0].get_position() 
        if molecular=='end':
            pos_last = axs[-1].get_position()
            x0 = pos_last.x0 + pos_last.width + 0.02
        else:
            x0 = pos.x0-.02-0.015*bool(substract_start)
        axs[0].set_position([x0, pos.y0, pos.width, pos.height])

    for molecule_i in molecule_is:
        if Rep=='HF':
            num = 1000
            xindices = np.linspace(0, 999, num=num, dtype=int)
        else:
            xindices = np.arange(1000)

        E, E_per_atom = load_or_calc(Rep=Rep, model_i=model_i, molecule_i=molecule_i, recalc=recalc, xindices=xindices, **model)
         
        if linestyle is None:
            in_dist = molecule_i == model_i or (isinstance(model_i, tuple) and molecule_i in model_i)
            linestyle = '-' if in_dist else ':'
        
        Z = get_Z(molecule_i)
        indices_sel = get_indices_sel(Z)
        
        label = fr'$C_{{{molecule_i+1}}}H_4$'
           
        # print(label, E.shape, E_per_atom.shape)
        
        if type(color) in [list, tuple, np.ndarray]:
            color_curr = color[molecule_i-1]
        else:
            color_curr = color
            
        if molecular:
            axs[0].plot(xindices, E - (substract_start=='all') * E[0], label=label, linestyle=linestyle, color=color_curr)
        
        for i, ax in enumerate(axs[1:]):
            if i<len(indices_sel):
                idx = indices_sel[i]
                dat = E_per_atom[:, idx] - bool(substract_start) * E_per_atom[0, idx]
                z = Z[idx]
                # ax.plot(xindices, dat, linestyle=linestyle or ('--' if z == 'H' else '-'), label=label)
            else:
                xindices = dat = [np.nan]
                # dat = 0
            # label = f'{z} ({idx})'
            # if len(E_per_atom_dict) > 1:
            #     label += f' - {label_method}'
            ax.plot(xindices, dat, linestyle=linestyle, color=color_curr, label=label)
        
    titles = ['Energy of molecule', 'Hydrogen', '1st Carbon in chain', '2nd Carbon in chain', '3rd Carbon in chain', '4th Carbon in chain', '5th Carbon in chain']
    for ax, title in zip(axs[not molecular:], titles[not molecular:]):
        ax.set_title(title)
        annotate(ax, ticks='relative', ylabel=None)
        
    if substract_start=='all':
        annotate(axs[1 - bool(molecular)], ticks='relative relaxed', ylabel='diff')
    elif substract_start:
        if molecular: annotate(axs[0], ticks='relative relaxed', ylabel=True)
        annotate(axs[1], ticks='relative', ylabel='diff')
    else:
        annotate(axs[not molecular], ticks='relative relaxed', ylabel=True)

    for ax in axs[2:]:
        # don't show y ticks:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    if legend:
        handles, labels = axs[1].get_legend_handles_labels()
        axs[-1].legend(handles, labels)
    
    return axs
