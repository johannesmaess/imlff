import os, shutil, getpass
from types import SimpleNamespace

def collect_config(*configs, config_elements=None):
    """
    Pass a number of dicts or objects with attributes and a list of elements you want to collect.
    The first dict that contains the element will provide the value for the config.
    Usecase: Pass a dictionary that contains overrides, next an object whose members are the defaults.
    """
    assert config_elements is not None, "Please provide the config_elements you require"
    config = SimpleNamespace()
    for k in config_elements:
        for c in configs:
            if not isinstance(c, dict):
                # assuming c is an object
                c = vars(c)
                
            # asssuming c is a dict
            if k in c:
                setattr(config, k, c[k])
                break
        else:
            # if no config was found, set to None
            setattr(config, k, None)
    return config

def proj_root_path():
    # assert /imlff is in path and cut everything behind it
    ident = '/imlff'
    path = os.getcwd()
    try:
        idx = path.index(ident)
    except ValueError:
        raise ValueError("Please name your project dir '{ident}', and run scripts from inside of it. This makes path resolution easier.")
    path = path[:idx + len(ident)]
    return path

# The following setup assumes that we are in an apptainer environment started with --bind /temp:/temp-in-container
# The tmp subdir is per-user: /temp is shared node-local scratch, and files/dirs staged
# there by one user are not writable/deletable by another (dynamic_sync would crash).
DEFAULT_TMP_DIR_IN_CONTAINER, DEFAULT_TMP_SUBDIR = '/temp-in-container', f'imlff-{getpass.getuser()}'
# DEFAULT_TMP_DIR_IN_CONTAINER, DEFAULT_TMP_SUBDIR = '/tmp', 'for-local-testing'
DIRS_TO_USE_TMP = ['data']

def proj_path(*dirs, use_tmp='auto',
              only_if_tmp_dir_exists=True,
              with_sync=True,
              tmp_dir=DEFAULT_TMP_DIR_IN_CONTAINER,
              tmp_subdir=DEFAULT_TMP_SUBDIR,
              verbose=True
              ):
    """_summary_
    Returns the path to the project root + dirs.
    Has options to sync into /temp first, for faster reads on the cluster.
    """
    
    relative_path = os.path.join(*dirs) if dirs else ''
    assert not relative_path.startswith('/'), "Please provide relative paths"
    
    if use_tmp == 'auto':
        main_dir = relative_path.split('/')[0]
        use_tmp = main_dir in DIRS_TO_USE_TMP
    
    not_in_tmp = os.path.join(proj_root_path(), relative_path)
    in_tmp = os.path.join(tmp_dir, tmp_subdir, relative_path)
    
    if not use_tmp:
        return not_in_tmp   # user does not want to use tmp (for this dir)
    elif only_if_tmp_dir_exists and not os.path.exists(tmp_dir):
        if verbose: print(f"Note: {tmp_dir} does not exist. Not using tmp.")
        return not_in_tmp   # we are not in the container
    elif not with_sync:
        return in_tmp       # user wants to use tmp, but does not want to sync (maybe dir does not exist)
    else:
        dynamic_sync(not_in_tmp, in_tmp, verbose=verbose)
        return in_tmp       # sync first, then return path

def dynamic_sync(src, dest, verbose=True):
    if verbose: print(f'Starting copy from \n{src} \nto \n{dest}')

    if os.path.isfile(src):
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Copy only if the source is newer, size differs, or destination doesn't exist
        if (not os.path.exists(dest)
                or os.path.getmtime(src) > os.path.getmtime(dest)
                or os.path.getsize(src) != os.path.getsize(dest)):
            if os.path.exists(dest):
                os.remove(dest)
            shutil.copy2(src, dest)
            if verbose: print(f"Copied file: {src} -> {dest}")
        else:
            if verbose: print(f"Skipped file (up-to-date): {src}")
    elif os.path.isdir(src):
        os.makedirs(dest, exist_ok=True)
        for entry in os.listdir(src):
            dynamic_sync(os.path.join(src, entry), os.path.join(dest, entry), verbose)
    else:
        raise ValueError(f"Source {src} is neither a file nor a directory")