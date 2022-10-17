import os
import json
import itertools


env_conf_file = 'env.json'
models_dir = 'models'
checkpoints_dir = 'checkpoints'
checkpoint_prefix = 'rl_model'
tb_log_dir = 'tb_logs'
tb_dev_id_file = 'tb_dev_id.txt'

def get_experiment(exp_dirpath):
    exp = {
        'name': dir_basename(exp_dirpath),
        'models_dir': os.path.join(exp_dirpath, models_dir),
        'checkpoints_dir': os.path.join(exp_dirpath, checkpoints_dir),
        'checkpoint_prefix': checkpoint_prefix,
        'tb_log_dir': os.path.join(exp_dirpath, tb_log_dir),
    }
    with open(os.path.join(exp_dirpath, 'env.json')) as f:
        exp['env_config'] = json.load(f)
    with open(os.path.join(exp_dirpath, tb_dev_id_file)) as f:
        lines = f.readlines()
        assert len(lines) == 1
        #TODO this only works if line break is just '\n'
        exp['tb_dev_id'] = lines[0].rstrip('\n')
    with open(os.path.join(exp_dirpath, 'plot.json')) as f:
        exp['plot'] = json.load(f)
    return exp

def init_experiment(exp):
    os.makedirs(exp['models_dir'], exist_ok=True)
    os.makedirs(exp['checkpoints_dir'], exist_ok=True)
    os.makedirs(exp['tb_log_dir'], exist_ok=True)


def fmt_run_name(run_id: int):
    return f'PPO_{run_id}'

def get_run(exp, run_id):
    run_name = fmt_run_name(run_id)
    return {
        'name': run_name,
        'id': run_id,
        'model': os.path.join(exp['models_dir'], run_name),
        'checkpoints_dir': os.path.join(exp['checkpoints_dir'], run_name),
        'checkpoint_prefix': exp['checkpoint_prefix'],
    }

def get_run_ids(exp):
    # Get runs for this experiment by inspecting the tensorboard log dir.
    tb_logs = list_subdirs(exp['tb_log_dir'])
    run_ids = []
    for tb_log_dir in tb_logs:
        run_name = dir_basename(tb_log_dir)
        assert run_name.startswith('PPO_')
        run_ids.append(int(run_name.split('_')[1]))
    return run_ids

def init_run(run):
    os.makedirs(run['checkpoints_dir'], exist_ok=True)

def fmt_checkpoint_name(run, checkpoint_id):
    return f'{run["checkpoint_prefix"]}_{checkpoint_id}_steps'

def get_checkpoint_ids(run):
    model_files = list_files(run['checkpoints_dir'])
    checkpoint_ids = []
    for model_file in model_files:
        name = os.path.basename(model_file)
        assert name.startswith(run['checkpoint_prefix'])
        assert name.endswith('_steps.zip')
        checkpoint_ids.append(int(name.split('_')[-2]))
    return checkpoint_ids

def get_checkpoint_model(run, checkpoint_id):
    return os.path.join(
        run['checkpoints_dir'], fmt_checkpoint_name(run, checkpoint_id)
    )



# env-related utils

def unbatch_obs(obs, n_agents):
    xs = [dict() for _ in range(n_agents)]
    for k, v in obs.items():
        for i, x in enumerate(v):
            xs[i][k] = x
    return xs



# file system utils

def listdir(directory, filter_=None):
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if filter_ is None or filter_(os.path.join(directory, f))
    ]

def list_files(directory):
    return listdir(directory, filter_=os.path.isfile)

def list_subdirs(directory):
    return listdir(directory, filter_=os.path.isdir)

def dir_basename(directory):
    return os.path.basename(directory.rstrip('/'))
