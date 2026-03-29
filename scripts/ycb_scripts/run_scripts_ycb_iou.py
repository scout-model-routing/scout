"""Run all SCOUT and baseline scripts across multiple seeds for YCB IoU experiment."""

import os

os.environ.setdefault('SCOUT_EXPERIMENT', 'ycb_iou')

from utils.config import scout_configs, scout_no_decouple_configs, START_SEED, END_SEED

FOLDER_SUFFIX = '_ycb_iou'
SCOUT_EXPERIMENT = 'ycb_iou'

SCRIPTS = [
    'scout_no_decouple',
    'baselines/knn_script',
    'baselines/mean_script',
    'baselines/mlp_script',
    'baselines/linear_regression_script',
    'baselines/mf_script',
    'scout_proper',
]

SEEDS = range(START_SEED, END_SEED)


def run_script_for_seed(seed, script_name, folder_suffix):
    """Run a single script for a given seed.

    Args:
        seed: Random seed.
        script_name: Path to script relative to scripts/ (e.g. 'scout_proper').
        folder_suffix: Suffix appended to output folders.

    Returns:
        int: Return code from the subprocess.
    """
    cmd = (
        f"SCOUT_EXPERIMENT={SCOUT_EXPERIMENT} PYTHONPATH=scripts"
        f" python scripts/{script_name}.py"
        f" --seed {seed} --folder_suffix {folder_suffix}"
        f" --for_iou"
    )

    if script_name == 'scout_proper':
        cmd += f" --T {scout_configs['T']} --beta {scout_configs['beta']}"
    elif script_name == 'scout_no_decouple':
        cmd += f" --T {scout_no_decouple_configs['T']} --beta {scout_no_decouple_configs['beta']}"
    else:
        cmd += " --experiment_type main"

    print(f"Running {script_name} for seed {seed}")
    result = os.system(cmd)
    print(f"Completed {script_name} for seed {seed}")
    return result


if __name__ == "__main__":
    for seed in SEEDS:
        for script_name in SCRIPTS:
            run_script_for_seed(seed, script_name, FOLDER_SUFFIX)
