"""Run SCOUT across multiple seeds and scanner counts."""

import os

os.environ.setdefault('SCOUT_EXPERIMENT', 'gso')

from joblib import Parallel, delayed

from utils.config import START_SEED, END_SEED

FOLDER_SUFFIX = '_many_scanners'
SCOUT_EXPERIMENT = 'gso'
SCRIPT_NAMES = ['many_scanners/scout_no_decouple_many_scanners', 'many_scanners/scout_proper_many_scanners']
NUM_SCANS_LIST = [10, 20, 50, 100, 200, 500, 1000]
SEEDS = range(START_SEED, END_SEED)
N_JOBS = 8


def run_script_for_seed(seed, script_name, folder_suffix, num_scans):
    """Run a single script for a given seed and scanner count.

    Args:
        seed: Random seed.
        script_name: Path to script relative to scripts/ (e.g. 'scout_proper').
        folder_suffix: Suffix appended to output folders.
        num_scans: Number of scanners for the experiment.

    Returns:
        int: Return code from the subprocess.
    """
    cmd = (
        f"SCOUT_EXPERIMENT={SCOUT_EXPERIMENT} PYTHONPATH=scripts"
        f" python scripts/{script_name}.py"
        f" --seed {seed}"
        f" --folder_suffix {folder_suffix}"
        f" --experiment_type many_scanners"
        f" --num_scanners {num_scans}"
    )

    print(f"Running {script_name} for seed {seed} and num_scans {num_scans}")
    result = os.system(cmd)
    print(f"Completed {script_name} for seed {seed} and num_scans {num_scans}")
    return result


if __name__ == "__main__":
    for num_scans in NUM_SCANS_LIST:
        for script_name in SCRIPT_NAMES:
            Parallel(n_jobs=N_JOBS)(
                delayed(run_script_for_seed)(
                    seed, script_name, f"{FOLDER_SUFFIX}_{num_scans}", num_scans,
                )
                for seed in SEEDS
            )
