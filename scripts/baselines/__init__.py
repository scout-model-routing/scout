"""Shared utilities for baseline evaluation scripts."""

import os

import numpy as np

from utils.evaluations import (
    single_point_exp, full_cost_range_exp,
    gen_cost_range_exp, latency_memory_exps,
)

METHODS = ['hunyuan', 'instant_mesh', 'trellis', 'triposr']


def convert_to_utility(y, predictions, for_iou):
    """Convert raw scores and predictions to utility space.

    Args:
        y: Ground truth scores.
        predictions: Model predictions.
        for_iou: If True, higher is better; if False, negate.

    Returns:
        tuple: (utility_y, utility_predictions)
    """
    if not for_iou:
        return -y, -predictions
    return y, predictions


def append_pass_column(arrays, for_iou):
    """Append a pass-option column to each array.

    Args:
        arrays: List of 2D numpy arrays.
        for_iou: If True, fill with 1.0; if False, fill with 0.0.

    Returns:
        List of arrays with the extra column appended.
    """
    fill = 1.0 if for_iou else 0.0
    return [np.hstack((arr, np.full((arr.shape[0], 1), fill))) for arr in arrays]


def validate_arrays(**named_arrays):
    """Assert no NaN or Inf values in any of the named arrays."""
    for name, arr in named_arrays.items():
        assert not np.isnan(arr).any(), f"{name} are nan"
        assert not np.isinf(arr).any(), f"{name} are inf"


def build_input_data(utility_y_test, utility_test_predictions,
                     utility_y_train_val, utility_train_val_predictions,
                     for_iou):
    """Build the standard 8-element input_data list for experiment functions.

    Appends pass columns and returns both with-pass and without-pass arrays.

    Args:
        utility_y_test: Test utilities (methods only).
        utility_test_predictions: Test predictions (methods only).
        utility_y_train_val: Train+val utilities (methods only).
        utility_train_val_predictions: Train+val predictions (methods only).
        for_iou: If True, fill pass column with 1.0; else 0.0.

    Returns:
        List of 8 arrays: [y_test, pred_test, y_tv, pred_tv,
                           y_test_wp, pred_test_wp, y_tv_wp, pred_tv_wp]
    """
    (utility_y_test_wp, utility_test_predictions_wp,
     utility_y_train_val_wp, utility_train_val_predictions_wp) = append_pass_column(
        [utility_y_test, utility_test_predictions,
         utility_y_train_val, utility_train_val_predictions],
        for_iou
    )
    return [
        utility_y_test, utility_test_predictions,
        utility_y_train_val, utility_train_val_predictions,
        utility_y_test_wp, utility_test_predictions_wp,
        utility_y_train_val_wp, utility_train_val_predictions_wp,
    ]


def run_and_save_regret_exps(results_fns, input_data, results_path, seed):
    """Run regret experiment functions and write results to disk."""
    exp_results = [fn(*input_data, verbose=True) for fn in results_fns]

    with open(os.path.join(results_path, f'results_{seed}.txt'), 'w') as f:
        for fn, exp_value in zip(results_fns, exp_results):
            for name, regret in zip(METHODS, exp_value):
                f.write(f"{fn.__name__}-{name}: {regret}\n")
            f.write(f"{fn.__name__}: {exp_value[-1]}\n")

    for idx, exp_value in enumerate(exp_results):
        print(f"{idx}: {[round(float(e), 8) for e in exp_value]}")


def run_experiments(experiment_type, input_data, results_path, seed):
    """Dispatch to the correct experiment based on type string.

    Args:
        experiment_type: One of 'main', 'deferral', 'hyper', 'hyper_cost'.
        input_data: 8-element list of arrays.
        results_path: Directory to write results.
        seed: Seed for filename.
    """
    if experiment_type == 'main':
        run_and_save_regret_exps(
            [single_point_exp, full_cost_range_exp, gen_cost_range_exp],
            input_data, results_path, seed
        )
    elif experiment_type == 'deferral':
        latencies, memories, lat_mem = latency_memory_exps(*input_data, verbose=True)
        np.save(os.path.join(results_path, f'utility_latencies_{seed}.npy'), latencies)
        np.save(os.path.join(results_path, f'utility_memories_{seed}.npy'), memories)
        np.save(os.path.join(results_path, f'utility_latency_memories_{seed}.npy'), lat_mem)
    elif experiment_type == 'hyper':
        run_and_save_regret_exps(
            [single_point_exp], input_data, results_path, seed
        )
    elif experiment_type == 'hyper_cost':
        run_and_save_regret_exps(
            [full_cost_range_exp], input_data, results_path, seed
        )


def setup_output_dirs(folder_prefix, folder_suffix):
    """Create and return results_path directory.

    Args:
        folder_prefix: Name of the baseline (e.g. 'knn_script').
        folder_suffix: Optional suffix from CLI args.

    Returns:
        string: results_path
    """
    from utils.config import results_folder
    results_path = f'{results_folder}/{folder_prefix}/results{folder_suffix}'
    os.makedirs(results_path, exist_ok=True)
    return results_path
