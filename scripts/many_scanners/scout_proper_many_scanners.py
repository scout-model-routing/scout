"""
SCOUT many-scanners: method selection with normalization constant estimation
and multiple pass/scanner columns.

Trains a softmax neural network on category-enhanced soft labels, estimates
per-sample normalization constants via cross-validated R^2 weighting and
ridge regression, then appends num_scanners pass columns and evaluates regret.
"""

import argparse
import os

import numpy as np
import torch

from utils.evaluations import single_point_exp
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.config import data_path, results_folder
from scout_proper import (
    single_fold_evaluation, predict_normalization_constants_with_preds,
    estimate_r2_statistics, _validate_arrays, _run_and_save_regret_exps, METHODS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pass_columns(n_rows, num_scanners):
    """Build a matrix of evenly-spaced pass values to append to scores.

    Args:
        n_rows: Number of samples.
        num_scanners: Number of scanner/pass columns.

    Returns:
        Array of shape (n_rows, num_scanners).
    """
    passing_values = [i / num_scanners for i in range(num_scanners)]
    return np.tile(passing_values, (n_rows, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SCOUT many-scanners evaluation script'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--folder_suffix', type=str, default='', help='Folder suffix for results')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature')
    parser.add_argument('--beta', type=float, default=0.7, help='Beta for tag enhancement')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--alpha', type=float, default=1000.0, help='Alpha for ridge regression')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--num_scanners', type=int, default=1, help='Number of scanners')
    args = parser.parse_args()

    seed = args.seed
    num_scanners = args.num_scanners

    # Set up output directories
    folder_prefix = 'scout_proper_many_scanners'
    results_path = f'{results_folder}/{folder_prefix}/results{args.folder_suffix}'
    os.makedirs(results_path, exist_ok=True)

    # Load data
    data = np.load(data_path)
    y_all = data['y_all']
    all_embeddings = data['all_embeddings']
    all_metadata = data['all_metadata']

    # Create k-fold splits with test set
    GSO_folds, train_val_dataset, test_dataset = k_fold_trainval_test_multi_object_styles(
        y_all, all_embeddings, all_metadata, k=5, test_split=0.2, seed=seed
    )
    print(f"\nCreated {len(GSO_folds)} folds for cross-validation")
    print(f"Final test set size: {len(test_dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    T = args.T
    alpha = args.alpha
    optimal_epoch = int(args.epochs)
    n_classes = y_all.shape[1]

    X_train_val = train_val_dataset.X
    y_train_val = train_val_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y
    metadata_train_val = train_val_dataset.metadata
    metadata_test = test_dataset.metadata

    print(f"Optimal epoch: {optimal_epoch}")
    kwargs = {
        'T': T,
        'beta': args.beta,
        'lr': args.lr,
        'epochs': optimal_epoch,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'device': device,
        'batch_size': args.batch_size,
        'n_classes': n_classes,
    }
    print(kwargs)

    # Train on full train+val and predict test
    _, lr_train_val_predictions, lr_test_predictions = single_fold_evaluation(
        X_train_val, y_train_val, X_test, y_test,
        metadata_train_val, metadata_test, kwargs,
        for_iou=False, verbose=False, return_predictions=True
    )

    # Cross-validated R^2 estimation for normalization constants
    r2s, S2s, expectations = estimate_r2_statistics(GSO_folds, kwargs, T, for_iou=False)
    print(f"R^2s: {r2s}")

    # Convert to utility space
    utility_train_val_predictions = T * np.log(lr_train_val_predictions)
    utility_y_train_val = -y_train_val
    utility_y_test = -y_test

    # Estimate and apply normalization constants
    normalization_constants_test_predictions, _ = predict_normalization_constants_with_preds(
        utility_y_train_val, lr_train_val_predictions, X_train_val, X_test,
        T, r2s, S2s, expectations, alpha=alpha, verbose=True
    )
    normalization_constants_test_predictions = np.clip(
        normalization_constants_test_predictions, 1e-8, 1e8
    )
    utility_test_predictions = T * np.log(
        lr_test_predictions * np.reshape(normalization_constants_test_predictions, (-1, 1))
    )

    # Validate outputs
    _validate_arrays(
        utility_test_predictions=utility_test_predictions,
        utility_y_test=utility_y_test,
        utility_y_train_val=utility_y_train_val,
        utility_train_val_predictions=utility_train_val_predictions,
    )

    # Append pass columns for scanner experiments
    all_passes_test = _build_pass_columns(utility_y_test.shape[0], num_scanners)
    all_passes_train_val = _build_pass_columns(utility_y_train_val.shape[0], num_scanners)
    utility_y_test_with_pass = np.hstack((utility_y_test, all_passes_test))
    utility_test_predictions_with_pass = np.hstack((utility_test_predictions, all_passes_test))
    utility_y_train_val_with_pass = np.hstack((utility_y_train_val, all_passes_train_val))
    utility_train_val_predictions_with_pass = np.hstack((utility_train_val_predictions, all_passes_train_val))

    print(f"Utility y test with pass: {utility_y_test_with_pass}")

    input_data = [
        utility_y_test,
        utility_test_predictions,
        utility_y_train_val,
        utility_train_val_predictions,
        utility_y_test_with_pass,
        utility_test_predictions_with_pass,
        utility_y_train_val_with_pass,
        utility_train_val_predictions_with_pass,
    ]

    # Run requested experiment type
    experiment_type = args.experiment_type
    if experiment_type == 'many_scanners':
        _run_and_save_regret_exps(
            [single_point_exp], input_data, results_path, seed
        )
    elif experiment_type == 'main':
        from utils.evaluations import full_cost_range_exp, gen_cost_range_exp
        _run_and_save_regret_exps(
            [single_point_exp, full_cost_range_exp, gen_cost_range_exp],
            input_data, results_path, seed
        )
    elif experiment_type == 'deferral':
        from utils.evaluations import latency_memory_exps
        utility_latencies, utility_memories, utility_latency_memories = latency_memory_exps(
            *input_data, verbose=True
        )
        np.save(f'{results_path}/utility_latencies_{seed}.npy', utility_latencies)
        np.save(f'{results_path}/utility_memories_{seed}.npy', utility_memories)
        np.save(f'{results_path}/utility_latency_memories_{seed}.npy', utility_latency_memories)


if __name__ == "__main__":
    main()
