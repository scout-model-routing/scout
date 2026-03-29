"""
SCOUT (no decouple): method selection without normalization constant estimation.

Appends a pass column to scores before splitting, trains a softmax neural
network on category-enhanced soft labels, then evaluates regret by slicing
the pass column back out.
"""

import argparse
import os

import numpy as np
import torch

from utils.evaluations import (
    single_point_exp, full_cost_range_exp,
    gen_cost_range_exp, latency_memory_exps,
)
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.config import data_path, results_folder
from scout_proper import single_fold_evaluation, _validate_arrays, _run_and_save_regret_exps, METHODS


def main():
    parser = argparse.ArgumentParser(description='SCOUT (no decouple) evaluation script')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--folder_suffix', type=str, default='', help='Folder suffix for results')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature')
    parser.add_argument('--beta', type=float, default=0.7, help='Beta for tag enhancement')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    for_iou = args.for_iou

    # Set up output directories
    folder_prefix = 'scout_no_decouple'
    results_path = f'{results_folder}/{folder_prefix}/results{args.folder_suffix}'
    os.makedirs(results_path, exist_ok=True)

    # Load data and append pass column before splitting
    data = np.load(data_path)
    y_all = data['y_all']
    pass_fill = 0.0 if not for_iou else 1.0
    y_all = np.hstack([y_all, np.full((y_all.shape[0], 1), pass_fill)])
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
    optimal_epoch = int(args.epochs)

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
    }
    print(kwargs)

    # Train on full train+val and predict test
    _, lr_train_val_predictions, lr_test_predictions = single_fold_evaluation(
        X_train_val, y_train_val, X_test, y_test,
        metadata_train_val, metadata_test, kwargs,
        for_iou=for_iou, verbose=False, return_predictions=True
    )

    # Convert to utility space
    utility_train_val_predictions = T * np.log(lr_train_val_predictions)
    utility_test_predictions = T * np.log(lr_test_predictions)
    if not for_iou:
        utility_y_train_val = -y_train_val
        utility_y_test = -y_test
    else:
        utility_y_train_val = y_train_val
        utility_y_test = y_test

    # Split into with-pass (full) and without-pass (methods only) views
    utility_test_predictions = T * np.log(lr_test_predictions)
    utility_test_predictions_with_pass = np.copy(utility_test_predictions)
    utility_test_predictions = utility_test_predictions[:, :-1]
    utility_y_test_with_pass = np.copy(utility_y_test)
    utility_y_test = utility_y_test_with_pass[:, :-1]
    utility_y_train_val_with_pass = np.copy(utility_y_train_val)
    utility_y_train_val = utility_y_train_val_with_pass[:, :-1]
    utility_train_val_predictions_with_pass = np.copy(utility_train_val_predictions)
    utility_train_val_predictions = utility_train_val_predictions_with_pass[:, :-1]

    # Validate outputs
    _validate_arrays(
        utility_test_predictions=utility_test_predictions,
        utility_y_test=utility_y_test,
        utility_y_train_val=utility_y_train_val,
        utility_train_val_predictions=utility_train_val_predictions,
    )

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
    if experiment_type == 'main':
        _run_and_save_regret_exps(
            [single_point_exp, full_cost_range_exp, gen_cost_range_exp],
            input_data, results_path, seed
        )
    elif experiment_type == 'deferral':
        utility_latencies, utility_memories, utility_latency_memories = latency_memory_exps(
            *input_data, verbose=True
        )
        np.save(f'{results_path}/utility_latencies_{seed}.npy', utility_latencies)
        np.save(f'{results_path}/utility_memories_{seed}.npy', utility_memories)
        np.save(f'{results_path}/utility_latency_memories_{seed}.npy', utility_latency_memories)
    elif experiment_type == 'hyper':
        _run_and_save_regret_exps(
            [single_point_exp], input_data, results_path, seed
        )


if __name__ == "__main__":
    main()
