"""
SCOUT: method selection with normalization constant estimation.

Trains a softmax neural network on category-enhanced soft labels, estimates
per-sample normalization constants via cross-validated R^2 weighting and
ridge regression, then evaluates regret under various cost regimes.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge

from utils.evaluations import (
    single_point_exp, full_cost_range_exp,
    gen_cost_range_exp, latency_memory_exps,
    regret_given_cost_vectorized_fixed,
)
from utils.dataset import enhance_with_tags, k_fold_trainval_test_multi_object_styles
from utils.random import seed_worker, set_seed
from utils.config import data_path, results_folder
from utils.models import train_model_epochs, SoftmaxNN

METHODS = ['hunyuan', 'instant_mesh', 'trellis', 'triposr']


# ---------------------------------------------------------------------------
# Core SCOUT functions
# ---------------------------------------------------------------------------

def predict_normalization_constants_with_preds(utility_y_train_val, train_val_predictions, X_train_val, X_test, T, r2s, S2s, expectations, alpha=10.0, verbose=False):
    """
    Estimate per-sample normalization constants via R^2-weighted combination
    and ridge regression.

    Args:
        utility_y_train_val: Train+val utility scores
        train_val_predictions: Softmax probability predictions on train+val
        X_train_val: Train+val embeddings
        X_test: Test embeddings
        T: Temperature parameter
        r2s: Per-method R^2 values from cross-validation
        S2s: Per-method prediction variances
        expectations: Per-method expectation ratios
        alpha: Ridge regression regularization strength
        verbose: Print weight diagnostics

    Returns:
        tuple: (test_constants, train_val_constants) predicted normalization
            constants for test and train+val sets
    """
    z_train_val_direct = np.exp(utility_y_train_val / T) / train_val_predictions
    sigma2 = S2s * (1.0 - np.array(r2s) + 1e-8)
    raw_weights = 1.0 / (sigma2 * np.array(expectations) + 1e-8)
    weights = raw_weights / raw_weights.sum()

    if verbose:
        print(f"expectations: {expectations}")
        print(f"sigma2: {sigma2}")
        print(f"Z combination weights: {weights}")

    normalization_constants_train_val = (z_train_val_direct * weights[None, :]).sum(axis=1)

    normalization_linear_model = Ridge(alpha=alpha)
    normalization_linear_model.fit(X_train_val, normalization_constants_train_val)
    normalization_constants_test_predictions = normalization_linear_model.predict(X_test)
    normalization_constants_train_val_predictions = normalization_linear_model.predict(X_train_val)

    return normalization_constants_test_predictions, normalization_constants_train_val_predictions


def single_fold_evaluation(X_train, y_train, X_val, y_val, metadata_train, metadata_val, kwargs, for_iou, verbose=False, return_predictions=False):
    """
    Train and evaluate a single fold of the SCOUT pipeline.

    Args:
        X_train: Training embeddings
        y_train: Training scores
        X_val: Validation embeddings
        y_val: Validation scores
        metadata_train: Training metadata
        metadata_val: Validation metadata
        kwargs: Hyperparameters dict (T, beta, lr, epochs, patience,
            weight_decay, hidden_dim, device, batch_size)
        for_iou: If True, higher is better
        verbose: Print diagnostics
        return_predictions: If True, also return train/val predictions

    Returns:
        dict or tuple: Training info dict, optionally with
            (train_predictions, val_predictions)
    """
    beta = kwargs.get('beta')
    T = kwargs.get('T')
    lr = kwargs.get('lr')
    epochs = kwargs.get('epochs')
    weight_decay = kwargs.get('weight_decay')
    hidden_dim = kwargs.get('hidden_dim')
    device = kwargs.get('device')
    batch_size = kwargs.get('batch_size')
    n_classes = y_train.shape[1]

    y_train_enhanced, y_val_enhanced = enhance_with_tags(y_train, y_val, metadata_train, metadata_val, for_iou=for_iou, beta=beta, T=T)
    X_train_tensor = torch.tensor(X_train).to(device)
    X_val_tensor = torch.tensor(X_val).to(device)
    y_train_probs_tensor = torch.tensor(y_train_enhanced).to(device)
    y_val_probs_tensor = torch.tensor(y_val_enhanced).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_probs_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset = TensorDataset(X_val_tensor, y_val_probs_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42)
    )

    set_seed(42)
    nn_softmax_model = SoftmaxNN(input_dim=X_train.shape[1],
                                 n_classes=n_classes,
                                 hidden_dim=hidden_dim)
    nn_softmax_model.to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(nn_softmax_model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Training Softmax Neural Network...")

    def regret_function(y_true, y_pred):
        utility_y_true = -y_true if not for_iou else y_true
        return regret_given_cost_vectorized_fixed(utility_y_true, y_pred, verbose=False)[-1]

    _, val_regrets, epochs = train_model_epochs(
        nn_softmax_model,
        train_loader,
        val_loader,
        X_train_tensor,
        X_val_tensor,
        y_train,
        y_val,
        regret_function,
        criterion,
        optimizer,
        epochs=epochs
    )

    nn_logits = nn_softmax_model(X_train_tensor)
    nn_val_logits = nn_softmax_model(X_val_tensor)
    nn_log_probs_train = torch.log_softmax(nn_logits, dim=1)
    nn_log_probs_val = torch.log_softmax(nn_val_logits, dim=1)
    nn_train_predictions = torch.softmax(nn_logits, dim=1)
    nn_val_predictions = torch.softmax(nn_val_logits, dim=1)

    train_loss = criterion(nn_log_probs_train, y_train_probs_tensor).item()
    val_loss = criterion(nn_log_probs_val, y_val_probs_tensor).item()

    lr_train_predictions = nn_train_predictions.cpu().detach().numpy()
    lr_val_predictions = nn_val_predictions.cpu().detach().numpy()

    info_dict = {
        'T': T,
        'beta': beta,
        'lr': lr,
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_regrets': val_regrets
    }
    if return_predictions:
        return info_dict, lr_train_predictions, lr_val_predictions
    else:
        return info_dict


def estimate_r2_statistics(folds, kwargs, T, for_iou):
    """
    Estimate per-method R^2, variance, and expectation ratios via cross-validation.

    Args:
        folds: List of (train_dataset, val_dataset) tuples.
        kwargs: Hyperparameters dict passed to single_fold_evaluation.
        T: Temperature parameter.
        for_iou: If True, higher is better.

    Returns:
        tuple: (r2s, S2s, expectations) averaged over folds.
    """
    fold_r2s = []
    fold_S2s = []
    fold_expectations = []

    for train_ds, val_ds in folds:
        _, _, p_hat = single_fold_evaluation(
            train_ds.X, train_ds.y, val_ds.X, val_ds.y,
            train_ds.metadata, val_ds.metadata, kwargs,
            for_iou=for_iou, verbose=False, return_predictions=True
        )

        y_val = val_ds.y
        if not for_iou:
            p_true = np.exp(-y_val / T) / np.sum(np.exp(-y_val / T), axis=1, keepdims=True)
            u_val = -y_val
        else:
            p_true = np.exp(y_val / T) / np.sum(np.exp(y_val / T), axis=1, keepdims=True)
            u_val = y_val

        method_r2s = []
        method_S2s = []
        method_expectations = []
        for method_idx in range(len(METHODS)):
            predictions = p_hat[:, method_idx]
            truths = p_true[:, method_idx]
            r2 = np.corrcoef(truths, predictions)[0, 1] ** 2
            method_r2s.append(r2)
            method_S2s.append(predictions.var())

            Z_i = np.sum(np.exp(u_val / T), axis=1)
            u_ij = u_val[:, method_idx]
            ratio = Z_i**4 / np.exp(u_ij / T)**2
            method_expectations.append(ratio.mean())

        fold_r2s.append(method_r2s)
        fold_S2s.append(method_S2s)
        fold_expectations.append(method_expectations)

    r2s = np.array(fold_r2s).mean(axis=0)
    S2s = np.array(fold_S2s).mean(axis=0)
    expectations = np.array(fold_expectations).mean(axis=0)
    return r2s, S2s, expectations


def _append_pass_column(arrays, fill_value):
    """Append a constant column to each array in the list."""
    return [np.hstack((arr, np.full((arr.shape[0], 1), fill_value))) for arr in arrays]


def _validate_arrays(**named_arrays):
    """Assert no NaN or Inf values in any of the named arrays."""
    for name, arr in named_arrays.items():
        assert not np.isnan(arr).any(), f"{name} are nan"
        assert not np.isinf(arr).any(), f"{name} are inf"


def _run_and_save_regret_exps(results_fns, input_data, results_path, seed):
    """Run regret experiment functions and write results to disk."""
    exp_results = [fn(*input_data, verbose=True) for fn in results_fns]

    with open(f'{results_path}/results_{seed}.txt', 'w') as f:
        for exp_name, exp_value in zip(results_fns, exp_results):
            for name, regret_for_method in zip(METHODS, exp_value):
                f.write(f"{exp_name.__name__}-{name}: {regret_for_method}\n")
            f.write(f"{exp_name.__name__}: {exp_value[-1]}\n")

    for exp_index, exp_value in enumerate(exp_results):
        print(f"{exp_index}: {[round(float(e), 8) for e in exp_value]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SCOUT evaluation script')
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
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    folder_suffix = args.folder_suffix

    folder_prefix = 'scout_proper'
    results_path = f'{results_folder}/{folder_prefix}/results{folder_suffix}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

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
    beta = args.beta
    lr = args.lr
    optimal_epoch = int(args.epochs)
    alpha = args.alpha
    for_iou = args.for_iou

    X_train_val = train_val_dataset.X
    y_train_val = train_val_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y
    metadata_train_val = train_val_dataset.metadata
    metadata_test = test_dataset.metadata
    n_classes = y_all.shape[1]

    print(f"Optimal epoch: {optimal_epoch}")
    kwargs = {
        'T': T,
        'beta': beta,
        'lr': lr,
        'epochs': optimal_epoch,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'device': device,
        'batch_size': args.batch_size,
        'n_classes': n_classes
    }
    print(kwargs)

    # Train on full train+val and predict test
    _, lr_train_val_predictions, lr_test_predictions = single_fold_evaluation(
        X_train_val, y_train_val, X_test, y_test,
        metadata_train_val, metadata_test, kwargs,
        for_iou=for_iou, verbose=False, return_predictions=True
    )

    # Cross-validated R^2 estimation for normalization constants
    r2s, S2s, expectations = estimate_r2_statistics(GSO_folds, kwargs, T, for_iou)
    print(f"R^2s: {r2s}")

    # Convert to utility space
    utility_train_val_predictions = T * np.log(lr_train_val_predictions)
    utility_test_predictions = T * np.log(lr_test_predictions)
    if not for_iou:
        utility_y_train_val = -y_train_val
        utility_y_test = -y_test
    else:
        utility_y_train_val = y_train_val
        utility_y_test = y_test

    # Estimate and apply normalization constants
    normalization_constants_test_predictions, _ = predict_normalization_constants_with_preds(
        utility_y_train_val, lr_train_val_predictions, X_train_val, X_test,
        T, r2s, S2s, expectations, alpha=alpha, verbose=True
    )
    normalization_constants_test_predictions = np.clip(normalization_constants_test_predictions, 1e-8, 1e8)
    utility_test_predictions = T * np.log(lr_test_predictions * np.reshape(normalization_constants_test_predictions, (-1, 1)))

    # Validate outputs
    _validate_arrays(
        utility_test_predictions=utility_test_predictions,
        utility_y_test=utility_y_test,
        utility_y_train_val=utility_y_train_val,
        utility_train_val_predictions=utility_train_val_predictions,
    )

    # Append pass column for deferral experiments
    pass_fill = 0.0 if not for_iou else 1.0
    (utility_y_test_with_pass,
     utility_test_predictions_with_pass,
     utility_y_train_val_with_pass,
     utility_train_val_predictions_with_pass) = _append_pass_column(
        [utility_y_test, utility_test_predictions,
         utility_y_train_val, utility_train_val_predictions],
        pass_fill
    )

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
    elif experiment_type == 'hyper_cost':
        _run_and_save_regret_exps(
            [full_cost_range_exp], input_data, results_path, seed
        )


if __name__ == "__main__":
    main()
