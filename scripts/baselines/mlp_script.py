"""MLP baseline for model selection using MLPNN."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from baselines import (
    convert_to_utility, build_input_data, validate_arrays, run_experiments,
    setup_output_dirs,
)
from utils.config import data_path
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.evaluations import regret_given_cost_vectorized_fixed
from utils.models import train_model_epochs_mlp, MLPNN
from utils.random import seed_worker, set_seed


def single_fold_evaluation(X_train, y_train, X_val, y_val, metadata_train, metadata_val, kwargs, verbose=False, return_predictions=False, for_iou=False):
    lr = kwargs.get('lr')
    epochs = kwargs.get('epochs')
    patience = kwargs.get('patience')
    weight_decay = kwargs.get('weight_decay')
    hidden_dim = kwargs.get('hidden_dim')
    device = kwargs.get('device')
    batch_size = kwargs.get('batch_size')

    n_classes = y_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_probs_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_probs_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

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
    nn_mlp_model = MLPNN(input_dim=X_train.shape[1],
                         n_classes=n_classes,
                         hidden_dim=hidden_dim)
    nn_mlp_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_mlp_model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Training MLP Neural Network...")

    def regret_function(y_true, y_pred):
        if for_iou:
            return regret_given_cost_vectorized_fixed(y_true, y_pred, verbose=False)[-1]
        else:
            return regret_given_cost_vectorized_fixed(-y_true, -y_pred, verbose=False)[-1]

    if epochs is not None:
        losses, val_regrets, epochs = train_model_epochs_mlp(
            nn_mlp_model,
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
    elif patience is not None:
        raise ValueError("Patience is not supported for MLP")
    else:
        raise ValueError("Either epochs or patience must be provided")

    nn_logits = nn_mlp_model(X_train_tensor)
    nn_val_logits = nn_mlp_model(X_val_tensor)
    nn_train_predictions = nn_logits
    nn_val_predictions = nn_val_logits

    train_loss = criterion(nn_logits, y_train_probs_tensor).item()
    val_loss = criterion(nn_val_logits, y_val_probs_tensor).item()

    mlp_train_predictions = nn_train_predictions.cpu().detach().numpy()
    mlp_val_predictions = nn_val_predictions.cpu().detach().numpy()

    info_dict = {
        'lr': lr,
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_regrets': val_regrets
    }
    if return_predictions:
        return info_dict, mlp_train_predictions, mlp_val_predictions
    else:
        return info_dict


def main():
    parser = argparse.ArgumentParser(description='MLP baseline evaluation script')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--folder_suffix', type=str, default='', help='Folder suffix for results')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    results_path = setup_output_dirs('mlp_script', args.folder_suffix)

    data = np.load(data_path)
    y_all = data['y_all']
    all_embeddings = data['all_embeddings']
    all_metadata = data['all_metadata']

    GSO_folds, train_val_dataset, test_dataset = k_fold_trainval_test_multi_object_styles(
        y_all, all_embeddings, all_metadata, k=5, test_split=0.2, seed=seed
    )
    print(f"\nCreated {len(GSO_folds)} folds for cross-validation")
    print(f"Final test set size: {len(test_dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    n_classes = y_all.shape[1]
    weight_decay = args.weight_decay
    lr = args.lr
    optimal_epoch = int(args.epochs)

    X_train_val = train_val_dataset.X
    y_train_val = train_val_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y
    metadata_train_val = train_val_dataset.metadata
    metadata_test = test_dataset.metadata

    print(f"Optimal epoch: {optimal_epoch}")
    kwargs = {
        'lr': lr,
        'epochs': optimal_epoch,
        'weight_decay': weight_decay,
        'hidden_dim': hidden_dim,
        'device': device,
        'batch_size': batch_size,
        'n_classes': n_classes
    }
    print(kwargs)

    _, mlp_train_val_predictions, mlp_test_predictions = single_fold_evaluation(
        X_train_val, y_train_val, X_test, y_test,
        metadata_train_val, metadata_test, kwargs,
        verbose=False, return_predictions=True, for_iou=args.for_iou
    )

    utility_y_test, utility_test_predictions = convert_to_utility(y_test, mlp_test_predictions, args.for_iou)
    utility_y_train_val, utility_train_val_predictions = convert_to_utility(y_train_val, mlp_train_val_predictions, args.for_iou)

    validate_arrays(
        utility_test_predictions=utility_test_predictions,
        utility_y_test=utility_y_test,
        utility_y_train_val=utility_y_train_val,
        utility_train_val_predictions=utility_train_val_predictions,
    )

    input_data = build_input_data(
        utility_y_test, utility_test_predictions,
        utility_y_train_val, utility_train_val_predictions,
        args.for_iou,
    )

    run_experiments(args.experiment_type, input_data, results_path, seed)


if __name__ == "__main__":
    main()
