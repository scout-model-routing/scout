"""Linear regression baseline: trains a Ridge regressor per method and evaluates."""

import argparse

import numpy as np
from sklearn.linear_model import Ridge

from baselines import (
    METHODS, convert_to_utility, build_input_data, validate_arrays,
    run_experiments, setup_output_dirs,
)
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.config import data_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folder_suffix', type=str, default='', help='Folder suffix for results')
    parser.add_argument('--alpha', type=float, default=10000.0, help='Alpha for ridge regression')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    results_path = setup_output_dirs('linear_regression_script', args.folder_suffix)

    data = np.load(data_path)
    y_all = data['y_all']
    all_embeddings = data['all_embeddings']
    all_metadata = data['all_metadata']

    GSO_folds, train_val_dataset, test_dataset = k_fold_trainval_test_multi_object_styles(
        y_all, all_embeddings, all_metadata, k=5, test_split=0.2, seed=seed
    )
    print(f"\nCreated {len(GSO_folds)} folds for cross-validation")
    print(f"Final test set size: {len(test_dataset)} samples")

    y_train_val = train_val_dataset.y
    X_train_val = train_val_dataset.X
    y_test = test_dataset.y
    X_test = test_dataset.X

    lr_train_val_predictions = []
    lr_test_predictions = []
    alpha = args.alpha
    for i, name in enumerate(METHODS):
        y_method_train_val = y_train_val[:, i]
        lr = Ridge(alpha=alpha)
        lr.fit(X_train_val, y_method_train_val)
        y_method_train_val_pred = lr.predict(X_train_val)
        y_method_test_pred = lr.predict(X_test)
        lr_train_val_predictions.append(y_method_train_val_pred)
        lr_test_predictions.append(y_method_test_pred)

    lr_train_val_predictions = np.array(lr_train_val_predictions).T
    lr_test_predictions = np.array(lr_test_predictions).T

    for_iou = args.for_iou
    utility_y_test, utility_test_predictions = convert_to_utility(
        y_test, lr_test_predictions, for_iou
    )
    utility_y_train_val, utility_train_val_predictions = convert_to_utility(
        y_train_val, lr_train_val_predictions, for_iou
    )

    validate_arrays(
        utility_test_predictions=utility_test_predictions,
        utility_y_test=utility_y_test,
        utility_y_train_val=utility_y_train_val,
        utility_train_val_predictions=utility_train_val_predictions,
    )

    input_data = build_input_data(
        utility_y_test, utility_test_predictions,
        utility_y_train_val, utility_train_val_predictions, for_iou
    )

    run_experiments(args.experiment_type, input_data, results_path, seed)


if __name__ == '__main__':
    main()
