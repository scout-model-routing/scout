"""Oracle baseline: uses ground-truth labels as predictions (upper bound)."""

import argparse

import numpy as np

from baselines import (
    convert_to_utility, build_input_data, run_experiments, setup_output_dirs,
)
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.config import data_path


def main():
    parser = argparse.ArgumentParser(description='Oracle baseline evaluation script')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--folder_suffix', type=str, default='', help='Folder suffix for results')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    results_path = setup_output_dirs('oracle_script', args.folder_suffix)

    data = np.load(data_path)
    y_all = data['y_all']
    all_embeddings = data['all_embeddings']
    all_metadata = data['all_metadata']

    _, train_val_dataset, test_dataset = k_fold_trainval_test_multi_object_styles(
        y_all, all_embeddings, all_metadata, k=5, test_split=0.2, seed=seed
    )

    y_train_val = train_val_dataset.y
    y_test = test_dataset.y
    train_val_predictions = train_val_dataset.y
    test_predictions = test_dataset.y

    for_iou = args.for_iou
    utility_y_test, utility_test_predictions = convert_to_utility(
        y_test, test_predictions, for_iou
    )
    utility_y_train_val, utility_train_val_predictions = convert_to_utility(
        y_train_val, train_val_predictions, for_iou
    )

    input_data = build_input_data(
        utility_y_test, utility_test_predictions,
        utility_y_train_val, utility_train_val_predictions, for_iou
    )

    run_experiments(args.experiment_type, input_data, results_path, seed)


if __name__ == "__main__":
    main()
