"""Matrix factorization baseline using BilinearRouter for model selection."""

import argparse

import numpy as np
import torch

from baselines import (
    convert_to_utility, build_input_data, validate_arrays, run_experiments,
    setup_output_dirs,
)
from utils.config import data_path
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.models import BilinearRouter, train_router


@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    return model.predict_all_models(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folder_suffix", type=str, default="")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=32)
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_path = setup_output_dirs("mf_script", args.folder_suffix)

    data = np.load(data_path)
    y_all = data["y_all"]
    all_embeddings = data["all_embeddings"]
    all_metadata = data["all_metadata"]

    GSO_folds, train_val_dataset, test_dataset = k_fold_trainval_test_multi_object_styles(
        y_all, all_embeddings, all_metadata, k=5, test_split=0.2, seed=seed
    )
    print(f"\nCreated {len(GSO_folds)} folds for cross-validation")
    print(f"Final test set size: {len(test_dataset)} samples")

    y_train_val = train_val_dataset.y
    X_train_val = train_val_dataset.X
    y_test = test_dataset.y
    X_test = test_dataset.X

    num_models = 4
    prompt_dim = X_train_val.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Prompt dim: {prompt_dim}, Embed dim: {args.model_embed_dim}, Models: {num_models}, Device: {device}")

    router = BilinearRouter(prompt_dim, num_models, args.model_embed_dim)
    print(f"Parameters: {sum(p.numel() for p in router.parameters()):,}")

    router = train_router(
        router, X_train_val, y_train_val,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size,
        device=device, verbose=True,
    )

    train_val_predictions = predict(router, X_train_val, device)
    test_predictions = predict(router, X_test, device)

    utility_y_test, utility_test_predictions = convert_to_utility(y_test, test_predictions, args.for_iou)
    utility_y_train_val, utility_train_val_predictions = convert_to_utility(y_train_val, train_val_predictions, args.for_iou)

    validate_arrays(
        test_pred=utility_test_predictions,
        y_test=utility_y_test,
        y_tv=utility_y_train_val,
        tv_pred=utility_train_val_predictions,
    )

    input_data = build_input_data(
        utility_y_test, utility_test_predictions,
        utility_y_train_val, utility_train_val_predictions,
        args.for_iou,
    )

    run_experiments(args.experiment_type, input_data, results_path, seed)


if __name__ == "__main__":
    main()
