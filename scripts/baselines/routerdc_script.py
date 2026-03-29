"""RouterDC baseline: routing via K-means clustering and dual contrastive losses."""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

from baselines import setup_output_dirs
from utils.evaluations import regret_given_cost_vectorized_fixed
from utils.dataset import k_fold_trainval_test_multi_object_styles
from utils.random import seed_worker, set_seed
from utils.config import data_path
from utils.models import get_backbone

METHODS = ['hunyuan', 'instant_mesh', 'trellis', 'triposr']


class RouterDataset(Dataset):
    """Dataset of (embedding, score, cluster_id) triples for RouterDC training."""

    def __init__(self, embeddings, scores, cluster_ids):
        self.embeddings = embeddings
        self.scores = scores
        self.cluster_ids = cluster_ids

    def __getitem__(self, index):
        return (self.embeddings[index],
                self.scores[index],
                self.cluster_ids[index])

    def __len__(self):
        return len(self.embeddings)


class RouterModule(nn.Module):
    """RouterDC model with sample-LLM and cluster contrastive losses.

    Matches shuhao02/RouterDC/train_router_mdeberta.py.
    """

    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function="cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function

    def compute_similarity(self, input1, input2):
        """Compute pairwise similarity between two sets of vectors."""
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1, dim=1).unsqueeze(1) * torch.norm(input2, dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T

    def forward(self, x, t=1):
        hidden_state = self.backbone(x)
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        """Contrastive loss between top-k positive and last-k negative models."""
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False, dim=-1)

        for i in range(top_k):
            positive_index = top_index[:, i].view(-1, 1)
            mask = torch.where(top_index_true[:, i].view(-1, 1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)
            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:, 0])
            log_x = log_x * mask
            loss += torch.mean(-log_x)
        return loss

    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        """Contrastive loss encouraging same-cluster samples to be similar."""
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id).view(-1)
            select_positive_index = positive_indexs[torch.randint(len(positive_indexs), (1,))].view(-1)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id).view(-1)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            perm = torch.randperm(len(negtive_indexs))[:last_k2]
            select_negtive_index = negtive_indexs[perm].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:, 0])
        return loss


def convert_chamfer_to_scores(chamfer, temperature=0.1):
    """Convert chamfer distances (lower=better) to scores (higher=better)."""
    return np.exp(-chamfer / temperature)


def convert_iou_to_scores(iou, temperature=0.1):
    """Convert IOU (higher=better) to scores (higher=better)."""
    return np.exp(iou / temperature)


def assign_clusters(embeddings, n_clusters=5, seed=42):
    """Assign cluster IDs using K-means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print(f"Clusters: {n_clusters}, distribution: {dict(zip(unique, counts))}")
    return cluster_ids


def evaluate(model, embeddings, scores_raw, device, verbose=False):
    """Evaluate model accuracy and regret on raw scores."""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(embeddings).to(device)
        logits, _ = model.forward(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)
    targets = scores_raw.argmax(axis=1)

    regrets = regret_given_cost_vectorized_fixed(scores_raw, probs)

    acc = (preds == targets).mean()
    pred_chamfer = scores_raw[np.arange(len(scores_raw)), preds]
    best_chamfer = scores_raw.max(axis=1)
    regret = (best_chamfer - pred_chamfer).mean()

    model.train()
    return acc, regret, probs, regrets


def main():
    parser = argparse.ArgumentParser(description='RouterDC evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folder_suffix', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--T', type=float, default=0.1, help='Temperature')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--total_steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=2, help='K+ positive LLMs')
    parser.add_argument('--last_k', type=int, default=1, help='K- negative LLMs')
    parser.add_argument('--H', type=int, default=3, help='Out-group negatives for cluster loss')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Backbone hidden dim')
    parser.add_argument('--hidden_state_dim', type=int, default=128, help='Encoder output / embedding dim')
    parser.add_argument('--cluster_loss_weight', type=float, default=1.0, help='Lambda for cluster loss')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters N')
    parser.add_argument('--score_temperature', type=float, default=0.1, help='Temperature for score conversion')
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--experiment_type', type=str, default='main', help='Experiment type')
    parser.add_argument('--for_iou', action='store_true', help='For IOU')
    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    results_path = setup_output_dirs('routerdc_script', args.folder_suffix)

    # Load data
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

    X_train_val = train_val_dataset.X
    y_train_val = train_val_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y

    num_models = len(METHODS)
    input_dim = X_train_val.shape[1]
    for_iou = args.for_iou

    # Convert scores for training
    if not for_iou:
        train_scores = convert_chamfer_to_scores(y_train_val, args.score_temperature)
    else:
        train_scores = convert_iou_to_scores(y_train_val, args.score_temperature)

    # Assign clusters for sample-sample loss
    train_clusters = assign_clusters(X_train_val, n_clusters=args.n_clusters, seed=seed)
    dataset = RouterDataset(X_train_val, train_scores, train_clusters)

    # Build model
    print(f"\nModel: input_dim={input_dim}, hidden_dim={args.hidden_dim}, "
          f"hidden_state_dim={args.hidden_state_dim}, node_size={num_models}")
    backbone = get_backbone(input_dim, args.hidden_dim, args.hidden_state_dim)
    model = RouterModule(
        backbone=backbone,
        hidden_state_dim=args.hidden_state_dim,
        node_size=num_models,
        similarity_function=args.similarity_function,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training: step-based loop matching original while(True) pattern
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"\nTraining: steps={args.total_steps}, batch_size={args.batch_size}, "
          f"top_k={args.top_k}, last_k={args.last_k}")
    print("=" * 60)

    pbar = tqdm(range(args.total_steps))
    step = 0
    while True:
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, scores, cluster_ids = batch
            inputs = inputs.to(device)
            scores = scores.to(device)
            cluster_ids = cluster_ids.to(device)

            x, hidden_state = model.forward(inputs, t=args.T)
            loss = model.compute_sample_llm_loss(x=x, index_true=scores, top_k=args.top_k, last_k=args.last_k)

            if args.cluster_loss_weight:
                cluster_loss = model.compute_cluster_loss(hidden_state=hidden_state, cluster_ids=cluster_ids, t=args.T, H=args.H)
                loss = loss + args.cluster_loss_weight * cluster_loss

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)
            step += 1
            if step >= args.total_steps:
                break
        if step >= args.total_steps:
            break

    # Evaluate
    if not for_iou:
        eval_y_train = -y_train_val
        eval_y_test = -y_test
    else:
        eval_y_train = y_train_val
        eval_y_test = y_test

    train_acc, train_regret, _, _ = evaluate(model, X_train_val, eval_y_train, device, verbose=True)
    test_acc, test_regret, _, test_regrets = evaluate(model, X_test, eval_y_test, device, verbose=True)

    print(f"Train Acc: {train_acc:.3f}, Train Regret: {train_regret:.4f}")
    print(f"Test Acc: {test_acc:.3f}, Test Regret: {test_regret:.4f}")

    with open(f"{results_path}/results_{seed}.txt", "w") as f:
        for exp_name, exp_value in zip(['single_point_exp'], [test_regrets]):
            for name, regret in zip(METHODS, exp_value):
                f.write(f"{exp_name}-{name}: {regret}\n")
            f.write(f"{exp_name}: {exp_value[-1]}\n")

    print(test_regrets)


if __name__ == "__main__":
    main()
