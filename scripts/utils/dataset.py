import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.config import model_categories_path


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embedding vectors with associated labels and metadata."""

    def __init__(self, embeddings, labels, metadata):
        self.X = embeddings
        self.y = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.metadata[idx]


def clean_obj_name(obj_name, object_styles):
    """Strip rendering style suffixes from an object name."""
    for obj_style in object_styles:
        if obj_name.endswith(obj_style):
            obj_name = obj_name[:-len(obj_style)]
            break
    return obj_name


def _load_category_scores(y_train, metadata_train, metadata_test):
    """
    Load category mappings and compute per-category mean scores.

    Args:
        y_train: Training score matrix
        metadata_train: Training metadata (column 0 = model name)
        metadata_test: Test metadata (column 0 = model name)

    Returns:
        tuple: (train_categories, test_categories, category_scores)
    """
    tags = pd.read_csv(model_categories_path)
    name_to_tag = dict(zip(tags['Model Name'], tags['Category']))

    train_categories = pd.Series(metadata_train[:, 0]).map(name_to_tag)
    test_categories = pd.Series(metadata_test[:, 0]).map(name_to_tag)

    if train_categories.isna().any():
        raise ValueError(f"NaN categories in train set: {metadata_train[train_categories.isna(), 0]}")
    if test_categories.isna().any():
        raise ValueError(f"NaN categories in test set: {metadata_test[test_categories.isna(), 0]}")

    category_scores = {}
    for category in tags['Category'].unique():
        mask = train_categories.values == category
        category_scores[category] = y_train[mask].mean(axis=0) if mask.any() else np.full(y_train.shape[1], np.nan)

    return train_categories, test_categories, category_scores


def k_fold_trainval_test_multi_object_styles(y_all, all_embeddings, all_metadata, object_styles=['_realistic', '_surround2', '_flash2'], k=5, test_split=0.2, seed=1):
    """
    Create k-fold cross-validation splits.
    First splits data into train+val vs test, then creates k-folds over train+val.

    Args:
        y_all: Target values
        all_embeddings: Feature embeddings
        all_metadata: Metadata for each sample
        object_styles: Style suffixes to strip when grouping objects
        k: Number of folds for cross-validation
        test_split: Fraction of data to hold out for final test set
        seed: Random seed

    Returns:
        folds: List of tuples (train_dataset, val_dataset) for each fold
        train_val_dataset: Combined train+val dataset
        test_dataset: Final held-out test set
    """
    obj_ids = np.array([str(m[0]) for m in all_metadata])
    cleaned_objs = np.array([clean_obj_name(o, object_styles) for o in obj_ids])
    unique_objs = np.unique(cleaned_objs)

    # Split objects into train+val vs test
    np.random.seed(seed)
    shuffled_objs = unique_objs.copy()
    np.random.shuffle(shuffled_objs)

    n_test = int(len(shuffled_objs) * test_split)
    objs_test = shuffled_objs[:n_test]
    objs_trainval = shuffled_objs[n_test:]

    # Create masks for train+val and test
    trainval_set = set(objs_trainval)
    test_set = set(objs_test)
    trainval_mask = np.array([o in trainval_set for o in cleaned_objs])
    test_mask = np.array([o in test_set for o in cleaned_objs])

    # Extract train+val data
    embeddings_trainval = all_embeddings[trainval_mask]
    y_trainval = y_all[trainval_mask]
    metadata_trainval = all_metadata[trainval_mask]

    # Extract test data
    embeddings_test = all_embeddings[test_mask]
    y_test = y_all[test_mask]
    metadata_test = all_metadata[test_mask]

    test_dataset = EmbeddingDataset(embeddings_test, y_test, metadata_test)

    print(f"Total objects: {len(unique_objs)}")
    print(f"Train+Val objects: {len(objs_trainval)}, Test objects: {len(objs_test)}")
    print(f"Train+Val samples: {len(embeddings_trainval)}, Test samples: {len(embeddings_test)}")

    # Create k-fold splits over train+val objects
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    train_val_dataset = EmbeddingDataset(embeddings_trainval, y_trainval, metadata_trainval)
    folds = []

    cleaned_objs_trainval = np.array([
        clean_obj_name(str(o[0]), object_styles) for o in metadata_trainval
    ])

    for fold_idx, (train_obj_indices, val_obj_indices) in enumerate(kf.split(objs_trainval)):
        train_objs = set(objs_trainval[train_obj_indices])
        val_objs = set(objs_trainval[val_obj_indices])

        # Create masks for samples based on object membership
        train_mask = np.array([o in train_objs for o in cleaned_objs_trainval])
        val_mask = np.array([o in val_objs for o in cleaned_objs_trainval])

        # Create datasets for this fold
        train_dataset = EmbeddingDataset(
            embeddings_trainval[train_mask],
            y_trainval[train_mask],
            metadata_trainval[train_mask]
        )
        val_dataset = EmbeddingDataset(
            embeddings_trainval[val_mask],
            y_trainval[val_mask],
            metadata_trainval[val_mask]
        )

        folds.append((train_dataset, val_dataset))

        print(f"Fold {fold_idx + 1}/{k}: Train objects={len(train_objs)}, Val objects={len(val_objs)}, "
              f"Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    return folds, train_val_dataset, test_dataset


def enhance_with_tags(y_train, y_test, metadata_train, metadata_test, for_iou=False, beta=0.3, T=0.1):
    """
    Enhance scores with category-level priors and apply softmax normalization.

    Blends per-sample scores with category-wise mean scores from the training
    set, then applies temperature-scaled softmax.

    Args:
        y_train: Training score matrix
        y_test: Test score matrix
        metadata_train: Training metadata (column 0 = model name)
        metadata_test: Test metadata (column 0 = model name)
        for_iou: If True, higher scores are better; if False, negate before softmax
        beta: Blending weight for original scores (1-beta for category)
        T: Softmax temperature

    Returns:
        tuple: (enhanced_train, enhanced_test) normalized score matrices
    """
    train_categories, test_categories, category_scores = _load_category_scores(
        y_train, metadata_train, metadata_test
    )

    def apply_enhancement(y, categories):
        """Apply category enhancement and softmax normalization."""
        category_score = np.vstack(categories.map(category_scores).values)

        # Adjust beta for missing categories (fallback to original scores)
        has_nan = np.isnan(category_score).any(axis=1)
        beta_adjusted = np.where(has_nan, 1.0, beta)
        category_score = np.where(np.isnan(category_score), y, category_score)

        # Blend original and category scores
        enhanced = beta_adjusted[:, None] * y + (1 - beta_adjusted[:, None]) * category_score

        # Apply temperature-scaled softmax
        if not for_iou:
            enhanced = softmax(-enhanced / T, axis=1)
        else:
            enhanced = softmax(enhanced / T, axis=1)

        return enhanced

    enhanced_train = apply_enhancement(y_train, train_categories)
    enhanced_test = apply_enhancement(y_test, test_categories)

    return enhanced_train, enhanced_test


def enhance_with_tags_no_dist(y_train, y_test, metadata_train, metadata_test, beta=0.3):
    """
    Enhance scores with category-level priors without softmax normalization.

    Same blending as enhance_with_tags but returns raw blended scores.

    Args:
        y_train: Training score matrix
        y_test: Test score matrix
        metadata_train: Training metadata (column 0 = model name)
        metadata_test: Test metadata (column 0 = model name)
        beta: Blending weight for original scores (1-beta for category)

    Returns:
        tuple: (enhanced_train, enhanced_test) blended score matrices
    """
    train_categories, test_categories, category_scores = _load_category_scores(
        y_train, metadata_train, metadata_test
    )

    def apply_enhancement(y, categories):
        """Apply category enhancement without normalization."""
        category_score = np.vstack(categories.map(category_scores).values)

        # Adjust beta for missing categories (fallback to original scores)
        has_nan = np.isnan(category_score).any(axis=1)
        beta_adjusted = np.where(has_nan, 1.0, beta)
        category_score = np.where(np.isnan(category_score), y, category_score)

        # Blend original and category scores
        enhanced = beta_adjusted[:, None] * y + (1 - beta_adjusted[:, None]) * category_score

        return enhanced

    enhanced_train = apply_enhancement(y_train, train_categories)
    enhanced_test = apply_enhancement(y_test, test_categories)

    return enhanced_train, enhanced_test
