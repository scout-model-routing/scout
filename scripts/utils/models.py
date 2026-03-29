import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class SoftmaxNN(nn.Module):
    """Feedforward network with softmax output for method selection."""

    def __init__(self, input_dim, n_classes, hidden_dim=128, dropout_rate=0.2):
        super(SoftmaxNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

    def predict_proba(self, x):
        """Return softmax probabilities over classes."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_logits(self, x):
        """Return raw logits."""
        logits = self.forward(x)
        return logits


class MLPNN(nn.Module):
    """Plain MLP for direct score regression."""

    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super(MLPNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


class BilinearRouter(nn.Module):
    """
    Learns a model embedding e_m per method; element-wise product with
    prompt embedding captures query-model interaction.
    """

    def __init__(self, prompt_dim: int, num_models: int, model_embed_dim: int):
        super().__init__()
        self.num_models = num_models
        self.prompt_proj = nn.Linear(prompt_dim, model_embed_dim, bias=False)
        self.model_embeddings = nn.Embedding(num_models, model_embed_dim)
        self.linear = nn.Linear(model_embed_dim, 1)

    def forward(self, prompt_emb, model_ids):
        """Score a batch of (prompt, model) pairs."""
        x = self.prompt_proj(prompt_emb)
        e = self.model_embeddings(model_ids)
        interaction = x * e
        return self.linear(interaction).squeeze(-1)

    def predict_all_models(self, prompt_emb):
        """Score all models for each prompt in the batch."""
        x = self.prompt_proj(prompt_emb)
        all_e = self.model_embeddings.weight
        interaction = x.unsqueeze(1) * all_e.unsqueeze(0)
        return self.linear(interaction).squeeze(-1)


def get_backbone(input_dim, hidden_dim, hidden_state_dim):
    """Build a 3-layer MLP backbone with ReLU and dropout."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_state_dim)
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_val_loss(model, val_loader, criterion, use_log_softmax=True):
    """Compute weighted-average validation loss over all batches.

    Args:
        model: Model in eval mode.
        val_loader: Validation data loader.
        criterion: Loss function.
        use_log_softmax: Apply log_softmax before criterion.

    Returns:
        Average validation loss.
    """
    val_loss = 0
    total_samples = 0
    for X_batch, y_batch in val_loader:
        logits = model(X_batch)
        if use_log_softmax:
            logits = torch.log_softmax(logits, dim=1)
        batch_loss = criterion(logits, y_batch).item()
        batch_size = X_batch.size(0)
        val_loss += batch_loss * batch_size
        total_samples += batch_size
    return val_loss / total_samples


def _get_predictions(model, X, use_softmax=True):
    """Get model predictions as a numpy array.

    Args:
        model: Model in eval mode.
        X: Input tensor.
        use_softmax: If True, apply softmax (for SoftmaxNN). If False, use raw output.

    Returns:
        Numpy array of predictions.
    """
    if use_softmax:
        return model.predict_proba(X).cpu().detach().numpy()
    else:
        return model(X).cpu().detach().numpy()


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_model_epochs(model, train_loader, val_loader, X_train, X_val, y_train_original, y_val_original, regret_function, criterion, optimizer, epochs):
    """Train a SoftmaxNN for a fixed number of epochs with regret tracking."""
    model.train()
    train_losses = []
    val_regrets = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}")

        model.eval()
        with torch.no_grad():
            val_loss = _compute_val_loss(model, val_loader, criterion)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            lr_train_predictions = _get_predictions(model, X_train)
            lr_val_predictions = _get_predictions(model, X_val)

            train_regret = regret_function(y_train_original, lr_train_predictions)
            val_regret = regret_function(y_val_original, lr_val_predictions)
            val_regrets.append(val_regret)
            print(f"Epoch {epoch+1}/{epochs}, Train Regret: {train_regret:.4f}, Val Regret: {val_regret:.4f}")

    # Final train and val predictions
    lr_train_predictions = _get_predictions(model, X_train)
    lr_val_predictions = _get_predictions(model, X_val)
    train_regret = regret_function(y_train_original, lr_train_predictions)
    val_regret = regret_function(y_val_original, lr_val_predictions)
    print(f"Final Train Regret: {train_regret:.4f}, Final Val Regret: {val_regret:.4f}")
    return train_losses, val_regrets, epochs

def train_model_epochs_mlp(model, train_loader, val_loader, X_train, X_val, y_train_original, y_val_original, regret_function, criterion, optimizer, epochs):
    """Train an MLPNN for a fixed number of epochs with regret tracking."""
    model.train()
    train_losses = []
    val_regrets = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}")

        model.eval()
        with torch.no_grad():
            val_loss = _compute_val_loss(model, val_loader, criterion, use_log_softmax=False)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            mlp_train_predictions = _get_predictions(model, X_train, use_softmax=False)
            mlp_val_predictions = _get_predictions(model, X_val, use_softmax=False)

            train_regret = regret_function(y_train_original, mlp_train_predictions)
            val_regret = regret_function(y_val_original, mlp_val_predictions)
            val_regrets.append(val_regret)
            print(f"Epoch {epoch+1}/{epochs}, Train Regret: {train_regret:.4f}, Val Regret: {val_regret:.4f}")

    # Final train and val predictions
    mlp_train_predictions = _get_predictions(model, X_train, use_softmax=False)
    mlp_val_predictions = _get_predictions(model, X_val, use_softmax=False)
    train_regret = regret_function(y_train_original, mlp_train_predictions)
    val_regret = regret_function(y_val_original, mlp_val_predictions)
    print(f"Final Train Regret: {train_regret:.4f}, Final Val Regret: {val_regret:.4f}")
    return train_losses, val_regrets, epochs


# ---------------------------------------------------------------------------
# Router training
# ---------------------------------------------------------------------------

def train_router(model, X_train, y_train, lr, weight_decay, epochs, batch_size, device, verbose=False):
    """Train a BilinearRouter on expanded (prompt, model_id, score) triples."""
    model = model.to(device)
    num_models = y_train.shape[1]
    num_samples = X_train.shape[0]

    X_expanded = np.repeat(X_train, num_models, axis=0)
    model_ids = np.tile(np.arange(num_models), num_samples)
    y_expanded = y_train.flatten(order="C")

    dataset = TensorDataset(
        torch.tensor(X_expanded, dtype=torch.float32),
        torch.tensor(model_ids, dtype=torch.long),
        torch.tensor(y_expanded, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            loss = criterion(model(xb, mb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:4d} | loss={epoch_loss / len(dataset):.6f}")

    return model
