"""
Mapping Complexity Profiles (MCP) - 4 mapping levels.
Level 1: Ridge regression (linear)
Level 2: Kernel ridge regression (RBF)
Level 3: 1-hidden-layer MLP (shallow learned nonlinear)
Level 4: 2-hidden-layer MLP (deep learned nonlinear)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RidgeRegressionMapping:
    """Level 1: Ridge regression (closed-form linear mapping)."""
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, Y):
        """
        Fit ridge regression.
        
        Args:
            X: [n_samples, d_source] source representations
            Y: [n_samples, d_target] target representations
        """
        self.model.fit(X, Y)
        
    def predict(self, X):
        """
        Predict Y from X.
        
        Args:
            X: [n_samples, d_source]
        
        Returns:
            Y_pred: [n_samples, d_target]
        """
        return self.model.predict(X)
    
    def score(self, X, Y):
        """
        Compute R² score.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
        
        Returns:
            r2: Multivariate R²
        """
        Y_pred = self.predict(X)
        return multivariate_r2(Y, Y_pred)


class KernelRidgeMapping:
    """Level 2: Kernel ridge regression with RBF kernel."""
    
    def __init__(self, alpha=1.0, bandwidth='median'):
        """
        Args:
            alpha: Regularization strength
            bandwidth: Bandwidth for RBF kernel ('median' or float)
        """
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.model = KernelRidge(alpha=alpha, kernel='rbf', gamma=None)
        self.gamma_ = None
        
    def fit(self, X, Y):
        """
        Fit kernel ridge regression.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
        """
        # Compute bandwidth
        if self.bandwidth == 'median':
            pairwise_dist = pdist(X, metric='euclidean')
            bandwidth = np.median(pairwise_dist)
            # Convert bandwidth to gamma for sklearn
            self.gamma_ = 1.0 / (2 * bandwidth ** 2)
        else:
            self.gamma_ = 1.0 / (2 * self.bandwidth ** 2)
        
        self.model.set_params(gamma=self.gamma_)
        self.model.fit(X, Y)
        
    def predict(self, X):
        """
        Predict Y from X.
        
        Args:
            X: [n_samples, d_source]
        
        Returns:
            Y_pred: [n_samples, d_target]
        """
        return self.model.predict(X)
    
    def score(self, X, Y):
        """
        Compute R² score.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
        
        Returns:
            r2: Multivariate R²
        """
        Y_pred = self.predict(X)
        return multivariate_r2(Y, Y_pred)


class MLPMapping(nn.Module):
    """Base MLP mapping class."""
    
    def __init__(self, input_dim, output_dim, hidden_dims, activation='gelu', dropout=0.1):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('gelu', 'relu', 'silu')
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.net(x)


class MLPMappingTrainer:
    """Trainer for MLP mappings (Levels 3 and 4)."""
    
    def __init__(self, input_dim, output_dim, hidden_dims, activation='gelu', 
                 lr=5e-4, weight_decay=1e-4, dropout=0.1, batch_size=512, 
                 max_epochs=300, patience=20, device='cuda'):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            lr: Learning rate
            weight_decay: Weight decay
            dropout: Dropout rate
            batch_size: Batch size
            max_epochs: Maximum epochs
            patience: Early stopping patience
            device: Device to use
        """
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        
        # Create model
        self.model = MLPMapping(input_dim, output_dim, hidden_dims, activation, dropout).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
    def fit(self, X_train, Y_train, X_val, Y_val):
        """
        Train MLP mapping.
        
        Args:
            X_train: [n_train, input_dim]
            Y_train: [n_train, output_dim]
            X_val: [n_val, input_dim]
            Y_val: [n_val, output_dim]
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        Y_train = torch.FloatTensor(Y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        Y_val = torch.FloatTensor(Y_val).to(self.device)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, Y_batch in train_loader:
                self.optimizer.zero_grad()
                Y_pred = self.model(X_batch)
                loss = self.criterion(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                Y_val_pred = self.model(X_val)
                val_loss = self.criterion(Y_val_pred, Y_val).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                break
        
        # Load best model
        self.model.load_state_dict(best_state)
        
    def predict(self, X):
        """
        Predict Y from X.
        
        Args:
            X: [n_samples, input_dim]
        
        Returns:
            Y_pred: [n_samples, output_dim]
        """
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            Y_pred = self.model(X)
        return Y_pred.cpu().numpy()
    
    def score(self, X, Y):
        """
        Compute R² score.
        
        Args:
            X: [n_samples, input_dim]
            Y: [n_samples, output_dim]
        
        Returns:
            r2: Multivariate R²
        """
        Y_pred = self.predict(X)
        return multivariate_r2(Y, Y_pred)


def multivariate_r2(Y_true, Y_pred):
    """
    Compute multivariate R² (fraction of variance explained across all dimensions).
    
    Args:
        Y_true: [n_samples, d]
        Y_pred: [n_samples, d]
    
    Returns:
        r2: Multivariate R²
    """
    # Total sum of squares
    Y_mean = np.mean(Y_true, axis=0, keepdims=True)
    total_ss = np.sum((Y_true - Y_mean) ** 2)
    
    # Residual sum of squares
    residual_ss = np.sum((Y_true - Y_pred) ** 2)
    
    # R²
    if total_ss < 1e-10:
        return 0.0
    r2 = 1.0 - residual_ss / total_ss
    return max(r2, 0.0)  # Clamp to non-negative


if __name__ == "__main__":
    # Test mappings
    print("Testing MCP mappings...")
    
    # Create dummy data
    n_samples = 1000
    d_source = 512
    d_target = 512
    
    X = np.random.randn(n_samples, d_source)
    Y = 0.8 * X + 0.2 * np.random.randn(n_samples, d_target)  # Mostly linear
    
    # Split
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    Y_train, Y_val, Y_test = Y[:n_train], Y[n_train:n_train+n_val], Y[n_train+n_val:]
    
    # Test Level 1: Ridge regression
    print("\nLevel 1: Ridge regression")
    mapping1 = RidgeRegressionMapping(alpha=1.0)
    mapping1.fit(X_train, Y_train)
    r2_1 = mapping1.score(X_test, Y_test)
    print(f"R²: {r2_1:.4f}")
    
    # Test Level 2: Kernel ridge regression
    print("\nLevel 2: Kernel ridge regression")
    mapping2 = KernelRidgeMapping(alpha=1.0, bandwidth='median')
    mapping2.fit(X_train, Y_train)
    r2_2 = mapping2.score(X_test, Y_test)
    print(f"R²: {r2_2:.4f}")
    
    # Test Level 3: 1-layer MLP
    print("\nLevel 3: 1-layer MLP")
    mapping3 = MLPMappingTrainer(
        input_dim=d_source, output_dim=d_target, hidden_dims=[512],
        activation='gelu', device='cpu'
    )
    mapping3.fit(X_train, Y_train, X_val, Y_val)
    r2_3 = mapping3.score(X_test, Y_test)
    print(f"R²: {r2_3:.4f}")
    
    # Test Level 4: 2-layer MLP
    print("\nLevel 4: 2-layer MLP")
    mapping4 = MLPMappingTrainer(
        input_dim=d_source, output_dim=d_target, hidden_dims=[1024, 512],
        activation='gelu', device='cpu'
    )
    mapping4.fit(X_train, Y_train, X_val, Y_val)
    r2_4 = mapping4.score(X_test, Y_test)
    print(f"R²: {r2_4:.4f}")
    
    print(f"\nMCP Profile: [{r2_1:.4f}, {r2_2:.4f}, {r2_3:.4f}, {r2_4:.4f}]")
    print("\nMappings test passed!")
