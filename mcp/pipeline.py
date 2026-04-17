"""
Full MCP computation pipeline.
Handles data splits, dimensionality reduction, and all 4 mapping levels.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .mappings import (
    RidgeRegressionMapping, KernelRidgeMapping, MLPMappingTrainer
)
from .metrics import MCPSummaryStatistics


class MCPPipeline:
    """
    Full MCP computation pipeline.
    
    For each (X, Y) pair:
    1. Split: 60% train / 20% val / 20% test
    2. Dimensionality reduction (if needed)
    3. Compute R² for all 4 mapping levels
    4. Compute summary statistics
    5. Repeat for n_splits random splits
    """
    
    def __init__(self, config=None, device='cuda'):
        """
        Args:
            config: Configuration dictionary (from mcp_config.yaml)
            device: Device for MLP training
        """
        self.device = device
        self.config = config or {}
        
        # Default hyperparameters
        self.max_dim = self.config.get('max_dim', 1024)
        self.pca_variance = self.config.get('pca_variance', 0.95)
        self.n_splits = self.config.get('n_splits', 5)
        
        # Mapping hyperparameters
        self.ridge_alpha = self.config.get('ridge_alpha', 1.0)
        self.kernel_alpha = self.config.get('kernel_alpha', 1.0)
        self.kernel_bandwidth = self.config.get('kernel_bandwidth', 'median')
        
        # MLP hyperparameters
        self.mlp_lr = self.config.get('mlp_lr', 5e-4)
        self.mlp_weight_decay = self.config.get('mlp_weight_decay', 1e-4)
        self.mlp_batch_size = self.config.get('mlp_batch_size', 512)
        self.mlp_max_epochs = self.config.get('mlp_max_epochs', 300)
        self.mlp_patience = self.config.get('mlp_patience', 20)
        
        # Summary statistics calculator
        self.stats_calculator = MCPSummaryStatistics()
        
    def reduce_dimensionality(self, X, method='pca'):
        """
        Reduce dimensionality if needed.
        
        Args:
            X: [n_samples, d]
            method: 'pca' or 'none'
        
        Returns:
            X_reduced: [n_samples, d_reduced]
            reducer: Fitted reducer (for applying to other data)
        """
        n_samples, d = X.shape
        
        if d <= self.max_dim or method == 'none':
            return X, None
        
        if method == 'pca':
            reducer = PCA(n_components=self.pca_variance, svd_solver='full')
            X_reduced = reducer.fit_transform(X)
            print(f"  PCA: {d} -> {X_reduced.shape[1]} dimensions (retained {self.pca_variance} variance)")
            return X_reduced, reducer
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    def split_data(self, X, Y, stratify=None):
        """
        Split data into train/val/test.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
            stratify: Optional labels for stratified split
        
        Returns:
            X_train, X_val, X_test, Y_train, Y_val, Y_test
        """
        # First split: train vs (val + test)
        if stratify is not None:
            X_train, X_temp, Y_train, Y_temp, s_train, s_temp = train_test_split(
                X, Y, stratify, test_size=0.4, random_state=None
            )
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_temp, Y_temp, test_size=0.5, random_state=None
            )
        else:
            X_train, X_temp, Y_train, Y_temp = train_test_split(
                X, Y, test_size=0.4, random_state=None
            )
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_temp, Y_temp, test_size=0.5, random_state=None
            )
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def compute_mcp_single_split(self, X, Y, stratify=None, reducer_X=None, reducer_Y=None):
        """
        Compute MCP for a single train/val/test split.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
            stratify: Optional labels for stratified split
            reducer_X: Optional fitted reducer for X
            reducer_Y: Optional fitted reducer for Y
        
        Returns:
            r2_profile: [R²₁, R²₂, R²₃, R²₄]
        """
        # Apply dimensionality reduction if reducers provided
        if reducer_X is not None:
            X = reducer_X.transform(X)
        if reducer_Y is not None:
            Y = reducer_Y.transform(Y)
        
        # Split data
        X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(X, Y, stratify)

        # Z-score standardize: fit on train, apply to all splits
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0) + 1e-8
        X_train, X_val, X_test = (X_train - X_mean) / X_std, (X_val - X_mean) / X_std, (X_test - X_mean) / X_std
        Y_train, Y_val, Y_test = (Y_train - Y_mean) / Y_std, (Y_val - Y_mean) / Y_std, (Y_test - Y_mean) / Y_std

        d_source = X_train.shape[1]
        d_target = Y_train.shape[1]

        # Level 1: Ridge regression
        mapping1 = RidgeRegressionMapping(alpha=self.ridge_alpha)
        mapping1.fit(X_train, Y_train)
        r2_1 = mapping1.score(X_test, Y_test)
        
        # Level 2: Kernel ridge regression
        mapping2 = KernelRidgeMapping(alpha=self.kernel_alpha, bandwidth=self.kernel_bandwidth)
        mapping2.fit(X_train, Y_train)
        r2_2 = mapping2.score(X_test, Y_test)
        
        # Level 3: 1-layer MLP
        mapping3 = MLPMappingTrainer(
            input_dim=d_source, output_dim=d_target, hidden_dims=[512],
            activation='gelu', lr=self.mlp_lr, weight_decay=self.mlp_weight_decay,
            batch_size=self.mlp_batch_size, max_epochs=self.mlp_max_epochs,
            patience=self.mlp_patience, device=self.device
        )
        mapping3.fit(X_train, Y_train, X_val, Y_val)
        r2_3 = mapping3.score(X_test, Y_test)
        
        # Level 4: 2-layer MLP
        mapping4 = MLPMappingTrainer(
            input_dim=d_source, output_dim=d_target, hidden_dims=[1024, 512],
            activation='gelu', lr=self.mlp_lr, weight_decay=self.mlp_weight_decay,
            batch_size=self.mlp_batch_size, max_epochs=self.mlp_max_epochs,
            patience=self.mlp_patience, device=self.device
        )
        mapping4.fit(X_train, Y_train, X_val, Y_val)
        r2_4 = mapping4.score(X_test, Y_test)
        
        return [r2_1, r2_2, r2_3, r2_4]
    
    def compute_mcp(self, X, Y, stratify=None, reduce_dim=True):
        """
        Compute MCP with multiple random splits.
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
            stratify: Optional labels for stratified split
            reduce_dim: Whether to apply dimensionality reduction
        
        Returns:
            results: Dictionary with:
                - r2_profiles: [n_splits, 4] array of R² values
                - r2_mean: Mean R² across splits
                - r2_std: Std R² across splits
                - stats_mean: Mean summary statistics
                - stats_std: Std summary statistics
        """
        # Dimensionality reduction
        if reduce_dim:
            X, reducer_X = self.reduce_dimensionality(X, method='pca')
            Y, reducer_Y = self.reduce_dimensionality(Y, method='pca')
        else:
            reducer_X, reducer_Y = None, None
        
        # Compute MCP for multiple splits
        r2_profiles = []
        for i in tqdm(range(self.n_splits), desc="Computing MCP splits"):
            r2_profile = self.compute_mcp_single_split(X, Y, stratify, reducer_X, reducer_Y)
            r2_profiles.append(r2_profile)
        
        r2_profiles = np.array(r2_profiles)
        
        # Compute statistics
        r2_mean = np.mean(r2_profiles, axis=0)
        r2_std = np.std(r2_profiles, axis=0)
        
        stats_mean = self.stats_calculator.compute(r2_mean)
        stats_std = self.stats_calculator.compute(r2_std)
        
        results = {
            'r2_profiles': r2_profiles,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'stats_mean': stats_mean,
            'stats_std': stats_std
        }
        
        return results
    
    def compute_mcp_directional(self, X, Y, stratify=None, reduce_dim=True):
        """
        Compute MCP in both directions (X→Y and Y→X).
        
        Args:
            X: [n_samples, d_source]
            Y: [n_samples, d_target]
            stratify: Optional labels for stratified split
            reduce_dim: Whether to apply dimensionality reduction
        
        Returns:
            results: Dictionary with 'forward' and 'reverse' results
        """
        print("Computing MCP (X → Y)...")
        forward = self.compute_mcp(X, Y, stratify, reduce_dim)
        
        print("Computing MCP (Y → X)...")
        reverse = self.compute_mcp(Y, X, stratify, reduce_dim)
        
        results = {
            'forward': forward,
            'reverse': reverse
        }
        
        return results


if __name__ == "__main__":
    # Test MCP pipeline
    print("Testing MCP pipeline...")
    
    # Create dummy data
    n_samples = 1000
    d_source = 512
    d_target = 512
    
    X = np.random.randn(n_samples, d_source)
    Y = 0.8 * X + 0.2 * np.random.randn(n_samples, d_target)  # Mostly linear
    
    # Create pipeline
    pipeline = MCPPipeline(config={'n_splits': 3}, device='cpu')
    
    # Compute MCP
    results = pipeline.compute_mcp(X, Y, reduce_dim=False)
    
    print("\nMCP Results:")
    print(f"R² mean: {results['r2_mean']}")
    print(f"R² std:  {results['r2_std']}")
    print("\nSummary Statistics (mean):")
    for key, value in results['stats_mean'].items():
        print(f"  {key}: {value:.4f}")
    
    # Test directional MCP
    print("\nTesting directional MCP...")
    results_dir = pipeline.compute_mcp_directional(X, Y, reduce_dim=False)
    
    print("\nForward (X → Y):")
    print(f"  L: {results_dir['forward']['stats_mean']['L']:.4f}")
    print(f"  K: {results_dir['forward']['stats_mean']['K']:.4f}")
    print(f"  G: {results_dir['forward']['stats_mean']['G']:.4f}")
    
    print("\nReverse (Y → X):")
    print(f"  L: {results_dir['reverse']['stats_mean']['L']:.4f}")
    print(f"  K: {results_dir['reverse']['stats_mean']['K']:.4f}")
    print(f"  G: {results_dir['reverse']['stats_mean']['G']:.4f}")
    
    print("\nMCP pipeline test passed!")
