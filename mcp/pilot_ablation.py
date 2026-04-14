"""
Pilot ablation study for MCP.
Tests robustness to hyperparameter choices:
- MLP hidden sizes
- Activation functions
- Kernel bandwidths
"""

import numpy as np
import torch
from tqdm import tqdm
import yaml
import json
import os

from .pipeline import MCPPipeline


def run_pilot_ablation(X, Y, config_path='configs/mcp_config.yaml', device='cuda'):
    """
    Run pilot ablation study.
    
    Args:
        X: [n_samples, d_source] representations from model A
        Y: [n_samples, d_target] representations from model B
        config_path: Path to MCP config
        device: Device for MLP training
    
    Returns:
        results: Dictionary with ablation results
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pilot_config = config.get('pilot_ablation', {})
    
    # Base config
    base_config = {
        'n_splits': 3,  # Fewer splits for pilot
        'ridge_alpha': 1.0,
        'kernel_alpha': 1.0,
        'kernel_bandwidth': 'median',
        'mlp_lr': 5e-4,
        'mlp_weight_decay': 1e-4,
        'mlp_batch_size': 512,
        'mlp_max_epochs': 100,  # Fewer epochs for pilot
        'mlp_patience': 10
    }
    
    results = {
        'baseline': None,
        'mlp_hidden_sizes': {},
        'activations': {},
        'kernel_bandwidths': {}
    }
    
    # Baseline run
    print("=" * 60)
    print("BASELINE RUN")
    print("=" * 60)
    pipeline = MCPPipeline(config=base_config, device=device)
    baseline_results = pipeline.compute_mcp(X, Y, reduce_dim=False)
    results['baseline'] = {
        'r2_mean': baseline_results['r2_mean'].tolist(),
        'r2_std': baseline_results['r2_std'].tolist(),
        'stats_mean': {k: float(v) for k, v in baseline_results['stats_mean'].items()}
    }
    print(f"Baseline R²: {baseline_results['r2_mean']}")
    print(f"Baseline L: {baseline_results['stats_mean']['L']:.4f}")
    
    # Ablation 1: MLP hidden sizes
    print("\n" + "=" * 60)
    print("ABLATION 1: MLP HIDDEN SIZES")
    print("=" * 60)
    hidden_sizes = pilot_config.get('vary', {}).get('mlp_hidden_sizes', [256, 512, 1024])
    
    for hidden_size in hidden_sizes:
        print(f"\nTesting hidden size: {hidden_size}")
        ablation_config = base_config.copy()
        # Note: This would require modifying the pipeline to accept custom hidden sizes
        # For now, we'll just note this in the results
        results['mlp_hidden_sizes'][hidden_size] = 'TODO: Implement custom hidden sizes'
        print(f"  (Custom hidden sizes not yet implemented in pipeline)")
    
    # Ablation 2: Activations
    print("\n" + "=" * 60)
    print("ABLATION 2: ACTIVATIONS")
    print("=" * 60)
    activations = pilot_config.get('vary', {}).get('activations', ['gelu', 'relu', 'silu'])
    
    for activation in activations:
        print(f"\nTesting activation: {activation}")
        # Note: This would require modifying the pipeline to accept custom activations
        results['activations'][activation] = 'TODO: Implement custom activations'
        print(f"  (Custom activations not yet implemented in pipeline)")
    
    # Ablation 3: Kernel bandwidths
    print("\n" + "=" * 60)
    print("ABLATION 3: KERNEL BANDWIDTHS")
    print("=" * 60)
    bandwidth_multipliers = pilot_config.get('vary', {}).get('kernel_bandwidths', [0.5, 1.0, 2.0])
    
    # First compute median bandwidth from data
    from scipy.spatial.distance import pdist
    pairwise_dist = pdist(X, metric='euclidean')
    median_bandwidth = np.median(pairwise_dist)
    print(f"Median bandwidth: {median_bandwidth:.4f}")
    
    for multiplier in bandwidth_multipliers:
        print(f"\nTesting bandwidth multiplier: {multiplier}x")
        ablation_config = base_config.copy()
        ablation_config['kernel_bandwidth'] = median_bandwidth * multiplier
        
        pipeline = MCPPipeline(config=ablation_config, device=device)
        ablation_results = pipeline.compute_mcp(X, Y, reduce_dim=False)
        
        results['kernel_bandwidths'][multiplier] = {
            'r2_mean': ablation_results['r2_mean'].tolist(),
            'r2_std': ablation_results['r2_std'].tolist(),
            'stats_mean': {k: float(v) for k, v in ablation_results['stats_mean'].items()}
        }
        
        print(f"  R²: {ablation_results['r2_mean']}")
        print(f"  L: {ablation_results['stats_mean']['L']:.4f}")
    
    # Robustness check
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK")
    print("=" * 60)
    robustness_threshold = pilot_config.get('robustness_threshold', 0.05)
    
    baseline_r2 = np.array(results['baseline']['r2_mean'])
    
    for multiplier in bandwidth_multipliers:
        ablation_r2 = np.array(results['kernel_bandwidths'][multiplier]['r2_mean'])
        max_diff = np.max(np.abs(ablation_r2 - baseline_r2))
        is_robust = max_diff < robustness_threshold
        status = "✓ ROBUST" if is_robust else "✗ NOT ROBUST"
        print(f"  Bandwidth {multiplier}x: max diff = {max_diff:.4f} {status}")
    
    return results


def generate_dummy_representations(n_samples=1000, d=512, seed=42):
    """Generate dummy representations for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate somewhat correlated representations
    X = np.random.randn(n_samples, d)
    Y = 0.7 * X + 0.3 * np.random.randn(n_samples, d)
    
    return X, Y


def save_results(results, output_path='results/pilot_ablation.json'):
    """Save ablation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Running pilot ablation study...")
    
    # Generate dummy data for testing
    print("Generating dummy representations...")
    X, Y = generate_dummy_representations(n_samples=500, d=256, seed=42)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Run ablation
    results = run_pilot_ablation(X, Y, device='cpu')
    
    # Save results
    save_results(results)
    
    print("\nPilot ablation complete!")
