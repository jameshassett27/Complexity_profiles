"""
MCP summary statistics.
Computes asymptote (A), linearity index (L), kernel gain (K), and learning gain (G).
"""

import numpy as np


class MCPSummaryStatistics:
    """Compute summary statistics from MCP profile."""
    
    def __init__(self, eps=1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps
        
    def compute(self, r2_profile):
        """
        Compute all summary statistics from R² profile.
        
        Args:
            r2_profile: List or array of 4 R² values [R²₁, R²₂, R²₃, R²₄]
        
        Returns:
            stats: Dictionary with A, L, K, G
        """
        r2_profile = np.array(r2_profile)
        assert len(r2_profile) == 4, "R² profile must have 4 values"
        
        R2_1, R2_2, R2_3, R2_4 = r2_profile
        
        # Asymptote: Total shared information accessible by any mapping
        A = R2_4
        
        # Linearity Index: Fraction of shared information that is linearly accessible
        L = R2_1 / max(R2_4, self.eps)
        
        # Kernel Gain: Fraction of nonlinear structure captured by fixed kernels
        nonlinear_component = max(R2_4 - R2_1, self.eps)
        K = (R2_2 - R2_1) / nonlinear_component
        
        # Learning Gain: Fraction of nonlinear structure requiring learned mappings
        G = (R2_4 - R2_2) / nonlinear_component
        
        stats = {
            'A': A,
            'L': L,
            'K': K,
            'G': G,
            'R2_1': R2_1,
            'R2_2': R2_2,
            'R2_3': R2_3,
            'R2_4': R2_4
        }
        
        return stats
    
    def compute_batch(self, r2_profiles):
        """
        Compute summary statistics for multiple profiles.
        
        Args:
            r2_profiles: Array of shape [n_profiles, 4]
        
        Returns:
            stats: Dictionary with arrays for each statistic
        """
        r2_profiles = np.array(r2_profiles)
        assert r2_profiles.shape[1] == 4, "R² profiles must have 4 columns"
        
        R2_1 = r2_profiles[:, 0]
        R2_2 = r2_profiles[:, 1]
        R2_3 = r2_profiles[:, 2]
        R2_4 = r2_profiles[:, 3]
        
        # Asymptote
        A = R2_4
        
        # Linearity Index
        L = R2_1 / np.maximum(R2_4, self.eps)
        
        # Kernel Gain
        nonlinear_component = np.maximum(R2_4 - R2_1, self.eps)
        K = (R2_2 - R2_1) / nonlinear_component
        
        # Learning Gain
        G = (R2_4 - R2_2) / nonlinear_component
        
        stats = {
            'A': A,
            'L': L,
            'K': K,
            'G': G,
            'R2_1': R2_1,
            'R2_2': R2_2,
            'R2_3': R2_3,
            'R2_4': R2_4
        }
        
        return stats


def print_mcp_profile(stats):
    """
    Pretty print MCP profile and summary statistics.
    
    Args:
        stats: Dictionary from MCPSummaryStatistics.compute()
    """
    print("MCP Profile:")
    print(f"  R²₁ (Linear): {stats['R2_1']:.4f}")
    print(f"  R²₂ (Kernel): {stats['R2_2']:.4f}")
    print(f"  R²₃ (MLP-1):  {stats['R2_3']:.4f}")
    print(f"  R²₄ (MLP-2):  {stats['R2_4']:.4f}")
    print("\nSummary Statistics:")
    print(f"  Asymptote (A):         {stats['A']:.4f}")
    print(f"  Linearity Index (L):    {stats['L']:.4f}")
    print(f"  Kernel Gain (K):        {stats['K']:.4f}")
    print(f"  Learning Gain (G):       {stats['G']:.4f}")


if __name__ == "__main__":
    # Test summary statistics
    print("Testing MCP summary statistics...")
    
    calculator = MCPSummaryStatistics()
    
    # Example 1: Mostly linear relationship
    r2_linear = [0.85, 0.86, 0.87, 0.88]
    stats_linear = calculator.compute(r2_linear)
    print("\nExample 1: Mostly linear")
    print_mcp_profile(stats_linear)
    
    # Example 2: Highly nonlinear
    r2_nonlinear = [0.20, 0.40, 0.70, 0.85]
    stats_nonlinear = calculator.compute(r2_nonlinear)
    print("\nExample 2: Highly nonlinear")
    print_mcp_profile(stats_nonlinear)
    
    # Example 3: Kernel captures most nonlinearity
    r2_kernel = [0.30, 0.75, 0.78, 0.80]
    stats_kernel = calculator.compute(r2_kernel)
    print("\nExample 3: Kernel captures most nonlinearity")
    print_mcp_profile(stats_kernel)
    
    # Example 4: Learning required
    r2_learning = [0.30, 0.45, 0.70, 0.85]
    stats_learning = calculator.compute(r2_learning)
    print("\nExample 4: Learning required")
    print_mcp_profile(stats_learning)
    
    # Test batch computation
    print("\nTesting batch computation...")
    r2_batch = np.array([r2_linear, r2_nonlinear, r2_kernel, r2_learning])
    stats_batch = calculator.compute_batch(r2_batch)
    print(f"Batch shape: {r2_batch.shape}")
    print(f"L values: {stats_batch['L']}")
    print(f"K values: {stats_batch['K']}")
    print(f"G values: {stats_batch['G']}")
    
    print("\nSummary statistics test passed!")
