"""
Run MCP pipeline on extracted hidden states.

Pilot scope per experimental_protocol_v3_final.md Section 5.4:
- Cross-architecture pairs at matched layers (DEM↔GPT2, DEM↔LSTM, GPT2↔LSTM)
- Self-mapping control: MCP(X, X + small noise) -> should be ~1.0 at all levels (ceiling)
- Shuffled control: MCP(X, permute_rows(Y)) -> should be ~0 at all levels (floor)
- Goal: verify MCP shows meaningful variation across the 4 levels, not flat
"""

import numpy as np
import os
import argparse
import json
from itertools import combinations

from .pipeline import MCPPipeline


def load_hidden_states(hidden_states_dir, models, layers):
    """Load saved hidden state arrays. Returns {model: {layer: array}}."""
    states = {}
    for model in models:
        states[model] = {}
        for layer in layers:
            path = os.path.join(hidden_states_dir, f'{model}_layer{layer}.npy')
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            arr = np.load(path)
            states[model][layer] = arr
            print(f"Loaded {model} layer {layer}: shape {arr.shape}")
    return states


def run_mcp_pair(pipeline, X, Y, name):
    """Run MCP on (X, Y), return serializable dict."""
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    print(f"\n[{name}]  n_samples={n}, X.shape={X.shape}, Y.shape={Y.shape}")

    res = pipeline.compute_mcp(X, Y, reduce_dim=True)
    r2 = res['r2_mean']
    print(f"  R²:  ridge={r2[0]:.3f}  kernel={r2[1]:.3f}  1-MLP={r2[2]:.3f}  2-MLP={r2[3]:.3f}")
    print(f"  L={res['stats_mean']['L']:.3f}  K={res['stats_mean']['K']:.3f}  G={res['stats_mean']['G']:.3f}")

    return {
        'name': name,
        'n_samples': int(n),
        'r2_mean': res['r2_mean'].tolist(),
        'r2_std': res['r2_std'].tolist(),
        'r2_profiles': res['r2_profiles'].tolist(),
        'stats_mean': res['stats_mean'],
        'stats_std': res['stats_std'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_states_dir', type=str, default='results/hidden_states')
    parser.add_argument('--output_dir', type=str, default='results/mcp')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--models', type=str, nargs='+', default=['dem', 'gpt2', 'lstm'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_controls', action='store_true',
                        help='Skip self-mapping and shuffled controls')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    states = load_hidden_states(args.hidden_states_dir, args.models, args.layers)
    pipeline = MCPPipeline(config={'n_splits': args.n_splits}, device=args.device)

    results = {'cross_architecture': [], 'controls': []}

    # --- Controls (protocol Section 5.8) ---
    if not args.skip_controls:
        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)

        # Use GPT-2 layer 8 as the canonical control target; fall back to any available.
        control_model, control_layer = None, None
        for m in args.models:
            if m in states and states[m]:
                control_model = m
                control_layer = sorted(states[m].keys())[len(states[m]) // 2]
                break

        if control_model is None:
            print("No hidden states available; skipping controls.")
        else:
            X = states[control_model][control_layer]
            print(f"\nControls use {control_model} layer {control_layer} as reference.")

            # Self-mapping ceiling: MCP(X, X + small noise). Expected: ~1.0 everywhere.
            rng = np.random.default_rng(0)
            X_noisy = X + 0.01 * rng.standard_normal(X.shape).astype(X.dtype)
            results['controls'].append(
                run_mcp_pair(pipeline, X, X_noisy, f'self_mapping({control_model}_L{control_layer})')
            )

            # Shuffled floor: MCP(X, permute_rows(X)). Expected: ~0 everywhere.
            perm = rng.permutation(len(X))
            results['controls'].append(
                run_mcp_pair(pipeline, X, X[perm], f'shuffled({control_model}_L{control_layer})')
            )

    # --- Cross-architecture pairs at matched layers ---
    print("\n" + "=" * 60)
    print("CROSS-ARCHITECTURE MCP")
    print("=" * 60)

    for model_a, model_b in combinations(args.models, 2):
        if model_a not in states or model_b not in states:
            continue
        common_layers = sorted(set(states[model_a].keys()) & set(states[model_b].keys()))
        for layer in common_layers:
            X = states[model_a][layer]
            Y = states[model_b][layer]
            name = f'{model_a}_vs_{model_b}_L{layer}'
            results['cross_architecture'].append(run_mcp_pair(pipeline, X, Y, name))

    # Save results
    out_path = os.path.join(args.output_dir, 'mcp_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: R² profiles (ridge / kernel / 1-MLP / 2-MLP)")
    print("=" * 60)
    for section in ('controls', 'cross_architecture'):
        if not results[section]:
            continue
        print(f"\n-- {section} --")
        for r in results[section]:
            r2 = r['r2_mean']
            L = r['stats_mean']['L']
            print(f"  {r['name']:40s}  {r2[0]:.3f} / {r2[1]:.3f} / {r2[2]:.3f} / {r2[3]:.3f}   L={L:.3f}")

    # Decision gate check (per PILOT_PLAN.md)
    print("\n" + "=" * 60)
    print("DECISION GATE")
    print("=" * 60)
    print("Checking that MCP shows meaningful variation across levels:")
    for r in results['cross_architecture']:
        r2 = r['r2_mean']
        spread = max(r2) - min(r2)
        status = "OK" if spread > 0.05 else "FLAT (concern)"
        print(f"  {r['name']:40s}  spread={spread:.3f}  [{status}]")


if __name__ == "__main__":
    main()
