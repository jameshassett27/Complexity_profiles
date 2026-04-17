"""
Run CKA baseline on extracted hidden states.
Computes linear and RBF-kernel CKA for every pair that MCP evaluates.
"""

import numpy as np
import os
import argparse
import json
from itertools import combinations

from .cka import linear_cka, rbf_cka


def load_hidden_states(hidden_states_dir, models, layers):
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


def run_cka_pair(X, Y, name):
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    lcka = linear_cka(X, Y)
    rcka = rbf_cka(X, Y)
    print(f"  {name:40s}  linear={lcka:.3f}  rbf={rcka:.3f}")
    return {'name': name, 'n_samples': int(n), 'linear_cka': lcka, 'rbf_cka': rcka}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_states_dir', type=str, default='results/hidden_states')
    parser.add_argument('--output_dir', type=str, default='results/cka')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--models', type=str, nargs='+', default=['dem', 'gpt2', 'lstm', 'rwkv'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    states = load_hidden_states(args.hidden_states_dir, args.models, args.layers)

    results = {'controls': [], 'cross_architecture': []}

    # Controls
    control_model = None
    for m in args.models:
        if m in states and states[m]:
            control_model = m
            break
    if control_model:
        control_layer = sorted(states[control_model].keys())[len(states[control_model]) // 2]
        X = states[control_model][control_layer]
        rng = np.random.default_rng(0)

        X_noisy = X + 0.01 * rng.standard_normal(X.shape).astype(X.dtype)
        results['controls'].append(run_cka_pair(X, X_noisy, f'self_mapping({control_model}_L{control_layer})'))

        perm = rng.permutation(len(X))
        results['controls'].append(run_cka_pair(X, X[perm], f'shuffled({control_model}_L{control_layer})'))

    # Cross-architecture
    print("\nCROSS-ARCHITECTURE CKA")
    for model_a, model_b in combinations(args.models, 2):
        if model_a not in states or model_b not in states:
            continue
        common_layers = sorted(set(states[model_a].keys()) & set(states[model_b].keys()))
        for layer in common_layers:
            X = states[model_a][layer]
            Y = states[model_b][layer]
            results['cross_architecture'].append(
                run_cka_pair(X, Y, f'{model_a}_vs_{model_b}_L{layer}')
            )

    out_path = os.path.join(args.output_dir, 'cka_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CKA (linear / rbf)")
    print("=" * 60)
    for section in ('controls', 'cross_architecture'):
        if not results[section]:
            continue
        print(f"\n-- {section} --")
        for r in results[section]:
            print(f"  {r['name']:40s}  {r['linear_cka']:.3f} / {r['rbf_cka']:.3f}")


if __name__ == "__main__":
    main()
