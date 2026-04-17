"""
Extract hidden states from trained models at specified layers.
Uses final-token representation per experimental protocol Section 4.
Saves numpy arrays for MCP pipeline input.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, RwkvConfig, RwkvForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wikitext103 import get_dataloader
from models.dem import DelayEmbeddingModel
from training.train_lstm import LSTMLanguageModel


def final_token(h):
    """Extract final-token hidden state. h: [batch, seq, dim] -> [batch, dim]."""
    return h[:, -1, :].cpu().numpy()


def extract_dem(checkpoint_path, layers, device, n_batches):
    """Extract DEM final-token hidden states.

    DEM has 8 mixing blocks. hidden_states[0] = post-projection,
    hidden_states[1..8] = after each mixing block. Requested layers beyond 8
    are clamped to the final block.
    """
    print(f"Loading DEM from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DelayEmbeddingModel(
        vocab_size=50257, model_dim=512, n_mixing_blocks=8,
        buffer_size=64, max_seq_len=256
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    max_block = model.n_mixing_blocks  # 8
    layer_to_idx = {l: min(l, max_block) for l in layers}

    loader = get_dataloader('validation', sequence_length=256, batch_size=32, shuffle=False, seed=42)
    layer_states = {l: [] for l in layers}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader, desc="DEM extraction")):
            if batch_idx >= n_batches:
                break
            x = x.to(device)
            _, hidden_states = model(x, return_hidden_states=True)
            for l in layers:
                idx = layer_to_idx[l]
                layer_states[l].append(final_token(hidden_states[idx]))

    return {l: np.concatenate(layer_states[l], axis=0) for l in layers}, layer_to_idx


def extract_gpt2(checkpoint_path, layers, device, n_batches):
    """Extract GPT-2 final-token hidden states.

    hidden_states[0] = token+pos embedding, hidden_states[i] = after block i.
    """
    print(f"Loading GPT-2 from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = GPT2Config(
        vocab_size=50257, n_positions=256, n_embd=512,
        n_layer=12, n_head=8,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    max_block = config.n_layer  # 12
    layer_to_idx = {l: min(l, max_block) for l in layers}

    loader = get_dataloader('validation', sequence_length=256, batch_size=32, shuffle=False, seed=42)
    layer_states = {l: [] for l in layers}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader, desc="GPT-2 extraction")):
            if batch_idx >= n_batches:
                break
            x = x.to(device)
            outputs = model(input_ids=x, output_hidden_states=True)
            for l in layers:
                idx = layer_to_idx[l]
                layer_states[l].append(final_token(outputs.hidden_states[idx]))

    return {l: np.concatenate(layer_states[l], axis=0) for l in layers}, layer_to_idx


def extract_lstm(checkpoint_path, layers, device, n_batches):
    """Extract LSTM final-token hidden states at each layer.

    PyTorch's multi-layer LSTM is a single module, so to get per-layer
    per-timestep states we must run each layer independently in a loop,
    feeding each layer's output as the next layer's input.
    """
    print(f"Loading LSTM from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = LSTMLanguageModel(
        vocab_size=50257, embedding_dim=512,
        hidden_dim=512, num_layers=2, dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    num_layers = model.lstm.num_layers  # 4
    hidden_dim = model.lstm.hidden_size
    embed_dim = model.embedding.embedding_dim

    # Build a stack of single-layer LSTMs and copy weights from the original.
    # This lets us capture the full [batch, seq, hidden] output at every layer.
    layer_stack = []
    for i in range(num_layers):
        in_dim = embed_dim if i == 0 else hidden_dim
        single = nn.LSTM(in_dim, hidden_dim, num_layers=1, batch_first=True).to(device).eval()
        # Copy weights from original multi-layer LSTM
        w_ih = getattr(model.lstm, f'weight_ih_l{i}')
        w_hh = getattr(model.lstm, f'weight_hh_l{i}')
        b_ih = getattr(model.lstm, f'bias_ih_l{i}')
        b_hh = getattr(model.lstm, f'bias_hh_l{i}')
        single.weight_ih_l0.data.copy_(w_ih.data)
        single.weight_hh_l0.data.copy_(w_hh.data)
        single.bias_ih_l0.data.copy_(b_ih.data)
        single.bias_hh_l0.data.copy_(b_hh.data)
        layer_stack.append(single)

    # LSTM has 4 layers; map requested layers (e.g. 4, 8, 12) to available ones (1..4)
    layer_to_idx = {l: min(l, num_layers) for l in layers}

    loader = get_dataloader('validation', sequence_length=256, batch_size=32, shuffle=False, seed=42)
    layer_states = {l: [] for l in layers}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader, desc="LSTM extraction")):
            if batch_idx >= n_batches:
                break
            x = x.to(device)
            # Embedding (apply dropout in eval mode = identity, but keep consistent with training)
            current = model.embedding(x)

            # Per-layer outputs: [batch, seq, hidden]
            per_layer_outputs = {}
            for i, lstm_layer in enumerate(layer_stack):
                current, _ = lstm_layer(current)
                per_layer_outputs[i + 1] = current  # 1-indexed

            for l in layers:
                idx = layer_to_idx[l]
                layer_states[l].append(final_token(per_layer_outputs[idx]))

    return {l: np.concatenate(layer_states[l], axis=0) for l in layers}, layer_to_idx


def extract_rwkv(checkpoint_path, layers, device, n_batches):
    """Extract RWKV final-token hidden states.

    RwkvForCausalLM with output_hidden_states=True returns a tuple of length
    num_hidden_layers + 1: hidden_states[0] = embedding, hidden_states[i] = after block i.
    Requested layers beyond num_hidden_layers are clamped to the final block.
    """
    print(f"Loading RWKV from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = RwkvConfig(
        vocab_size=50257,
        context_length=256,
        hidden_size=512,
        num_hidden_layers=12,
        attention_hidden_size=512,
        intermediate_size=512 * 4,
        layer_norm_epsilon=1e-5,
        rescale_every=0,
        tie_word_embeddings=False,
    )
    model = RwkvForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    max_block = config.num_hidden_layers  # 12
    layer_to_idx = {l: min(l, max_block) for l in layers}

    loader = get_dataloader('validation', sequence_length=256, batch_size=32, shuffle=False, seed=42)
    layer_states = {l: [] for l in layers}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(loader, desc="RWKV extraction")):
            if batch_idx >= n_batches:
                break
            x = x.to(device)
            outputs = model(input_ids=x, output_hidden_states=True)
            for l in layers:
                idx = layer_to_idx[l]
                layer_states[l].append(final_token(outputs.hidden_states[idx]))

    return {l: np.concatenate(layer_states[l], axis=0) for l in layers}, layer_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/hidden_states')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of validation batches (32 samples each; 50 = ~1600 samples)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['dem', 'gpt2', 'lstm', 'rwkv'],
                        choices=['dem', 'gpt2', 'lstm', 'rwkv'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extractors = {
        'dem':  (extract_dem,  'dem_seed0_final.pt'),
        'gpt2': (extract_gpt2, 'gpt2_seed0_final.pt'),
        'lstm': (extract_lstm, 'lstm_seed0_final.pt'),
        'rwkv': (extract_rwkv, 'rwkv_seed0_final.pt'),
    }

    layer_map_summary = {}
    for model_name in args.models:
        extractor_fn, ckpt_name = extractors[model_name]
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping.")
            continue

        hidden_states, layer_to_idx = extractor_fn(ckpt_path, args.layers, args.device, args.n_batches)
        layer_map_summary[model_name] = layer_to_idx

        for layer, states in hidden_states.items():
            out_path = os.path.join(args.output_dir, f'{model_name}_layer{layer}.npy')
            np.save(out_path, states)
            print(f"Saved {model_name} layer {layer} (internal idx {layer_to_idx[layer]}): "
                  f"shape {states.shape} -> {out_path}")

    print("\n=== Layer mapping summary ===")
    print("Requested layer -> actual internal layer used:")
    for model_name, mapping in layer_map_summary.items():
        print(f"  {model_name}: {mapping}")
    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
