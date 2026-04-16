"""
Training script for RWKV-Small using HuggingFace transformers.
Uses the same WikiText-103 data pipeline and training hyperparameters as DEM/GPT-2/LSTM.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import yaml
from transformers import RwkvConfig, RwkvForCausalLM

from data.wikitext103 import get_dataloader


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluation"):
            x, y = x.to(device), y.to(device)

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--final_lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--total_tokens', type=int, default=1000000000)  # 1B tokens
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_run', action='store_true', help='Run small test (1M tokens)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.test_run:
        args.total_tokens = 1000000
        args.eval_interval = 1000
        args.save_interval = 5000
        print("Running in TEST mode (1M tokens)")

    # Create RWKV-Small model
    print("Creating RWKV-Small model...")
    config = RwkvConfig(
        vocab_size=50257,
        context_length=256,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        attention_hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        layer_norm_epsilon=1e-5,
        rescale_every=0,  # Disable rescaling during training
        tie_word_embeddings=False,
    )
    model = RwkvForCausalLM(config)
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    print("Loading WikiText-103...")
    train_loader = get_dataloader(
        split='train', sequence_length=256,
        batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = get_dataloader(
        split='validation', sequence_length=256,
        batch_size=args.batch_size, shuffle=False, seed=args.seed
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    total_steps = args.total_tokens // (args.batch_size * 256)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.final_lr
    )

    print(f"Total training steps: {total_steps}")
    print(f"Target tokens: {args.total_tokens / 1e9:.2f}B")

    step = 0
    tokens_seen = 0
    best_val_ppl = float('inf')
    val_ppl = float('inf')

    print("\nStarting training...")
    while tokens_seen < args.total_tokens:
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            tokens_seen += x.size(0) * x.size(1)

            if step % args.eval_interval == 0:
                val_ppl = evaluate(model, val_loader, args.device)
                print(f"\nStep {step}, Tokens: {tokens_seen / 1e6:.1f}M")
                print(f"  Val PPL: {val_ppl:.2f}")
                print(f"  LR:      {optimizer.param_groups[0]['lr']:.6f}")

                if step % args.save_interval == 0:
                    ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f'rwkv_seed{args.seed}_step{step}.pt'
                    )
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_ppl': val_ppl,
                        'tokens_seen': tokens_seen,
                        'config': config.to_dict(),
                    }, ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl

            if tokens_seen >= args.total_tokens:
                break

    # Final evaluation and checkpoint
    val_ppl = evaluate(model, val_loader, args.device)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl

    final_ckpt_path = os.path.join(
        args.checkpoint_dir, f'rwkv_seed{args.seed}_final.pt'
    )
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_ppl': val_ppl,
        'tokens_seen': tokens_seen,
        'config': config.to_dict(),
    }, final_ckpt_path)

    print(f"\nTraining complete!")
    print(f"Final validation perplexity: {val_ppl:.2f}")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Total tokens seen: {tokens_seen / 1e6:.1f}M")


if __name__ == "__main__":
    main()
