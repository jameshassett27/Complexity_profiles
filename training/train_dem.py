"""
Training script for Delay Embedding Model (DEM).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import yaml

from models.dem import DelayEmbeddingModel
from data.wikitext103 import get_dataloader


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, y, ignore_index=-1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * x.size(0)
        total_tokens += x.size(0)
    
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluation"):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = nn.functional.cross_entropy(logits, y, ignore_index=-1)
            
            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0)
    
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--n_mixing_blocks', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced for GPU memory
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
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test run: reduce tokens for quick validation
    if args.test_run:
        args.total_tokens = 1000000  # 1M tokens
        args.eval_interval = 1000
        args.save_interval = 5000
        print("Running in TEST mode (1M tokens)")
    
    # Create model
    print("Creating DEM model...")
    model = DelayEmbeddingModel(
        vocab_size=50257,
        model_dim=args.model_dim,
        n_mixing_blocks=args.n_mixing_blocks,
        buffer_size=64,
        max_seq_len=256
    )
    model = model.to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create data loaders
    print("Loading WikiText-103...")
    train_loader = get_dataloader(
        split='train',
        sequence_length=256,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed
    )
    val_loader = get_dataloader(
        split='validation',
        sequence_length=256,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine learning rate schedule
    total_steps = args.total_tokens // (args.batch_size * 256)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.final_lr
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Target tokens: {args.total_tokens / 1e9:.2f}B")
    
    # Training loop
    step = 0
    tokens_seen = 0
    best_val_ppl = float('inf')
    val_ppl = float('inf')  # Initialize to prevent UnboundLocalError
    patience_counter = 0
    max_patience = 5
    
    print("\nStarting training...")
    while tokens_seen < args.total_tokens:
        # Train for eval_interval steps
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            
            logits = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = nn.functional.cross_entropy(logits, y, ignore_index=-1)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step += 1
            tokens_seen += x.size(0) * x.size(1)
            
            if step % args.eval_interval == 0:
                # Evaluate
                train_ppl = train_epoch(model, train_loader, optimizer, scheduler, args.device)
                val_ppl = evaluate(model, val_loader, args.device)
                
                print(f"\nStep {step}, Tokens: {tokens_seen / 1e6:.1f}M")
                print(f"  Train PPL: {train_ppl:.2f}")
                print(f"  Val PPL:   {val_ppl:.2f}")
                print(f"  LR:        {optimizer.param_groups[0]['lr']:.6f}")
                
                # Save checkpoint
                if step % args.save_interval == 0:
                    checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f'dem_seed{args.seed}_step{step}.pt'
                    )
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_ppl': val_ppl,
                        'tokens_seen': tokens_seen
                    }, checkpoint_path)
                    print(f"  Saved checkpoint: {checkpoint_path}")
                
                # Early stopping
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered (patience {max_patience})")
                    break
            
            if tokens_seen >= args.total_tokens:
                break
        
        if patience_counter >= max_patience:
            break
    
    # Final save
    final_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'dem_seed{args.seed}_final.pt'
    )
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_ppl': val_ppl,
        'tokens_seen': tokens_seen
    }, final_checkpoint_path)
    
    # Always evaluate at end of training
    val_ppl = evaluate(model, val_loader, args.device)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl

    print(f"\nTraining complete!")
    print(f"Final validation perplexity: {val_ppl:.2f}")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Total tokens seen: {tokens_seen / 1e6:.1f}M")


if __name__ == "__main__":
    main()
