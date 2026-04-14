"""
Delay Embedding Model (DEM) - Built from scratch.
Motivated by Takens (1981) and Ostrow et al. (2024).
Context integration through explicit delay-coordinate reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DelayBuffer(nn.Module):
    """
    Circular buffer for exponential delay concatenation.
    Maintains O(1) memory regardless of sequence length.
    """
    
    def __init__(self, buffer_size=64):
        """
        Args:
            buffer_size: Size of circular buffer
        """
        super().__init__()
        self.buffer_size = buffer_size
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, seq_len, embed_dim]
        
        Returns:
            delayed: [batch_size, seq_len, n_delays * embed_dim]
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Initialize buffer
        buffer = torch.zeros(batch_size, self.buffer_size, embed_dim, device=embeddings.device)
        
        # Delays to extract: [1, 2, 4, 8, 16, 32]
        delays = [1, 2, 4, 8, 16, 32]
        n_delays = len(delays)
        
        # Collect delayed embeddings
        delayed_list = []
        
        for t in range(seq_len):
            # Update buffer with current embedding
            buffer_pos = t % self.buffer_size
            buffer[:, buffer_pos, :] = embeddings[:, t, :]
            
            # Extract delayed embeddings
            delayed_embs = []
            for delay in delays:
                if delay <= t:
                    delay_pos = (t - delay) % self.buffer_size
                    delayed_embs.append(buffer[:, delay_pos, :])
                else:
                    # For early timesteps, use current embedding (no delay available)
                    delayed_embs.append(embeddings[:, t, :])
            
            # Concatenate all delays
            delayed_t = torch.cat(delayed_embs, dim=-1)  # [batch_size, n_delays * embed_dim]
            delayed_list.append(delayed_t)
        
        # Stack across time
        delayed = torch.stack(delayed_list, dim=1)  # [batch_size, seq_len, n_delays * embed_dim]
        
        return delayed


class TemporalMixingBlock(nn.Module):
    """
    Feedforward temporal mixing block.
    LayerNorm → Linear → GELU → Linear → residual
    """
    
    def __init__(self, dim, expansion=4):
        """
        Args:
            dim: Model dimension
            expansion: Expansion factor for hidden layer
        """
        super().__init__()
        hidden_dim = dim * expansion
        
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        
        Returns:
            out: [batch_size, seq_len, dim]
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual


class DelayEmbeddingModel(nn.Module):
    """
    Delay Embedding Model (DEM).
    
    Architecture:
        Input tokens
            → BPE embedding
            → Exponential delay concatenation: [e_t, e_{t-1}, e_{t-2}, e_{t-4}, e_{t-8}, e_{t-16}, e_{t-32}]
            → Circular buffer of size 64
            → Learned linear projection + LayerNorm
            → N feedforward temporal mixing blocks
            → Final LayerNorm → output projection → vocabulary logits
    """
    
    def __init__(self, vocab_size=50257, model_dim=512, n_mixing_blocks=8, 
                 buffer_size=64, max_seq_len=256, dropout=0.1):
        """
        Args:
            vocab_size: Vocabulary size (50257 for GPT-2 BPE)
            model_dim: Model dimension
            n_mixing_blocks: Number of temporal mixing blocks
            buffer_size: Circular buffer size
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.n_mixing_blocks = n_mixing_blocks
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        
        # Delay buffer (6 delays: 1, 2, 4, 8, 16, 32)
        self.delay_buffer = DelayBuffer(buffer_size=buffer_size)
        self.n_delays = 6
        
        # Projection from delay space to model space
        self.delay_projection = nn.Linear(self.n_delays * model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        
        # Temporal mixing blocks
        self.mixing_blocks = nn.ModuleList([
            TemporalMixingBlock(model_dim, expansion=4)
            for _ in range(n_mixing_blocks)
        ])
        
        # Final layer norm and output projection
        self.norm_final = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, return_hidden_states=False):
        """
        Args:
            idx: [batch_size, seq_len] token indices
            return_hidden_states: If True, return hidden states from all blocks
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden_states: Optional list of [batch_size, seq_len, model_dim]
        """
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        x = self.token_embedding(idx)  # [batch_size, seq_len, model_dim]
        x = self.dropout(x)
        
        # Delay embedding
        x_delayed = self.delay_buffer(x)  # [batch_size, seq_len, n_delays * model_dim]
        
        # Project to model dimension
        x = self.delay_projection(x_delayed)
        x = self.norm1(x)
        x = self.dropout(x)
        
        # Collect hidden states if requested
        hidden_states = []
        if return_hidden_states:
            hidden_states.append(x.clone())
        
        # Temporal mixing blocks
        for block in self.mixing_blocks:
            x = block(x)
            x = self.dropout(x)
            if return_hidden_states:
                hidden_states.append(x.clone())
        
        # Final layer norm and output
        x = self.norm_final(x)
        logits = self.output_projection(x)
        
        if return_hidden_states:
            return logits, hidden_states
        else:
            return logits
    
    def get_num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test DEM architecture
    print("Testing Delay Embedding Model...")
    
    # Create model
    model = DelayEmbeddingModel(
        vocab_size=50257,
        model_dim=512,
        n_mixing_blocks=8,
        buffer_size=64,
        max_seq_len=256
    )
    
    print(f"Total parameters: {model.get_num_params() / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 4
    seq_len = 256
    idx = torch.randint(0, 50257, (batch_size, seq_len))
    
    logits, hidden_states = model(idx, return_hidden_states=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Number of hidden state layers: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")
    
    # Test without hidden states
    logits_only = model(idx)
    print(f"Logits-only shape: {logits_only.shape}")
    
    print("\nDEM architecture test passed!")
