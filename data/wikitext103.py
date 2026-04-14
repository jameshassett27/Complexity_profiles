"""
WikiText-103 data pipeline with GPT-2 BPE tokenizer.
Shared across ALL models for fair comparison.
"""

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class WikiText103Dataset(Dataset):
    """WikiText-103 dataset with GPT-2 BPE tokenization."""
    
    def __init__(self, split='train', sequence_length=256, tokenizer=None, seed=42):
        """
        Args:
            split: 'train', 'validation', or 'test'
            sequence_length: Context window size
            tokenizer: GPT-2 tokenizer (if None, will load default)
            seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.seed = seed
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load WikiText-103
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        
        # Tokenize all texts
        print(f"Tokenizing {split} split...")
        all_tokens = []
        for text in tqdm(dataset['text']):
            if text.strip():  # Skip empty lines
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens in {split} split: {len(self.tokens)}")
        print(f"Vocab size: {len(self.tokenizer)}")
        
    def __len__(self):
        # Number of sequences we can extract
        return max(0, len(self.tokens) - self.sequence_length)
    
    def __getitem__(self, idx):
        # Get sequence of length sequence_length
        x = self.tokens[idx:idx + self.sequence_length]
        # Target is next token (shifted by 1)
        y = self.tokens[idx + 1:idx + self.sequence_length + 1]
        
        return x, y


class WikiText103SentenceDataset(Dataset):
    """
    WikiText-103 dataset for sentence-level extraction (Task A).
    Extracts individual sentences for representation extraction.
    """
    
    def __init__(self, split='validation', n_sentences=10000, seed=42, tokenizer=None):
        """
        Args:
            split: 'validation' or 'test'
            n_sentences: Number of sentences to sample
            seed: Random seed for reproducibility
            tokenizer: GPT-2 tokenizer
        """
        self.n_sentences = n_sentences
        self.seed = seed
        np.random.seed(seed)
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load WikiText-103
        print(f"Loading WikiText-103 {split} split for sentence extraction...")
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        
        # Extract non-empty sentences
        sentences = [text.strip() for text in dataset['text'] if text.strip()]
        print(f"Total sentences in {split}: {len(sentences)}")
        
        # Sample n_sentences with fixed seed
        if n_sentences < len(sentences):
            indices = np.random.choice(len(sentences), size=n_sentences, replace=False)
            self.sentences = [sentences[i] for i in indices]
        else:
            self.sentences = sentences
        
        print(f"Sampled {len(self.sentences)} sentences")
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False, return_tensors='pt')
        return tokens.squeeze(0), sentence


def get_dataloader(split='train', sequence_length=256, batch_size=64, shuffle=True, seed=42):
    """
    Create a DataLoader for WikiText-103.
    
    Args:
        split: 'train', 'validation', or 'test'
        sequence_length: Context window size
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed
    
    Returns:
        DataLoader
    """
    dataset = WikiText103Dataset(split=split, sequence_length=sequence_length, seed=seed)
    
    # Set generator seed for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def get_sentence_dataloader(split='validation', n_sentences=10000, batch_size=128, seed=42):
    """
    Create a DataLoader for sentence-level extraction (Task A).
    
    Args:
        split: 'validation' or 'test'
        n_sentences: Number of sentences
        batch_size: Batch size
        seed: Random seed
    
    Returns:
        DataLoader
    """
    dataset = WikiText103SentenceDataset(split=split, n_sentences=n_sentences, seed=seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_sentences
    )
    
    return dataloader


def collate_sentences(batch):
    """
    Collate function for sentence dataset.
    Pads sequences to max length in batch.
    """
    tokens_list, sentences = zip(*batch)
    
    # Get max length
    max_len = max(t.size(0) for t in tokens_list)
    
    # Pad sequences
    padded_tokens = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, :tokens.size(0)] = tokens
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros_like(padded_tokens)
    for i, tokens in enumerate(tokens_list):
        attention_mask[i, :tokens.size(0)] = 1
    
    return padded_tokens, attention_mask, sentences


if __name__ == "__main__":
    # Test the data pipeline
    print("Testing WikiText-103 data pipeline...")
    
    # Test training data
    train_loader = get_dataloader('train', sequence_length=256, batch_size=64, seed=42)
    batch_x, batch_y = next(iter(train_loader))
    print(f"Training batch shape: {batch_x.shape}, {batch_y.shape}")
    print(f"Sample tokens: {batch_x[0, :10]}")
    
    # Test sentence data
    sentence_loader = get_sentence_dataloader('validation', n_sentences=100, batch_size=10, seed=42)
    tokens, mask, sentences = next(iter(sentence_loader))
    print(f"\nSentence batch shape: {tokens.shape}, {mask.shape}")
    print(f"Sample sentence: {sentences[0][:100]}...")
    
    print("\nData pipeline test passed!")
