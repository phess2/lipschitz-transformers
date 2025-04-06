"""Defines FineWeb data loader adapted to JAX (credit to modded-nanogpt for PyTorch version)"""

import jax
import jax.numpy as jnp
import numpy as np
import glob
import sys
import os
from huggingface_hub import hf_hub_download
from typing import Iterator, Tuple, List

def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        raise ValueError("ERROR: magic number mismatch in the data .bin file!")
    assert header[1] == 1, "unsupported version"
    return header[2]  # number of tokens

def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class FineWebDataset:
    """JAX dataset for FineWeb tokens across multiple shards."""
    
    def __init__(self, filename_pattern: str, context_length: int, process_rank: int = 0, num_processes: int = 1):
        self.context_length = context_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files match pattern {filename_pattern}"
        
        # Load and validate all data shards, count number of tokens in total
        self.ntok_total = sum(_peek_data_shard(f) for f in self.files)
        
        # Initialize with first shard
        self.reset()
    
    def __len__(self):
        return self.ntok_total // (self.context_length * self.num_processes)
    
    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.context_length
        self.tokens = _load_data_shard(self.files[self.current_shard])
    
    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.context_length
        self.tokens = _load_data_shard(self.files[self.current_shard])
    
    def next_batch(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B = batch_size
        T = self.context_length
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = jnp.array(buf.astype(np.int32))
        x = buf[:-1].reshape(B, T)  # inputs
        y = buf[1:].reshape(B, T)   # targets
        
        # Advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        
        return x, y

class FineWebDataLoader:
    """JAX dataloader for FineWeb tokens."""
    
    def __init__(self, dataset: FineWebDataset, batch_size: int, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        num_batches = len(self.dataset) // self.batch_size if self.drop_last \
            else (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        for _ in range(num_batches):
            yield self.dataset.next_batch(self.batch_size)

def cross_entropy_loss(model, w, inputs, targets):
        logits = model(inputs, w)  # shape is [batch, seq_len, vocab_size]
        batch_indices = jnp.arange(logits.shape[0])[:, None]  # shape is [batch, 1]
        seq_indices = jnp.arange(logits.shape[1])[None, :]    # shape is [1, seq_len]
        losses = -logits[batch_indices, seq_indices, targets] + jax.nn.logsumexp(logits, axis=-1)  # shape is [batch, seq_len]
        return losses.mean()

def load_fineweb(sequence_length: int = 1024, batch_size: int = 512):
    train_dataset = FineWebDataset('data/fineweb10B/fineweb_train_*.bin', sequence_length)
    val_dataset = FineWebDataset('data/fineweb10B/fineweb_val_*.bin', sequence_length)
    
    train_loader = FineWebDataLoader(train_dataset, batch_size=batch_size)
    val_loader = FineWebDataLoader(val_dataset, batch_size=batch_size)
    
    return {
        "train_loader": train_loader,
        "test_loader": val_loader,
        "loss": cross_entropy_loss
    }

def get(fname: str) -> None:
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                       repo_type="dataset", local_dir=local_dir)

if __name__ == "__main__":
    get("fineweb_val_%06d.bin" % 0)
    num_chunks = 20  # full fineweb10B. Each chunk is ~98.5M tokens
    if len(sys.argv) >= 2:  # we can pass an argument to download less
        num_chunks = int(sys.argv[1])
    for i in range(1, num_chunks):
        get("fineweb_train_%06d.bin" % i)
