import jax
import jax.numpy as jnp

from modula.abstract import Bond

class ReLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.maximum(0, x)

class AddHeads(Bond):
    """Reshapes an input to have heads.

    Input shape: (batch_size, sequence_length, embed_dim) 
    Output shape: (batch_size, num_heads, sequence_length, head_size)
    
    Adapted from Karpathy's nanoGPT.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
        self.num_heads = num_heads
    
    def forward(self, x, w):
        B, T, D = x.shape
        return jnp.reshape(x, (B, T, self.num_heads, D // self.num_heads)).transpose(0, 2, 1, 3)

class RemoveHeads(Bond):
    """Inverse of AddHeads."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        B, num_heads, T, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, num_heads * head_dim)

class AttentionQK(Bond):
    """Computes the query and key matrix multiplication in attention."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # what is this sensitivity?
    
    def forward(self, x, w):
        q, k = x
        scale = 1 / q.shape[-1]
        scores = q @ k.T * scale
        return scores

class CausalMask(Bond):
    """Masks the upper triangular part of the attention scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # what is this sensitivity?
    
    def forward(self, x, w):
        scores = x
        mask = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
        return jnp.where(mask, scores, -jnp.inf)

class Softmax(Bond):
    """Softmax with a sharpness parameter."""
    def __init__(self, scale):
        super().__init__()
        self.smooth = True
        self.sensitivity = scale / 2
        self.scale = scale
    
    def forward(self, x, w):
        return jax.nn.softmax(self.scale * x, axis=-1)

class AttentionOutput(Bond):
    """Computes attention values from the scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # what is this sensitivity?
    
    def forward(self, x, w):
        v, scores = x
        return scores @ v