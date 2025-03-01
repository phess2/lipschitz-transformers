from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width) @ ReLU()
    for _ in range(depth-2):
        m = m @ Linear(width, width) @ ReLU()
    return m @ Linear(width, input_dim)

def Attention(num_heads, d_embed, d_query, d_value, softmax_scale, causal):
    """Multi-head attention"""
    Q = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = AddHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, num_heads * d_value) @ RemoveHeads()

    AttentionScores = Softmax(softmax_scale) @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ 1/3 * AttentionOutput() @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, softmax_scale=1.0):
    # still needs to be implemented:
    #   - RoPE
    #   - putting the blocks together
    pass