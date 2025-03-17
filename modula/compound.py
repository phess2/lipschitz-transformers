from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width) @ ReLU()
    for _ in range(depth-2):
        m = m @ Linear(width, width) @ ReLU()
    return m @ Linear(width, input_dim)

def Attention(num_heads, d_embed, d_query, d_value, attention_scale, layer_idx=0):
    """Multi-head attention"""

    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed, tracker=f"v{layer_idx}")
    W = Linear(d_embed, num_heads * d_value, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax(attention_scale) @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, layer_idx=i)
        mlp = Linear(d_embed, 4*d_embed, tracker=f"mlp_out{i}") @ GeLU() @ Linear(4*d_embed, d_embed, tracker=f"mlp_in{i}")
        att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
        mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block

    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed, tracker="mlp_final")
    return out @ blocks @ embed

def OrthogonalAttention(num_heads, d_embed, softmax_scale, layer_idx=0):
    """
    Orthogonal attention uses 3-tensors for Q, K, V to make the input and output dimensions explicitly equal.
    """
    Q = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed, tracker=f"q{layer_idx}")
    K = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed, tracker=f"k{layer_idx}")
    V = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed, tracker=f"v{layer_idx}")
    W = HeadedLinearOut(num_heads, d_embed, d_embed, tracker=f"w{layer_idx}") @ TransposeHeads()

    AttentionScores = Softmax(softmax_scale) @ CausalMask() @ AttentionQK() @ Rope(d_embed) @ (Q, K)
    return ReduceHeads() @ ((1/3) * W) @ ApplyAttentionScores() @ (V, AttentionScores)

def OrthogonalGPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = OrthogonalAttention(num_heads, d_embed, attention_scale, layer_idx=i)
        mlp = Linear(d_embed, d_embed, tracker=f"mlp_out{i}") @ GeLU() @ Linear(d_embed, d_embed, tracker=f"mlp_in{i}")
        att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
        mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block
    
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed, tracker="mlp_final")

    return out @ blocks @ embed