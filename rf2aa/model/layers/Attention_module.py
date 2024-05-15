import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opt_einsum import contract as einsum
from einops import rearrange
from rf2aa.util_module import init_lecun_normal

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model*r_ff, d_model)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # initialize linear layer right before residual connection: zero initialize
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, src):
        src = self.norm(src)
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src

# Attention model used in template embedding (if ptwise attn is enabled)
class Attention(nn.Module):
    # calculate multi-head attention
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out, p_drop=0.1):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        #
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        #
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        #
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]
        #
        query = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key = self.to_k(key).reshape(B, K, self.h, self.dim)
        value = self.to_v(value).reshape(B, K, self.h, self.dim)
        #
        query = query * self.scaling
        attn = einsum('bqhd,bkhd->bhqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bhqk,bkhd->bqhd', attn, value)
        out = out.reshape(B, Q, self.h*self.dim)
        #
        out = self.to_out(out)

        return out


# MSA Attention (row/column) from AlphaFold architecture
class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super(SequenceWeight, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_query = nn.Linear(d_msa, n_head*d_hidden)
        self.to_key = nn.Linear(d_msa, n_head*d_hidden)
        self.dropout = nn.Dropout(p_drop)
        self.reset_parameter()
    
    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
       
        tar_seq = msa[:,0]
        
        q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
        k = self.to_key(msa).view(B, N, L, self.h, self.dim)
        
        q = q * self.scale
        attn = einsum('bqihd,bkihd->bkihq', q, k)
        attn = F.softmax(attn, dim=1)
        return self.dropout(attn)

class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32, nseq_normalization=False, bias=True):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa, bias=bias)
        self.norm_pair = nn.LayerNorm(d_pair, bias=bias)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.nseq_normalization = nseq_normalization
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair): # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa) # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        if self.nseq_normalization:
            key = key * self.scaling * 1/math.sqrt(N)  #fd: from nate, change to match msa xformer paper
        else:
            key = key * self.scaling 

        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

class MSAColAttention(nn.Module):
    """
    Efficient implementation of MSA gated column attention using FlashAttention, when available.
    """
    def __init__(self, d_msa=256, n_head=8, d_hidden=32, bias=True):
        """
        args:
            d_msa: latent dimension of MSA embedding
            n_head: number of attention heads
            d_hidden: dimmension of each attention head
        """
        super(MSAColAttention, self).__init__()
        
        # Initilialize linear layers (Q, K, V) and layer normalization
        self.norm_msa = nn.LayerNorm(d_msa, bias=bias)
        self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
        
        # Gating
        self.to_g = nn.Linear(d_msa, n_head * d_hidden)
        
        # Output projection
        self.to_out = nn.Linear(n_head * d_hidden, d_msa)

        # Scaling
        self.scale_factor = 1 / math.sqrt(d_hidden)

        # Parameters
        self.n_head = n_head # number of heads
        self.d_head = d_hidden # the per-head dimension
        
        # Iniialize parameters
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize to ensure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        """
        Input:
            msa: (B, N, L, d_msa) MSA tensor. MSA tensors must be float16 or bfloat16 in order to use PyTorch's optimized SDPA function. Generally, this is handled by mixed precision training.
        Output:
            attn: (B, N, L, d_msa) Attention output tensor.
        """
        # Layer normalization (Note: automatically performed in float32 for mixed precision training given overflow issues with float16)
        msa = self.norm_msa(msa) # (B, N, L, d_msa)
        
        # (B, N, L, d_msa) --(projection)--> (B, N, L, h * d_head) --(rearrange)--> (BL, H, h, N, d_head)
        # We need to rearrange the tensor to match the input format of the PyTorch's optimized SDPA function, namely:
        # 1. The last two dimensions will undergo the attention operations (n, d_h)
        # 2. The tensors must be four-dimensional, so we need to flatten the batch and residue dimensions (b, l)
        query = rearrange(self.to_q(msa), 'b n l (h d_h) -> (b l) h n d_h', h=self.n_head)
        key = rearrange(self.to_k(msa), 'b n l (h d_h) -> (b l) h n d_h', h=self.n_head)
        value = rearrange(self.to_v(msa), 'b n l (h d_h) -> (b l) h n d_h', h=self.n_head)
        
        # Gating
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, d_msa)
        
        # SDPA, using PyTorch's scaled_dot_product_attention (defaults to FlashAttention with float16 or bfloat16)
        use_flash = torch.cuda.get_device_properties(0).major >= 8
        with torch.backends.cuda.sdp_kernel(enable_flash=use_flash, enable_math=True, enable_mem_efficient=False):
           attn = F.scaled_dot_product_attention(query, key, value, scale=self.scale_factor) # (BL, h, N, N) -> (BL, h, N, d_h)
        
        # Concatenate the heads and unsqueeze the batch (b) and residue (l) dimensions
        # (BL, h, N, d_h) -> (B, N, L, d_msa)
        attn = rearrange(attn, '(b l) h n d_h -> b n l (h d_h)', b=msa.shape[0])  
        
        # Apply gating
        attn = gate * attn
        
        # Output projection W_o
        return self.to_out(attn) # (B, N, L, d_msa)

class OldMSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(OldMSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * self.scaling
        attn = einsum('bqihd,bkihd->bihqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)

        return out

class MSAColGlobalAttention(nn.Module):
    """
    Efficient implementation of MSA gated global column attention using FlashAttention, when available.
    """
    def __init__(self, d_msa=64, n_head=8, d_hidden=8, bias=True):
        super(MSAColGlobalAttention, self).__init__()
        
        # Initilialize linear layers (Q, K, V) and layer normalization
        self.norm_msa = nn.LayerNorm(d_msa, bias=bias)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False) # Note that the key is not multi-headed
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False) # Note that the value is not multi-headed
        
        # Gating
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        
        # Output projection
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        # Scaling
        self.scale_factor = 1 / math.sqrt(d_hidden)
        
        # Parameters
        self.h = n_head
        self.d_head = d_hidden
        
        # Initialize parameters
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        """
        Input:
            msa: (B, N, L, d_msa) MSA tensor. MSA tensors must be float16 or bfloat16 in order to use PyTorch's optimized SDPA function. Generally, this is handled by mixed precision training.
        Output:
            attn: (B, N, L, d_msa) Attention output tensor.
        """
        # Layer normalization (Note: automatically performed in float32 for mixed precision training given overflow issues with float16)
        msa = self.norm_msa(msa) # (B, N, L, d_msa)
        
        # (B, N, L, d_msa) --(projection)--> (B, N, L, h * d_head) --(rearrange)--> (B, N, L, h, d_head)
        query = rearrange(self.to_q(msa), 'b n l (h d_h) -> b n l h d_h', h=self.h)
        query = query.mean(dim=1) # (B, L, h, d_head)
    
        # Key and value are not multi-headed   
        key = rearrange(self.to_k(msa), 'b n l d_h -> b l n d_h') # (B, L, N, d_h)
        value = rearrange(self.to_v(msa), 'b n l d_h -> b l n d_h') # (B, L, N, d_h)
        
        # Gating
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, d_msa)
        
        # SDPA, using PyTorch's scaled_dot_product_attention (defaults to FlashAttention with float16 or bfloat16)
        use_flash = torch.cuda.get_device_properties(0).major >= 8
        with torch.backends.cuda.sdp_kernel(enable_flash=use_flash, enable_math=True, enable_mem_efficient=False):
           attn = F.scaled_dot_product_attention(query, key, value, scale=self.scale_factor) # (B, L, h, N)

        # Concatenate on the head dimension, re-introduce a dimension to make multiplication work
        attn = rearrange(attn, 'b l h n -> b 1 l (h n)') # (B, 1, L, d_msa)

        # Apply gating
        out = gate * attn # (B, N, L, d_msa)
        
        # Output projection W_o
        return self.to_out(out) # (B, N, L, d_msa)

class OldMSAColGlobalAttention(nn.Module):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(OldMSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1) # (B, L, h, dim)
        key = self.to_k(msa) # (B, N, L, dim)
        value = self.to_v(msa) # (B, N, L, dim)
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, h*dim)
        #
        query = query * self.scaling
        attn = einsum('bihd,bkid->bihk', query, key) # (B, L, h, N)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1) # (B, 1, L, h*dim)
        out = gate * out # (B, N, L, h*dim)
        #
        out = self.to_out(out)
        return out

# TriangleAttention & TriangleMultiplication from AlphaFold architecture
class TriangleAttention(nn.Module):
    def __init__(self, d_pair, n_head=4, d_hidden=32, p_drop=0.1, start_node=True):
        super(TriangleAttention, self).__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)

        self.to_out = nn.Linear(n_head*d_hidden, d_pair)

        self.scaling = 1/math.sqrt(d_hidden)
        
        self.h = n_head
        self.dim = d_hidden
        self.start_node=start_node
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair):
        B, L = pair.shape[:2]

        pair = self.norm(pair)
        
        # input projection
        query = self.to_q(pair).reshape(B, L, L, self.h, -1)
        key = self.to_k(pair).reshape(B, L, L, self.h, -1)
        value = self.to_v(pair).reshape(B, L, L, self.h, -1)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim)
        
        # attention
        query = query * self.scaling
        if self.start_node:
            attn = einsum('bijhd,bikhd->bijkh', query, key)
        else:
            attn = einsum('bijhd,bkjhd->bijkh', query, key)
        attn = attn + bias.unsqueeze(1).expand(-1,L,-1,-1,-1) # (bijkh)
        attn = F.softmax(attn, dim=-2)
        if self.start_node:
            out = einsum('bijkh,bikhd->bijhd', attn, value).reshape(B, L, L, -1)
        else:
            out = einsum('bijkh,bkjhd->bijhd', attn, value).reshape(B, L, L, -1)
        out = gate * out # gated attention
        
        # output projection
        out = self.to_out(out)
        return out

class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hidden=128, outgoing=True, bias=True):
        super(TriangleMultiplication, self).__init__()
        self.norm = nn.LayerNorm(d_pair, bias=bias)
        self.left_proj = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        #
        self.gate = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_hidden, bias=bias)
        self.out_proj = nn.Linear(d_hidden, d_pair)

        self.outgoing = outgoing
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal distribution for regular linear weights
        self.left_proj = init_lecun_normal(self.left_proj)
        self.right_proj = init_lecun_normal(self.right_proj)

        # Set Bias of Linear layers to zeros
        nn.init.zeros_(self.left_proj.bias)
        nn.init.zeros_(self.right_proj.bias)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.left_gate.weight)
        nn.init.ones_(self.left_gate.bias)
        
        nn.init.zeros_(self.right_gate.weight)
        nn.init.ones_(self.right_gate.bias)
        
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, pair):
        B, L = pair.shape[:2]
        pair = self.norm(pair)

        left = self.left_proj(pair) # (B, L, L, d_h)
        left_gate = torch.sigmoid(self.left_gate(pair))
        left = left_gate * left
        
        right = self.right_proj(pair) # (B, L, L, d_h)
        right_gate = torch.sigmoid(self.right_gate(pair))
        right = right_gate * right
        
        if self.outgoing:
            out = einsum('bikd,bjkd->bijd', left, right/float(L))
        else:
            out = einsum('bkid,bkjd->bijd', left, right/float(L))
        out = self.norm_out(out)
        out = self.out_proj(out)

        gate = torch.sigmoid(self.gate(pair)) # (B, L, L, d_pair)
        out = gate * out
        return out

# Instead of triangle attention, use Tied axail attention with bias from coordinates..?
class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True, bias=True):
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair, bias=bias)
        self.norm_bias = nn.LayerNorm(d_bias, bias=bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias):
        # pair: (B, L, L, d_pair)
        B, L = pair.shape[:2]
        
        if self.is_row:
            pair = pair.permute(0,2,1,3)
            bias = bias.permute(0,2,1,3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
        query = query * self.scaling
        key = key / L # normalize for tied attention
        attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        attn = attn + bias # apply bias
        attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
        out = einsum('bijh,bnjhd->bnihd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        
        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)
        return out

class BiasedUntiedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, is_row=True):
        super(BiasedUntiedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias):
        # pair: (B, L, L, d_pair)
        B, L = pair.shape[:2]
        
        if self.is_row:
            pair = pair.permute(0,2,1,3)
            bias = bias.permute(0,2,1,3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)

        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
        query = query * self.scaling
        key = key / L # normalize for tied attention
        attn = einsum('bnihk,bnjhk->bnijh', query, key) # tied attention
        attn = attn + bias # apply bias
        attn = F.softmax(attn, dim=-2) # (B, L, L, L, h)
        
        out = einsum('bnijh,bnjhd->bnihd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        
        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)
        return out

