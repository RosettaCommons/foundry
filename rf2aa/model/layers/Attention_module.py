import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opt_einsum import contract as einsum
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
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
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
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(MSAColAttention, self).__init__()
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
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(MSAColGlobalAttention, self).__init__()
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
    def __init__(self, d_pair, d_hidden=128, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        #
        self.gate = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_hidden)
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
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
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
        attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        attn = attn + bias # apply bias
        attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
        out = einsum('bijh,bnjhd->bnihd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        
        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)
        return out

