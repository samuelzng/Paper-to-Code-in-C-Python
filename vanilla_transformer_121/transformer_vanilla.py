import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_m = d_model

        # Look Up Table
        self.lut = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        # To scale the semantic representations ( PE is in range (-1, 1))
        return self.lut(x) * math.sqrt(self.d_m)
    
class PositionalEncoding(nn.Module):
    # Vanilla Version -> Rotary BE now
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE doesn't require gradients
        pe = torch.zeros(max_len, d_model) 

        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model) #<- Calculate once
        ) #[256]

        pe[:,0::2] = torch.sin(position * div_term) # [max_len, 256]
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe) # Move to GPU and save in state key
    
    def forward(self, x, ):
        # x -> [batch, len, d_model]
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class PointwiseFeedForward(nn.Module):
    """
    (max(xW1 + b1 , 0)W2 + b2)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # Dimension reduction on 2048
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> [batch, len, d_model]
        return self.w2(self.dropout(self.w1(x).relu()))

def clones(module, N):
    # To create N identical layers (with different addresses)
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def subsequent_mask(size):
    # Construct lower triangular matrix -> No attention to subsequent info
    attn_shape = (1, size, size)

    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    # Q, K, V -> [batch, heads, len, d_model//heads]
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    p_attn = scores.softmax(dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4) # Wq, Wk, Wv, Wo
        # However, initialize one big matrix (d_model, d_model * 3) can reduce the cost
        # self.c_atten = nn.Linear(d_model, 3 * d_model)
        # self.c_proj = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Q, K, V are still X -> Variable Shadowing
        # [Batch, Len, D_model]

        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask, self.dropout)

        x = (
            x.transpose(-1,-2)
            .contiguous()
            .view(nbatches, -1, self.d_k*self.h)
        )

        return x

class LayerNorm(nn.Module):
    # Vanilla LayerNorm -> Nowadays use RMSNorm
    # For each word vector, Gaussian Normalize
    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        return self.a * (x - mean) / (std + self.eps)  + self.b

class RMSNorm(nn.Module):
    # Remove the mean and bias
    def __init__(self, features, eps=1e-8):
        super().__init__()

        self.a = nn.Parameter(torch.ones(features))
        self.eps = eps
    
    def forward(self, x):

        # x / rqrt(x^2+eps)
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return norm_x * self.a
    
class SublayerConnection(nn.Module):
    # Modulize the Add-Norm Operation
    def __init__(self, size, dropout):
        super().__init__()
        
        self.norm = LayerNorm(size)
        # self.norm = RMSNorm(size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Here we introduce Pre-Norm (Vanilla is post Norm)
        # Resisual connection
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()

        self.attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout),2)
        self.size = size
    
    def forward(self, x, mask):
        # Mask prevents attention on <pad> in encoder_layer
        
        # First Multi Head Attention -> Sublayer Connection expects a funtion
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        # Then Feed-forward Networks -> FFN is callable already (No lambda)
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        # x by Pre-Norm is residual where not normalized
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        
        self.self_attn = self_attn
        # For cross attention, same module as self attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

        self.size = size
    
    def forward(self, x, memory, src_mask, tgt_mask):

        m = memory
        # src_mask -> Excluding <pad>; tgt_mask -> Prevent attending to subsequent
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()

        #Weight Typing -> Same as Embedding Vocab
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        # Only for the last word -> Cost Saving by slices
        return log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
   
    c = copy.deepcopy

    # Instantiation
    attn = MultiHeadAttention(h, d_model)
    ff = PointwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(src_vocab, d_model), c(position)), 
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(position)), 
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    # Weight Tying
    # Align: Generator <-> Source Embedding
    model.generator.proj.weight = model.src_embed[0].lut.weight
    
    
    # model.tgt_embed[0].lut.weight = model.src_embed[0].lut.weight

    return model

if __name__ == "__main__":
    # 1. Initialize a toy model for sanity check
    print(">>> Constructing the Transformer model...")
    model = make_model(src_vocab=100, tgt_vocab=100, N=2, d_model=512)
    model.eval() # Disable dropout for deterministic results

    # 2. Generate synthetic data (Batch=2, Src_Len=10, Tgt_Len=9)
    # Using random integers to simulate token IDs
    src = torch.randint(1, 100, (2, 10))
    
    # ------------------ CRITICAL FIX ------------------
    # The src_mask shape must be [Batch, 1, 1, Src_Len]
    # This ensures correct broadcasting across [Batch, Heads, Seq_Len, Src_Len]
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
    # --------------------------------------------------

    tgt = torch.randint(1, 100, (2, 9))
    # subsequent_mask returns [1, 9, 9], broadcasting aligns it with Batch dim
    tgt_mask = subsequent_mask(9).unsqueeze(0) 

    # 3. Run the forward pass
    print(">>> Running forward pass...")
    try:
        output = model(src, tgt, src_mask, tgt_mask)
        log_probs = model.generator(output)

        print(f"Input Shape (Src):      {src.shape}")
        print(f"Mask Shape (Src):       {src_mask.shape}  (Expected: [2, 1, 1, 10])")
        print(f"Input Shape (Tgt):      {tgt.shape}")
        print(f"Generator Output:       {log_probs.shape} (Expected: [2, 9, 100])")
        
        # Verification: [Batch, Tgt_Len, Vocab_Size]
        if log_probs.shape == (2, 9, 100):
            print(">>> ✅ Success! Transformer model built and dimensionalities match.")
        else:
            print(">>> ❌ Failure: Output dimension mismatch.")
            
    except RuntimeError as e:
        print(f">>> ❌ Runtime Error: {e}")
        print("Tip: Check your mask dimensions and broadcasting rules.")


