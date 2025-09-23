import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer

    FiLM applies feature-wise affine transformation: h = γ * h + β
    where γ (scale) and β (shift) are learned from conditioning input
    """
    def __init__(self, conditioning_dim: int, feature_dim: int):
        super().__init__()
        # Learn both scale (gamma) and shift (beta) parameters
        self.film_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, feature_dim * 2)  # *2 for gamma and beta
        )

    def forward(self, features, conditioning):
        """
        Args:
            features: [batch, feature_dim, height, width] - feature maps to modulate
            conditioning: [batch, conditioning_dim] - conditioning vector (time+label embeddings)
        Returns:
            modulated features: [batch, feature_dim, height, width]
        """
        # Generate scale and shift parameters
        film_params = self.film_mlp(conditioning)  # [batch, feature_dim * 2]

        # Split into gamma (scale) and beta (shift)
        gamma, beta = film_params.chunk(2, dim=1)  # Each: [batch, feature_dim]

        # Reshape for broadcasting: [batch, feature_dim, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM: feature-wise linear modulation
        return gamma * features + beta

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()

        # Precompute fixed sinusoidal embeddings and register as buffer
        pos = torch.arange(0, time_steps, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(pos * div)
        embeddings[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("embeddings", embeddings, persistent=False)

        # Backward-compat alias (in case any external code references old name)
        self.embedings = self.embeddings

    def forward(self, t):
        # Accept int, list, tuple, or tensor; ensure correct dtype/device for indexing
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.long, device=self.embeddings.device)
        else:
            t = t.to(self.embeddings.device).long()
        embeds = self.embeddings[t]
        return embeds  # [batch, embed_dim]



class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_groups: int, dropout_prob: float, use_film: bool = False):
        super().__init__()
        self.relu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.use_film = use_film

        if use_film:
            # FiLM layer for conditioning injection
            self.film_layer = FiLMLayer(time_emb_dim, out_channels)
        else:
            # Simple time embedding injection
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )

    def forward(self, x, time_emb):
        # First conv + norm
        h = self.conv1(self.relu(self.norm1(x)))

        if self.use_film:
            # Apply FiLM conditioning (feature-wise linear modulation)
            h = self.film_layer(h, time_emb)
        else:
            # Inject time embedding
            time_emb_proj = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb_proj

        # Dropout + second conv
        h = self.dropout(h)
        h = self.conv2(self.relu(self.norm2(h)))

        return self.skip(x) + h
    
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float, use_flash_attention: bool = True):
        super().__init__()
        self.linear_qkv = nn.Linear(C, C * 3, bias=False)
        self.linear_out = nn.Linear(C, C, bias=False)
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
        if q.is_cuda:
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=self.use_flash_attention, enable_mem_efficient=not self.use_flash_attention, enable_math=not self.use_flash_attention):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
            except Exception:
                pass
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W, C)
        qkv = self.linear_qkv(x).view(batch_size, H * W, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)  # (batch, num_heads, h*w, C//num_heads)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        x = self._sdpa(q, k, v, dropout_p=self.dropout_prob if self.training else 0.0)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, H * W, C)
        x = self.linear_out(x)
        return x.view(batch_size, H, W, C).permute(0, 3, 1, 2).contiguous()

class CrossAttention(nn.Module):
    """Cross-attention where queries come from image features and keys/values from context.

    Args:
        query_dim: channel dimension of the image features (C of [B,C,H,W] or last dim of [B,N,C])
        context_dim: channel dimension of the context features ([B,L,context_dim] or [B,context_dim])
        num_heads: number of attention heads (query_dim must be divisible by num_heads)
        dropout_prob: dropout probability applied inside attention
    """
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout_prob: float = 0.0, use_flash_attention: bool = True):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.use_flash_attention = use_flash_attention

        # Linear projections for Q (from image), K and V (from context)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim, bias=False)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
        if q.is_cuda:
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=self.use_flash_attention, enable_mem_efficient=not self.use_flash_attention, enable_math=not self.use_flash_attention):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
            except Exception:
                pass
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image features to tokens [B, N, C] from [B, C, H, W] or pass through if already [B, N, C]."""
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return x

    def _from_tokens(self, x_tokens: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
        """Restore tokens [B, N, C] back to [B, C, H, W] given spatial shape (H, W)."""
        if spatial_shape is None:
            return x_tokens
        b, n, c = x_tokens.shape
        h, w = spatial_shape
        return x_tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Track spatial shape if image features are 4D
        spatial_shape = None
        if x.dim() == 4:
            spatial_shape = (x.shape[2], x.shape[3])
        x_tokens = self._to_tokens(x)  # [B, N, Cq]

        # Ensure context is [B, L, Dc]
        if context.dim() == 2:
            context = context.unsqueeze(1)

        # Project to Q, K, V
        q = self.to_q(x_tokens)           # [B, N, Cq]
        k = self.to_k(context)            # [B, L, Cq]
        v = self.to_v(context)            # [B, L, Cq]

        # Reshape for multi-head attention
        b, n, c = q.shape
        _, l, _ = k.shape
        head_dim = c // self.num_heads
        q = q.view(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        k = k.view(b, l, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, L, Hd]
        v = v.view(b, l, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, L, Hd]

        # Cross attention
        attn_out = self._sdpa(q, k, v, dropout_p=self.dropout_prob if self.training else 0.0)  # [B, H, N, Hd]

        # Merge heads and project out
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(b, n, c)  # [B, N, C]
        out = self.to_out(attn_out)  # [B, N, C]

        return self._from_tokens(out, spatial_shape)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UnetLayer(nn.Module):
    def __init__(self,
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout_prob: float,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 num_heads: int = 4,
                 use_film: bool = False,
                 use_cross_attention: bool = False,
                 context_dim: int = None,
                 use_flash_attention: bool = True):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, time_emb_dim, num_groups, dropout_prob, use_film)
        if attention:
            self.attention = Attention(out_channels, num_heads, dropout_prob, use_flash_attention=use_flash_attention)
        if use_cross_attention and context_dim is not None:
            self.cross_attention = CrossAttention(query_dim=out_channels, context_dim=context_dim, num_heads=num_heads, dropout_prob=dropout_prob, use_flash_attention=use_flash_attention)
        if upscale:
            self.upsample = Upsample(out_channels)
            self.conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=1)  # 1x1 conv to halve channels
        else:
            self.downsample = Downsample(out_channels)

    def forward(self, x, time_emb, context=None):

        x = self.res_block(x, time_emb)
        res_x = x
        if hasattr(self, 'attention'):
            x = self.attention(x)
        if hasattr(self, 'cross_attention') and context is not None:
            x = self.cross_attention(x, context)

        x = x + res_x

        if hasattr(self, 'upsample'):
            # Upsampling path
            x = self.upsample(x)  # 2x spatial, same channels
            x = self.conv(x)      # Halve channels
            return x, x
        else:
            # Downsampling path
            residual = x
            x = self.downsample(x)  # Halve spatial, same channels
            return x, residual

class UNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 64, 128, 256, 128],
            Upscales: List = [False, False, True, True],
            Attentions: List = [False, True, True, False],
            num_groups: int = 8,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            time_steps: int = 1000,
            input_channels: int = 1,
            output_channels: int = 1,
            label_embedding: bool = True,
            use_film: bool = False,
            use_cross_attention: bool = True,
            context_dim: int = None,
            use_flash_attention: bool = True):
        super().__init__()
        # Time embedding dimension (simpler for MNIST)
        time_emb_dim = 256

        self.num_layers = len(Channels) - 1
        for i in range(self.num_layers):
            layer = UnetLayer(Upscales[i], Attentions[i], num_groups, dropout_prob,
                            Channels[i], Channels[i+1], time_emb_dim, num_heads, use_film,
                            use_cross_attention=use_cross_attention, context_dim=context_dim,
                            use_flash_attention=use_flash_attention)
            setattr(self, f'Layer{i+1}', layer)

        # Calculate final output channels for concatenation
        out_channels = (Channels[-1]//2) + Channels[1]
        self.input_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        self.late_conv1 = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.late_conv2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.SiLU()

        # Zero-init final conv for stable early training
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

        # Time and label embeddings (simplified)
        self.time_embeddings = SinusoidalEmbeddings(time_steps, time_emb_dim)

        # Time embedding MLP (simplified)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
    def forward(self, x, t, context = None):
        # Get time embedding and process through MLP
        time_emb = self.time_embeddings(t)  # [batch, time_emb_dim]
        time_emb = self.time_mlp(time_emb)   # Process through MLP
        
        resiudal = []

        x = self.input_conv(x)

        # ENCODER - each layer gets the combined embedding
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, time_emb, context)  # Pass combined embedding and context
            resiudal.append(r)
        
        # DECODER - each layer gets the combined embedding
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, time_emb, context)[0], resiudal.pop()), dim=1)
        return self.output_conv(self.relu(self.late_conv2(self.relu(self.late_conv1(x)))))   

