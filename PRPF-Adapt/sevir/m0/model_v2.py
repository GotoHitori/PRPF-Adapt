import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

                           
                             
                           

def warp(img, flow):
    B, C, H, W = img.size()
          
    xx = torch.arange(0, W, device=img.device)
    yy = torch.arange(0, H, device=img.device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((xx, yy), dim=0).float()             
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)                
    vgrid = grid + flow
    
                                     
    vgrid_x = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = torch.stack((vgrid_x, vgrid_y), dim=-1)                
    
    return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

def sn(module: nn.Module):
    return nn.utils.spectral_norm(module)
def _gn_groups(ch):
    g = min(32, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return g
def _inverse_softplus(x):
    return torch.log(torch.expm1(x))

def pad_tensor(x: torch.Tensor, multiple: int):
    if x.dim() == 4:
        B, C, H, W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x = F.pad(x, (l, r, t, b), mode='reflect')
        return x, (t, b, l, r)
    elif x.dim() == 5:
        B, T, C, H, W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x_flat = x.view(B * T, C, H, W)
            x_flat = F.pad(x_flat, (l, r, t, b), mode='reflect')
            x = x_flat.view(B, T, C, H2, W2)
        return x, (t, b, l, r)
    else:
        raise ValueError("pad_tensor expects 4D or 5D tensor")

def pad_tensor_back(x: torch.Tensor, pads):
    t, b, l, r = pads
    if x.dim() in (4, 5):
        if t or b or l or r:
            x = x[..., t:x.shape[-2] - b, l:x.shape[-1] - r]
        return x
    else:
        return x

def build_2d_sincos_pos_embed(ch: int, H: int, W: int, device=None) -> torch.Tensor:
    assert ch % 4 == 0, "pos_embed channels must be divisible by 4"
    device = device or 'cpu'
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(ch // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (ch // 4)))
    siny = torch.sin(yy[..., None] * omega)
    cosy = torch.cos(yy[..., None] * omega)
    sinx = torch.sin(xx[..., None] * omega)
    cosx = torch.cos(xx[..., None] * omega)
    pos = torch.cat([siny, cosy, sinx, cosx], dim=-1)
    pos = pos.permute(2, 0, 1).unsqueeze(0)
    return pos

                           
                       
                           

class FrameDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        ch = 32
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),             
            sn(nn.Conv2d(ch, ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            sn(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            sn(nn.Conv2d(ch * 4, ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = sn(nn.Conv2d(ch * 8, 1, 4, 1, 0))

    def forward(self, x):
        h = self.net(x)
        return self.head(h).flatten(1).mean(dim=1, keepdim=True)

class SeqDiscriminator3D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        ch = 16
        self.layers = nn.Sequential(
            sn(nn.Conv3d(in_ch, ch, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.1),             
            sn(nn.Conv3d(ch, ch * 2, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.1),
            sn(nn.Conv3d(ch * 2, ch * 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            nn.Dropout3d(0.1),
            sn(nn.Conv3d(ch * 4, ch * 8, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
        )
        self.head = sn(nn.Conv3d(ch * 8, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.layers(x)
        return self.head(h).flatten(1).mean(dim=1, keepdim=True)

                           
        
                           
class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class SimVPConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(BasicConv2d(ch, ch), BasicConv2d(ch, ch))
    def forward(self, x): return self.net(x) + x

class GNConvBlock(nn.Module):
    """Generator conv block: Conv2d + GroupNorm + SiLU.
    Spectral norm is reserved for the discriminators; the generator uses
    GroupNorm (batch-size independent, stable for small-batch spatiotemporal
    training) instead, which is the standard choice for generative backbones."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class RadarEncoder(nn.Module):
    """Radar branch (paper Sec. III.B): three Conv2d + GroupNorm + SiLU blocks;
    the first block is strided to downsample 128 -> 64."""
    def __init__(self, T, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            GNConvBlock(T * in_ch, out_ch, k=3, s=2, p=1),
            GNConvBlock(out_ch, out_ch, k=3, s=1, p=1),
            GNConvBlock(out_ch, out_ch, k=3, s=1, p=1),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        return self.net(x)
class PanguTimeAlignEncoder(nn.Module):
    """Pangu branch (paper Sec. III.B / Fig. 2(a)).
    Time-align 3D = per-frame shallow conv -> temporal interpolation (Tw -> Tout)
    -> channel-align conv; then three Conv2d + GroupNorm + SiLU blocks.
    Channel count (Cw) and spatial size are preserved so the downstream fusion
    projections (to_att / pangu_mapper) remain valid."""
    def __init__(self, Cw, Tout):
        super().__init__()
        self.Tout = Tout
        self.shallow = GNConvBlock(Cw, Cw, k=3, s=1, p=1)
        self.channel_align = GNConvBlock(Cw, Cw, k=3, s=1, p=1)
        self.blocks = nn.Sequential(
            GNConvBlock(Cw, Cw, k=3, s=1, p=1),
            GNConvBlock(Cw, Cw, k=3, s=1, p=1),
            GNConvBlock(Cw, Cw, k=3, s=1, p=1),
        )
    def forward(self, x):
        B, Tw, C, H, W = x.shape
        y = self.shallow(x.reshape(B * Tw, C, H, W)).reshape(B, Tw, C, H, W)
        y = y.permute(0, 2, 1, 3, 4)
        y = F.interpolate(y, size=(self.Tout, H, W), mode='trilinear', align_corners=False)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        y = self.channel_align(y.reshape(B * self.Tout, C, H, W))
        y = self.blocks(y)
        return y.reshape(B, self.Tout, C, H, W)

class DualHeadDecoder(nn.Module):
    """Shared decoder (paper Sec. III.D / Fig. 2(a)): a ConvTranspose2d upsamples
    64 -> 128, followed by three Conv2d + GroupNorm + SiLU blocks; a radar head and
    a flow head then branch from the shared representation."""
    def __init__(self, in_ch, mid_ch=128):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.Sequential(
            GNConvBlock(mid_ch, mid_ch, k=3, s=1, p=1),
            GNConvBlock(mid_ch, mid_ch, k=3, s=1, p=1),
            GNConvBlock(mid_ch, mid_ch, k=3, s=1, p=1),
        )
        self.head_radar = nn.Conv2d(mid_ch, 1, 3, 1, 1)
        self.head_flow = nn.Conv2d(mid_ch, 2, 3, 1, 1)
    def forward(self, x):
        x = self.up(x)
        x = self.blocks(x)
        return self.head_radar(x), self.head_flow(x) * 0.1

class LTAMBias(nn.Module):
    def __init__(self, num_heads: int, init_min: float = 0.01, init_max: float = 1.0, eps: float = 1e-6):
        super().__init__()
        assert num_heads >= 1
        assert init_min > 0.0 and init_max > 0.0
        self.num_heads = int(num_heads)
        self.eps = float(eps)

        if self.num_heads == 1:
            init_lam = torch.tensor([math.sqrt(init_min * init_max)], dtype=torch.float32)
        else:
            init_lam = torch.logspace(
                math.log10(init_min), math.log10(init_max), steps=self.num_heads, dtype=torch.float32
            )
        theta0 = _inverse_softplus(init_lam)
        self.theta = nn.Parameter(theta0)

    def lambdas(self) -> torch.Tensor:
        return F.softplus(self.theta) + self.eps

    def build_attn_mask(
        self,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> torch.Tensor:
        q_pos = q_pos.to(device=device)
        k_pos = k_pos.to(device=device)

        dist = (q_pos[:, None] - k_pos[None, :]).abs().to(dtype=dtype)
        lam = self.lambdas().to(device=device, dtype=dtype).view(self.num_heads, 1, 1)
        bias = -lam * dist

        if causal:
            future = (k_pos[None, :] > q_pos[:, None])
            neg_inf = torch.finfo(dtype).min
            bias = bias.masked_fill(future.unsqueeze(0), neg_inf)

        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.num_heads, bias.shape[1], bias.shape[2])
        return bias

def _pick_num_heads(embed_dim: int, max_heads: int) -> int:
    max_heads = max(1, int(max_heads))
    embed_dim = int(embed_dim)
    for h in range(min(max_heads, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1

class RadarLTAMModulator(nn.Module):
    def __init__(self, T: int, Cin: int, embed_dim: int = 64, heads: int = 4,
                 init_min: float = 0.01, init_max: float = 1.0, init_scale: float = 0.1):
        super().__init__()
        self.T = int(T)
        self.Cin = int(Cin)
        self.embed_dim = int(embed_dim)
        self.heads = _pick_num_heads(self.embed_dim, heads)

        self.in_proj = nn.Linear(self.Cin, self.embed_dim)
        self.mha = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.ltam = LTAMBias(num_heads=self.heads, init_min=init_min, init_max=init_max)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x):
        B, T, C, H, W = x.shape
        if T != self.T:
            self.T = int(T)

        tok = x.mean(dim=(-2, -1))
        tok = self.in_proj(tok)
        tok = self.norm(tok)

        q = tok[:, -1:, :]
        k = tok
        v = tok

        q_pos = torch.tensor([T - 1], device=x.device, dtype=torch.long)
        k_pos = torch.arange(T, device=x.device, dtype=torch.long)

        attn_mask = self.ltam.build_attn_mask(
            q_pos=q_pos,
            k_pos=k_pos,
            batch_size=B,
            dtype=tok.dtype,
            device=x.device,
            causal=True,
        )

        _, attn_w = self.mha(q, k, v, attn_mask=attn_mask, need_weights=True)
        w = attn_w.squeeze(1)
        w = w * float(T)

        x_weighted = x * w.view(B, T, 1, 1, 1)
        return x + torch.tanh(self.scale) * (x_weighted - x)

class MLP(nn.Module):
    def __init__(self, dim, hidden_ratio=4.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class CrossAttentionAlign(nn.Module):
    def __init__(self, Cw: int, Cd: int = 64, nheads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.Cw = Cw
        self.Cd = Cd
        self.q_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.k_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.v_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.norm1 = nn.LayerNorm(Cd)
        self.mha = nn.MultiheadAttention(embed_dim=Cd, num_heads=nheads, batch_first=True)
        self.norm2 = nn.LayerNorm(Cd)
        self.mlp = MLP(Cd, hidden_ratio=mlp_ratio)
        self.out_proj = nn.Linear(Cd, Cw)

    def forward(self, cond: torch.Tensor, wfm_d: torch.Tensor):
        B, Cd, He, We = cond.shape
        Hp, Wp = wfm_d.shape[-2:]
        device = cond.device

        q_feat = self.q_proj(cond) + build_2d_sincos_pos_embed(Cd, He, We, device=device)
        k_feat = self.k_proj(wfm_d) + build_2d_sincos_pos_embed(Cd, Hp, Wp, device=device)
        v_feat = self.v_proj(wfm_d)

        q = q_feat.flatten(2).transpose(1, 2)
        k = k_feat.flatten(2).transpose(1, 2)
        v = v_feat.flatten(2).transpose(1, 2)

        shortcut = q
        q_norm = self.norm1(q)
        attn_out, _ = self.mha(q_norm, k, v)
        q = shortcut + attn_out

        shortcut = q
        q = self.norm2(q)
        q = shortcut + self.mlp(q)

        out = self.out_proj(q)
        out = out.transpose(1, 2).reshape(B, self.Cw, He, We)
        return out

class ChannelCrossAttentionGlobal(nn.Module):
    def __init__(self, Ca: int, Cb: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.Ca, self.Cb = Ca, Cb
        self.emb_a = nn.Parameter(torch.randn(Ca, d) * 0.02)
        self.emb_b = nn.Parameter(torch.randn(Cb, d) * 0.02)
        self.val2emb_a = nn.Linear(1, d)
        self.val2emb_b = nn.Linear(1, d)

        self.norm_a = nn.LayerNorm(d)
        self.norm_b = nn.LayerNorm(d)

        self.mha_ab = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj_gate_a = nn.Linear(d, 1)
        self.proj_gate_b = nn.Linear(d, 1)

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        a_val = A.mean(dim=(-2, -1), keepdim=False)
        b_val = B.mean(dim=(-2, -1), keepdim=False)
        tok_a = self.emb_a.unsqueeze(0) + self.val2emb_a(a_val.unsqueeze(-1))
        tok_b = self.emb_b.unsqueeze(0) + self.val2emb_b(b_val.unsqueeze(-1))

        tok_a = self.norm_a(tok_a)
        tok_b = self.norm_b(tok_b)

        out_ab, _ = self.mha_ab(tok_a, tok_b, tok_b)
        gate_a = torch.sigmoid(self.proj_gate_a(out_ab)).squeeze(-1)
        A2 = A * (1.0 + gate_a.view(-1, self.Ca, 1, 1))

        out_ba, _ = self.mha_ba(tok_b, tok_a, tok_a)
        gate_b = torch.sigmoid(self.proj_gate_b(out_ba)).squeeze(-1)
        B2 = B * (1.0 + gate_b.view(-1, self.Cb, 1, 1))
        return A2, B2

class ChannelSelfAttentionGlobal(nn.Module):
    def __init__(self, C: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.C = C
        self.emb = nn.Parameter(torch.randn(C, d) * 0.02)
        self.val2emb = nn.Linear(1, d)

        self.norm = nn.LayerNorm(d)

        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj_gate = nn.Linear(d, 1)

    def forward(self, X: torch.Tensor):
        x_val = X.mean(dim=(-2, -1), keepdim=False)
        tok = self.emb.unsqueeze(0) + self.val2emb(x_val.unsqueeze(-1))

        tok = self.norm(tok)

        out, _ = self.mha(tok, tok, tok)
        gate = torch.sigmoid(self.proj_gate(out)).squeeze(-1)
        return X * (1.0 + gate.view(-1, self.C, 1, 1))

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_ratio=mlp_ratio)

    def calculate_mask(self, H, W, device):
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask
        else:
            return None

    def forward(self, x, attn_mask=None):
        H, W = x.shape[1], x.shape[2]
        B, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if attn_mask is not None:
            attn_mask_tiled = attn_mask.repeat(B, 1, 1)
            attn_mask_tiled = attn_mask_tiled.repeat_interleave(self.num_heads, dim=0)
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows, attn_mask=attn_mask_tiled)
        else:
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class SwinBasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        dummy_block = self.blocks[1] if self.depth > 1 else None
        attn_mask = None
        if dummy_block is not None and dummy_block.shift_size > 0:
            attn_mask = dummy_block.calculate_mask(H, W, x.device)

        for blk in self.blocks:
            if blk.shift_size > 0:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x, attn_mask=None)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class TemporalCrossAttention(nn.Module):
    def __init__(
        self,
        C: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        ltam_init_min: float = 0.01,
        ltam_init_max: float = 1.0,
        causal: bool = True,
    ):
        super().__init__()
        self.C = C
        self.heads = heads
        self.causal = bool(causal)

        self.mha = nn.MultiheadAttention(C, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(C)
        self.ltam = LTAMBias(num_heads=heads, init_min=ltam_init_min, init_max=ltam_init_max)
        self.norm2 = nn.LayerNorm(C)
        self.mlp = MLP(C, hidden_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x_t = x.permute(0, 2, 1, 3, 4)

        tokens = x_t.mean(dim=(-2, -1))

        pos = torch.arange(T, device=x.device, dtype=torch.long)
        attn_mask = self.ltam.build_attn_mask(
            q_pos=pos,
            k_pos=pos,
            batch_size=B,
            dtype=tokens.dtype,
            device=x.device,
            causal=self.causal,
        )

        residual = tokens
        tokens_norm = self.norm1(tokens)
        y, _ = self.mha(tokens_norm, tokens_norm, tokens_norm, attn_mask=attn_mask, need_weights=False)
        tokens = residual + y

        residual = tokens
        tokens_norm = self.norm2(tokens)
        tokens = residual + self.mlp(tokens_norm)

        tokens = tokens.unsqueeze(-1).unsqueeze(-1)
        x = x + tokens.permute(0, 2, 1, 3, 4)
        return x

class CS_MMSA_Block(nn.Module):
    def __init__(self, radar_dim: int, pangu_dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.radar_dim = radar_dim
        self.pangu_dim = pangu_dim
        self.pangu_mapper = nn.Conv2d(pangu_dim, radar_dim, kernel_size=1, bias=False)

        self.r_norm = nn.GroupNorm(_gn_groups(radar_dim), radar_dim, affine=False)
        self.r_dw = nn.Conv2d(radar_dim, radar_dim, kernel_size=3, padding=1, groups=radar_dim, bias=False)
        self.r_pw = nn.Conv2d(radar_dim, radar_dim, kernel_size=1, bias=False)

        self.p_norm = nn.GroupNorm(_gn_groups(radar_dim), radar_dim, affine=False)
        self.p_dw = nn.Conv2d(radar_dim, radar_dim, kernel_size=3, padding=1, groups=radar_dim, bias=False)
        self.p_pw = nn.Conv2d(radar_dim, radar_dim, kernel_size=1, bias=False)

        self.gate = nn.Sequential(
            nn.Conv2d(radar_dim * 2, radar_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.refine_norm = nn.GroupNorm(_gn_groups(radar_dim), radar_dim)
        self.refine_dw = nn.Conv2d(radar_dim, radar_dim, kernel_size=3, padding=1, groups=radar_dim, bias=False)
        self.refine_pw = nn.Conv2d(radar_dim, radar_dim, kernel_size=1, bias=False)

        self.inject_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, radar_feat: torch.Tensor, pangu_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = radar_feat.shape
        p_up = F.interpolate(pangu_feat, size=(H, W), mode='bilinear', align_corners=False)
        p_m = self.pangu_mapper(p_up)

        r0 = self.r_norm(radar_feat)
        r_tex = F.silu(self.r_pw(self.r_dw(r0)))
        r_res = r_tex - r0

        p0 = self.p_norm(p_m)
        p_loc = F.silu(self.p_pw(self.p_dw(p0)))

        g = self.gate(torch.cat([r_res, p_loc], dim=1))
        x = radar_feat + torch.tanh(self.inject_scale) * (g * r_res)

        x_ref = F.silu(self.refine_pw(self.refine_dw(self.refine_norm(x))))
        return x + 0.10 * x_ref

                                                                               
                                                                      
                                                                  
                                                                              
                                      
                                                                              
                                                                               
                                            
                                                                                 
                                                                               
                                                                                 
                                                                              
                                       
                                                                         
                                                                                
                                                                                  
                                                                              
                                                                      
                                                                                  
                                                                              
                                                                        
                                                                                
 
                                                                  
                                                     
                                                                             
                                              
                                                                       
                                                                               
PANGU_PHYS_GROUPS = {
    "wind":    list(range(0, 12)) + [31, 32],
    "thermal": list(range(12, 24)) + [30],
    "mass":    list(range(24, 30)) + [33],
}


class ATrousLaplacianPyramid(nn.Module):
    """Undecimated (a-trous) Laplacian pyramid on a feature map.
    Depthwise, separable binomial low-pass h=[1,4,6,4,1]/16 (a Gaussian approx;
    h = ([1,1]/2) self-convolved 4x) with dilation 2^j at level j. No down-
    sampling, so the sub-bands satisfy sum_j B_j == input EXACTLY (telescoping).
    Returns [B_0, ..., B_J]: B_0 finest band-pass ... B_{J-1} band-pass,
    B_J = low-pass residual (coarsest)."""
    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.J = num_levels
        k = torch.tensor([1., 4., 6., 4., 1.])
        k = k / k.sum()
        self.register_buffer("kh", k.view(1, 1, 1, 5), persistent=False)                  
        self.register_buffer("kv", k.view(1, 1, 5, 1), persistent=False)                

    def _blur(self, x, dilation: int):
        B, C, H, W = x.shape
        pad = 2 * dilation                            
        kh = self.kh.expand(C, 1, 1, 5)
        kv = self.kv.expand(C, 1, 5, 1)
        x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
        x = F.conv2d(x, kh, groups=C, dilation=(1, dilation))
        x = F.pad(x, (0, 0, pad, pad), mode="reflect")
        x = F.conv2d(x, kv, groups=C, dilation=(dilation, 1))
        return x

    def forward(self, x):
        bands = []
        cur = x
        for j in range(self.J):
            low = self._blur(cur, dilation=2 ** j)
            bands.append(cur - low)                  
            cur = low
        bands.append(cur)                                    
        return bands                                


class PASMFusion(nn.Module):
    """Predictability-graded Spectral-Authority Cession fusion (research content II).

    Interface mirrors a fusion block but is split into `analyze` (lead-time-
    invariant: pyramid decomposition + spatial group routing, computed once) and
    `synthesize` (per lead time: FiLM coeffs + authorization + reconstruction).
    `forward` runs both for standalone use.

    Args
    ----
    radar_dim   : radar feature width C (output width = C, unchanged downstream).
    pangu_ch    : number of RAW pangu channels (34).
    num_levels  : pyramid levels J (J+1 sub-bands).
    Tout        : number of lead times (for the time embedding / breathing).
    use_breathe : k_c varies with lead time (else frozen)         [ablation].
    use_protect : finest band structurally protected (a_0 := 0)   [ablation].
    use_group   : physics-grouped FiLM (else a single FiLM)       [ablation].
    use_decay   : long-lead finest-band decay d_0(t)              [ablation].
    """
    def __init__(self, radar_dim: int, pangu_ch: int = 34, num_levels: int = 3,
                 Tout: int = 20, time_embed_dim: int = 32, hidden: int = 128,
                 use_breathe: bool = True, use_protect: bool = True,
                 use_group: bool = True, use_decay: bool = True,
                 phys_groups=None):
        super().__init__()
        self.C = radar_dim
        self.J = num_levels
        self.n_bands = num_levels + 1
        self.Tout = Tout
        self.use_breathe = use_breathe
        self.use_protect = use_protect
        self.use_group = use_group
        self.use_decay = use_decay

        phys_groups = phys_groups or PANGU_PHYS_GROUPS
        if use_group:
            self.group_idx = [phys_groups["wind"], phys_groups["thermal"], phys_groups["mass"]]
        else:
            self.group_idx = [list(range(pangu_ch))]                           
        self.G = len(self.group_idx)
        for g, idx in enumerate(self.group_idx):
            self.register_buffer(f"gidx_{g}", torch.tensor(idx, dtype=torch.long), persistent=False)

        self.pyramid = ATrousLaplacianPyramid(num_levels)
        self.time_embed = nn.Embedding(Tout, time_embed_dim)

                                                                          
        self.film = nn.ModuleList()
        for idx in self.group_idx:
            mlp = nn.Sequential(
                nn.Linear(len(idx) + time_embed_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, self.n_bands * self.C * 2),
            )
            nn.init.zeros_(mlp[-1].weight)                                   
            nn.init.zeros_(mlp[-1].bias)
            self.film.append(mlp)

                                                                         
        self.route = nn.Conv2d(self.C, self.G, kernel_size=1)

                                                                                 
        self.k_max = float(num_levels)                          
        self.k_min = 1.0                                                   
        self.kc_slope = nn.Parameter(torch.tensor(2.0))                                   
        self.kc_bias = nn.Parameter(torch.tensor(0.0))
        self.r_mlp = nn.Linear(self.C + pangu_ch, 1)                          
        self.tau_raw = nn.Parameter(_inverse_softplus(torch.tensor(0.7)))                    

                                                                  
        self.d_bias = nn.Parameter(torch.tensor(4.0))                
        self.d_slope = nn.Parameter(torch.tensor(0.0))                                

    def analyze(self, radar_feat):
        """Lead-time-invariant: sub-bands, spatial routing, radar descriptor."""
        bands = self.pyramid(radar_feat)                                     
        route = F.softmax(self.route(radar_feat), dim=1)             
        r_desc = radar_feat.mean(dim=(2, 3))                     
        return bands, route, r_desc

    def _film_coeffs(self, pangu_pool_t, e_t):
        """gamma, beta : (B, n_bands, G, C)."""
        B = pangu_pool_t.shape[0]
        gammas, betas = [], []
        for g, mlp in enumerate(self.film):
            idx = getattr(self, f"gidx_{g}")
            desc = pangu_pool_t.index_select(1, idx)                      
            out = mlp(torch.cat([desc, e_t], dim=1)).view(B, self.n_bands, 2, self.C)
            gammas.append(torch.tanh(out[:, :, 0]))                      
            betas.append(out[:, :, 1])
        return torch.stack(gammas, dim=2), torch.stack(betas, dim=2)

    def _authorization(self, t_idx, r_desc, pangu_pool_t):
        """a : (B, n_bands) per-band authorization; d0 : (B,1,1,1) finest-band decay."""
        B = r_desc.shape[0]
        device = r_desc.device
        t_norm = float(t_idx) / max(self.Tout - 1, 1)
        r_shift = self.r_mlp(torch.cat([r_desc, pangu_pool_t], dim=1)).squeeze(1)        
        slope = F.softplus(self.kc_slope) + 1e-3
        t_term = slope * (t_norm - 0.5) if self.use_breathe else 0.0
        rho = torch.sigmoid(t_term + self.kc_bias + r_shift)                                  
        k_c = self.k_max - (self.k_max - self.k_min) * rho                                   
        tau = F.softplus(self.tau_raw) + 1e-1
        j = torch.arange(self.n_bands, device=device).float().unsqueeze(0)                
        a = torch.sigmoid((j - k_c.unsqueeze(1)) / tau)                            
        if self.use_protect:
            a = a.clone(); a[:, 0] = 0.0                                                       
        if self.use_decay:
            d0 = torch.sigmoid(self.d_bias - F.softplus(self.d_slope) * t_norm).view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            d0 = torch.ones(B, 1, 1, 1, device=device)
        return a, d0

    def synthesize(self, bands, route, r_desc, pangu_pool_t, t_idx):
        B = bands[0].shape[0]
        device = bands[0].device
        e_t = self.time_embed(torch.full((B,), int(t_idx), device=device, dtype=torch.long))
        gamma, beta = self._film_coeffs(pangu_pool_t, e_t)                             
        a, d0 = self._authorization(t_idx, r_desc, pangu_pool_t)                             

        out = bands[0].new_zeros(bands[0].shape)
        for jb in range(self.n_bands):
                                                                                   
            g_eff = torch.einsum("bgc,bghw->bchw", gamma[:, jb], route)
            a_j = a[:, jb].view(B, 1, 1, 1)
            band = bands[jb] * (1.0 + a_j * g_eff)                                                    
            if jb == self.n_bands - 1:                                                               
                b_eff = torch.einsum("bgc,bghw->bchw", beta[:, jb], route)
                band = band + a_j * b_eff
            if jb == 0:                                                                            
                band = band * d0
            out = out + band
        return out

    def forward(self, radar_feat, pangu_pool_t, t_idx):
        bands, route, r_desc = self.analyze(radar_feat)
        return self.synthesize(bands, route, r_desc, pangu_pool_t, t_idx)


                           
                         
                           
                                                                                   
                                                                                  
                                                                       

class PRPF_SetGoGAN_Generator(nn.Module):
    """PRPF generator. Every fusion sub-module is individually toggleable so that
    per-module ablations (CS-MMSA / cross-attention / channel fusion / LTAM / Swin /
    temporal) and capacity sweeps can be run from a single class without editing code.

    Args
    ----
    att_dim          : internal feature width C (fusion width = 2*att_dim).
    swin_depth       : number of Swin blocks in the spatial-refinement stage.
    use_ltam         : radar-history temporal modulation (LTAM).
    use_csmmsa       : F^(0) predictability-graded spectral fusion (PASM/SAC).
                       (Flag name kept for ablation-tooling back-compat.)
    use_crossattn    : global cross-attention alignment branch -> F^(1).
    use_channel_fusion: bidirectional channel cross-attention + channel self-attention.
    use_swin         : Swin spatial refinement.
    use_temporal     : temporal refinement (3D conv + temporal channel self-attention).
    Disabled pangu-consuming branches are not instantiated, so each ablation variant
    reports its own honest parameter count.
    """
    def __init__(self,
                 obs_in_shape=(5,1,128,128),
                 wfm_in_shape=(5,34,32,32),
                 num_lead_times=20,
                 att_dim=128,
                 swin_depth=4,
                 window_size=8,
                 use_ltam=True,
                 use_csmmsa=True,
                 use_crossattn=True,
                 use_channel_fusion=True,
                 use_swin=True,
                 use_temporal=True,
                                                                                 
                 pasm_levels=3,
                 pasm_breathe=True,
                 pasm_protect=True,
                 pasm_group=True,
                 pasm_decay=True):
        super().__init__()
        Tin, Cin, Hr, Wr = obs_in_shape
        Tw, Cw, Hp, Wp = wfm_in_shape
        self.Tout = num_lead_times
        self.window_size = window_size

                           
        self.use_ltam = use_ltam
        self.use_csmmsa = use_csmmsa
        self.use_crossattn = use_crossattn
        self.use_channel_fusion = use_channel_fusion
        self.use_swin = use_swin
        self.use_temporal = use_temporal
                                                                               
                                                                         
        self._need_pangu_enc = use_crossattn

        if self.use_ltam:
            self.ltam_mod = RadarLTAMModulator(Tin, Cin, embed_dim=att_dim, heads=4)
        self.radar_enc = RadarEncoder(Tin, Cin, att_dim)
        if self._need_pangu_enc:
            self.pangu_enc = PanguTimeAlignEncoder(Cw, num_lead_times)

                                                                                   
                                                                                
                                                                         
        if self.use_csmmsa:
            self.fusion_block = PASMFusion(
                radar_dim=att_dim, pangu_ch=Cw, num_levels=pasm_levels,
                Tout=num_lead_times,
                use_breathe=pasm_breathe, use_protect=pasm_protect,
                use_group=pasm_group, use_decay=pasm_decay,
            )
        if self.use_crossattn:
            self.to_att = nn.Conv2d(Cw, att_dim, kernel_size=1, bias=False)
            self.cross_align = CrossAttentionAlign(Cw=att_dim, Cd=att_dim, nheads=4, mlp_ratio=4.0)

        self.fusion_dim = att_dim * 2
        if self.use_channel_fusion:
            self.chan_cross = ChannelCrossAttentionGlobal(Ca=att_dim, Cb=att_dim, d=att_dim, heads=4)
            self.chan_self = ChannelSelfAttentionGlobal(C=self.fusion_dim, d=att_dim, heads=4)

        if self.use_swin:
            self.swin_texture = SwinBasicLayer(dim=self.fusion_dim, depth=swin_depth,
                                               num_heads=4, window_size=window_size)

        if self.use_temporal:
            self.temporal_conv = nn.Sequential(
                 nn.Conv3d(self.fusion_dim, self.fusion_dim, kernel_size=(3,3,3), padding=(1,1,1)),
                 nn.GroupNorm(_gn_groups(self.fusion_dim), self.fusion_dim),
                 nn.SiLU()
            )
            self.temporal_attn = TemporalCrossAttention(self.fusion_dim, heads=4)

        self.decoder = DualHeadDecoder(self.fusion_dim)

    def forward(self, x_obs, x_wfm):
        B = x_obs.shape[0]

        x_obs_mod = self.ltam_mod(x_obs) if self.use_ltam else x_obs
        radar_feat = self.radar_enc(x_obs_mod)
        pangu_seq = self.pangu_enc(x_wfm) if self._need_pangu_enc else None

                                    
                                                                                
                                                                              
                                                                                
                                                              
        if self.use_csmmsa:
            pangu_pool = x_wfm.mean(dim=(3, 4))                                         
            pangu_pool = F.interpolate(pangu_pool.permute(0, 2, 1), size=self.Tout,
                                       mode='linear', align_corners=False).permute(0, 2, 1)               
            pasm_bands, pasm_route, pasm_rdesc = self.fusion_block.analyze(radar_feat)

        fused_seq = []
        for t in range(self.Tout):
            p_t = pangu_seq[:, t] if self._need_pangu_enc else None

                                                                                     
            if self.use_csmmsa:
                F0 = self.fusion_block.synthesize(pasm_bands, pasm_route, pasm_rdesc,
                                                  pangu_pool[:, t], t)
            else:
                F0 = radar_feat
                                                                                         
            if self.use_crossattn:
                F1 = self.cross_align(radar_feat, self.to_att(p_t))
            else:
                F1 = radar_feat

                                                                                  
            if self.use_channel_fusion:
                fused_a, fused_b = self.chan_cross(F0, F1)
                fused_t = torch.cat([fused_a, fused_b], dim=1)
                fused_t = self.chan_self(fused_t)
            else:
                fused_t = torch.cat([F0, F1], dim=1)

            if self.use_swin:
                fused_pad, pads_swin = pad_tensor(fused_t, self.window_size)
                fused_pad = self.swin_texture(fused_pad)
                fused_t = pad_tensor_back(fused_pad, pads_swin)

            fused_seq.append(fused_t.unsqueeze(2))

        feat_vol = torch.cat(fused_seq, dim=2)
        if self.use_temporal:
            feat_vol = self.temporal_conv(feat_vol)
            feat_vol = self.temporal_attn(feat_vol)

        feat_flat = feat_vol.permute(0, 2, 1, 3, 4).reshape(B * self.Tout, self.fusion_dim, 64, 64)

                                                                                                  
        pred_radar, res_flow = self.decoder(feat_flat)
        pred_radar = pred_radar.view(B, self.Tout, 1, 128, 128)
        res_flow = res_flow.view(B, self.Tout, 2, 128, 128)
        return pred_radar, res_flow