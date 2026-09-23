import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def copy_shared_weights(source: nn.Module, target: nn.Module):
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name in set(source_state).intersection(target_state):
        if source_state[name].shape == target_state[name].shape:
            target_state[name] = source_state[name]
    target.load_state_dict(target_state)

                           
                             
                           

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

class RadarSimVPEncoder(nn.Module):
    def __init__(self, T, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv2d(T*in_ch, 64, s=2), 
            SimVPConvBlock(64),
            SimVPConvBlock(64),
            BasicConv2d(64, out_ch, s=1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        return self.net(x)

class PanguSimVPEncoder(nn.Module):
    def __init__(self, Cw):
        super().__init__()
        self.net = nn.Sequential(BasicConv2d(Cw, Cw), SimVPConvBlock(Cw), SimVPConvBlock(Cw))
    def forward(self, x):
        B, T, C, H, W = x.shape
        y = x.view(B*T, C, H, W)
        return self.net(y).view(B, T, C, H, W)

class Temporal3DUpsampler(nn.Module):
    def __init__(self, Cw, Tout):
        super().__init__()
        self.dw3d = nn.Conv3d(Cw, Cw, (3,3,3), padding=(1,1,1), bias=False)
        self.norm = nn.GroupNorm(_gn_groups(Cw), Cw)
        self.act = nn.SiLU()
        self.Tout = Tout
    def forward(self, x):
        x = x.permute(0,2,1,3,4) 
        x = F.interpolate(x, size=(self.Tout, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False)
        return self.act(self.norm(self.dw3d(x))).permute(0,2,1,3,4)

class DualHeadDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BasicConv2d(in_ch, 64)
        self.head_radar = nn.Conv2d(64, 1, 3, 1, 1)
        self.head_flow = nn.Conv2d(64, 2, 3, 1, 1) 
    def forward(self, x):
        x = self.up(self.conv(x))
                                      
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


class OutputResidualCalibrator(nn.Module):
    def __init__(self, pangu_dim, lead_times=20, hidden_dim=16,
                 max_residual_dbz=0.5, radar_scale=70.0):
        super().__init__()
        self.lead_times = int(lead_times)
        self.max_residual = float(max_residual_dbz) / float(radar_scale)
        self.radar_scale = float(radar_scale)
        self.pangu_mapper = nn.Conv2d(pangu_dim, 8, 1)
        self.context = nn.Sequential(
            nn.Conv2d(12, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.residual_head = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer("thresholds_dbz", torch.tensor((12.0, 24.0, 32.0)))
        self.threshold_bias_dbz = nn.Parameter(torch.zeros(3))
        self.regularizer = torch.tensor(0.0)

    def forward(self, base, x_obs, pangu_seq):
        batch, leads, _, height, width = base.shape
        if leads != self.lead_times:
            raise ValueError(f"expected {self.lead_times} lead times, got {leads}")
        pangu = pangu_seq.reshape(batch * leads, pangu_seq.shape[2], *pangu_seq.shape[-2:])
        pangu = F.interpolate(pangu, size=(height, width), mode="bilinear", align_corners=False)
        pangu = self.pangu_mapper(pangu).view(batch, leads, 8, height, width)
        last = x_obs[:, -1:].expand(-1, leads, -1, -1, -1)
        change = (x_obs[:, -1:] - x_obs[:, -2:-1]).expand(-1, leads, -1, -1, -1)
        lead = torch.linspace(0.0, 1.0, leads, device=base.device, dtype=base.dtype)
        lead = lead.view(1, leads, 1, 1, 1).expand(batch, -1, -1, height, width)
        features = torch.cat((base, last, change, pangu, lead), dim=2)
        raw = self.residual_head(self.context(features.flatten(0, 1))).view_as(base)

        distance = (base * self.radar_scale - self.thresholds_dbz.view(1, 1, -1, 1, 1)).abs()
        threshold_gates = torch.sigmoid((3.0 - distance) / 1.0)
        band_gate = threshold_gates.amax(dim=2, keepdim=True)
        residual = self.max_residual * torch.tanh(raw) * (0.05 + 0.95 * band_gate)
        threshold_bias = (
            threshold_gates * self.threshold_bias_dbz.view(1, 1, -1, 1, 1)
        ).sum(dim=2, keepdim=True) / self.radar_scale
        calibration = torch.clamp(
            residual + threshold_bias, -self.max_residual, self.max_residual
        )
        self.regularizer = calibration.square().mean()
        return torch.relu(base + calibration)

                           
                          
                           
PANGU_U_IDX = 0
PANGU_V_IDX = 4

class PRPF_SetGoGAN_Generator(nn.Module):
    def __init__(self,
                 obs_in_shape=(5,1,128,128),
                 wfm_in_shape=(5,34,32,32),
                 num_lead_times=20,
                 att_dim=64,
                 window_size=8,
                 router_echo_threshold=1.41,
                 router_max_residual_dbz=0.5,
                 model_variant="m0"):
        super().__init__()
        Tin, Cin, Hr, Wr = obs_in_shape
        Tw, Cw, Hp, Wp = wfm_in_shape
        self.Tout = num_lead_times
        self.window_size = window_size
        if model_variant not in {"m0", "m3"}:
            raise ValueError("model_variant must be m0 or m3")
        self.model_variant = model_variant

        self.ltam_mod = RadarLTAMModulator(Tin, Cin, embed_dim=att_dim, heads=4)
        self.radar_enc = RadarSimVPEncoder(Tin, Cin, att_dim)
        self.pangu_enc = PanguSimVPEncoder(Cw)
        self.pangu_up = Temporal3DUpsampler(Cw, num_lead_times)

        self.fusion_block = CS_MMSA_Block(radar_dim=att_dim, pangu_dim=Cw, num_heads=4)
        if self.model_variant == "m3":
            self.reliability_router = OutputResidualCalibrator(
                Cw,
                num_lead_times,
                max_residual_dbz=router_max_residual_dbz,
            )
        self.to_att = nn.Conv2d(Cw, att_dim, kernel_size=1, bias=False)
        self.cross_align = CrossAttentionAlign(Cw=att_dim, Cd=att_dim, nheads=4, mlp_ratio=4.0)
        self.chan_cross = ChannelCrossAttentionGlobal(Ca=att_dim, Cb=att_dim, d=att_dim, heads=4)
        self.fusion_dim = att_dim * 2
        self.chan_self = ChannelSelfAttentionGlobal(C=self.fusion_dim, d=att_dim, heads=4)
        self.swin_texture = SwinBasicLayer(dim=self.fusion_dim, depth=2, num_heads=4, window_size=window_size)

        self.temporal_conv = nn.Sequential(
             nn.Conv3d(self.fusion_dim, self.fusion_dim, kernel_size=(3,3,3), padding=(1,1,1)),
             nn.GroupNorm(_gn_groups(self.fusion_dim), self.fusion_dim),
             nn.SiLU()
        )
        self.temporal_attn = TemporalCrossAttention(self.fusion_dim, heads=4)

                 
        self.decoder = DualHeadDecoder(self.fusion_dim)

    def forward(self, x_obs, x_wfm):
        B = x_obs.shape[0]

        x_obs_mod = self.ltam_mod(x_obs)
        radar_feat = self.radar_enc(x_obs_mod)
        pangu_seq = self.pangu_up(self.pangu_enc(x_wfm))

        fused_seq = []
        for t in range(self.Tout):
            p_t = pangu_seq[:, t]
            p_t_d = self.to_att(p_t)

            fused_base = self.fusion_block(radar_feat, p_t)
            aligned = self.cross_align(radar_feat, p_t_d)
            fused_a, fused_b = self.chan_cross(fused_base, aligned)
            fused_t = torch.cat([fused_a, fused_b], dim=1)
            fused_t = self.chan_self(fused_t)

            fused_pad, pads_swin = pad_tensor(fused_t, self.window_size)
            fused_pad = self.swin_texture(fused_pad)
            fused_t = pad_tensor_back(fused_pad, pads_swin)

            fused_seq.append(fused_t.unsqueeze(2))

        feat_vol = torch.cat(fused_seq, dim=2)
        feat_vol = self.temporal_conv(feat_vol)
        feat_vol = self.temporal_attn(feat_vol)

        feat_flat = feat_vol.permute(0, 2, 1, 3, 4).reshape(B * self.Tout, self.fusion_dim, 64, 64)
        
                          
        pred_radar, res_flow = self.decoder(feat_flat)
        pred_radar = pred_radar.view(B, self.Tout, 1, 128, 128)
        
                                                                                       
        pred_radar = torch.relu(pred_radar)
        if self.model_variant == "m3":
            pred_radar = self.reliability_router(pred_radar, x_obs, pangu_seq)
        
        res_flow = res_flow.view(B, self.Tout, 2, 128, 128)

        self.router_regularizer = (
            self.reliability_router.regularizer
            if self.model_variant == "m3" else pred_radar.new_zeros(())
        )

        return pred_radar, res_flow
