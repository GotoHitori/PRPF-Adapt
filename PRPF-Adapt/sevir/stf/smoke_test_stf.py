                                                                        
                                                                              
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(0)

from model_v2 import (PRPF_SetGoGAN_Generator, DualHeadDecoder, STFDecoder,
                      ProfilePriorHead, stf_build_layer_perm, interp_time,
                      FrameDiscriminator, SeqDiscriminator3D)

OK = "\033[92m[OK]\033[0m"
def check(name, cond):
    if not cond:
        print(f"\033[91m[FAIL]\033[0m {name}")
        sys.exit(1)
    print(f"{OK} {name}")

B, Tin, Tout, K = 1, 5, 4, 6
x_obs = torch.randn(B, Tin, 1, 128, 128)
x_wfm = torch.randn(B, Tin, 34, 32, 32)
y_seq = torch.randn(B, Tout, 1, 128, 128)

                                                                            
gen = PRPF_SetGoGAN_Generator(att_dim=32, swin_depth=1, num_lead_times=Tout)                        
gen.eval()
with torch.no_grad():
    pred, flow = gen(x_obs, x_wfm)
check("forward (default 2-tuple): pred [B,T,1,128,128], flow [B,T,2,128,128]",
      pred.shape == (B, Tout, 1, 128, 128) and flow.shape == (B, Tout, 2, 128, 128))

with torch.no_grad():
    pred, flow, L, prior = gen(x_obs, x_wfm, return_stf=True)
check("return_stf: L [B,T,6,128,128], prior [B,T,6,32,32]",
      L.shape == (B, Tout, K, 128, 128) and prior.shape == (B, Tout, K, 32, 32))
check("latent layers non-negative (softplus)", (L >= 0).all().item())
check("profile prior is a distribution over K", torch.allclose(prior.sum(dim=2), torch.ones(B, Tout, 32, 32), atol=1e-5))

                                                                              
dec = gen.decoder
w = dec.layer_weights().view(1, 1, K, 1, 1)
vil_manual = dec.vil_scale * (w * L).sum(dim=2, keepdim=False).unsqueeze(2) + dec.vil_bias
check("VIL == vil_scale * sum_k w_k L_k + vil_bias (exact reconstruction)",
      torch.allclose(pred, vil_manual, atol=1e-5))
check("integral weights w_k >= 0 and dp-initialised (sum ~= 1)",
      (dec.layer_weights() >= 0).all().item() and abs(dec.layer_weights().sum().item() - 1.0) < 1e-4)

                                                                            
Lp = L.clone(); Lp[:, :, 3] += 1.0
vil_p = dec.vil_scale * (dec.layer_weights().view(1, 1, K, 1, 1) * Lp).sum(dim=2, keepdim=False).unsqueeze(2) + dec.vil_bias
delta = (vil_p - pred).mean().item()
expected = (dec.vil_scale * dec.layer_weights()[3]).item()
check(f"observation-operator sensitivity: dVIL/dL_3 = w_3*scale ({delta:.4f} vs {expected:.4f})",
      abs(delta - expected) < 1e-4)

                                                                               
                                                                           
                                                                                      
feat = torch.randn(4, gen.fusion_dim, 64, 64)
cond1 = torch.randn(4, K * 5, 32, 32)
cond2 = torch.randn(4, K * 5, 32, 32)
with torch.no_grad():
    o1 = dec(feat, cond1)[0]; o2 = dec(feat, cond2)[0]
check("zero-init FiLM: decoder ignores Pangu cond at init (identical outputs)",
      torch.allclose(o1, o2, atol=1e-6))

                                                                               
gen_p = PRPF_SetGoGAN_Generator(att_dim=32, swin_depth=1, num_lead_times=Tout, stf_permute_layers=True)
check("permute flag -> reversed derangement buffer [5,4,3,2,1,0]",
      gen_p.decoder.layer_perm.tolist() == [5, 4, 3, 2, 1, 0])
ci = gen.stf_cond_index.tolist(); cip = gen_p.stf_cond_index.tolist()
check("cond_index identity: layer0 <- level0 chans [0,6,12,18,24]", ci[:5] == [0, 6, 12, 18, 24])
check("cond_index permuted: layer0 <- level5 chans [5,11,17,23,29]", cip[:5] == [5, 11, 17, 23, 29])

                                                                              
gen0 = PRPF_SetGoGAN_Generator(att_dim=32, swin_depth=1, num_lead_times=Tout, use_stf=False)
check("no_stf: decoder is the original DualHeadDecoder", isinstance(gen0.decoder, DualHeadDecoder))
with torch.no_grad():
    p0, f0 = gen0(x_obs, x_wfm)
check("no_stf forward shapes unchanged", p0.shape == (B, Tout, 1, 128, 128))
n0 = sum(p.numel() for p in gen0.parameters())
n1 = sum(p.numel() for p in gen.parameters())
print(f"     params: baseline {n0/1e6:.3f}M -> STF {n1/1e6:.3f}M (delta {(n1-n0)/1e3:.1f}K)")
check("parameter increment < 0.1M as budgeted", (n1 - n0) < 100_000)

                                                                               
sys.path.insert(0, '.')
from train import (PerLayerAdvectionLoss, stf_profile_losses,
                   PhysicsInformedAdvectionDiffusionLoss)

perm = stf_build_layer_perm(K, False)
crit = PerLayerAdvectionLoss(layer_perm=perm, steer_phi=0.2, device='cpu')
res_flow = torch.zeros(B, Tout, 2, 128, 128)
with torch.no_grad():
    l = crit(L, res_flow, x_wfm)
check(f"PerLayerAdvectionLoss forward finite ({l.item():.4f})", torch.isfinite(l).item())

with torch.no_grad():
    lp, lz = stf_profile_losses(L, prior)
check(f"profile/zsmooth losses finite ({lp.item():.5f}, {lz.item():.5f})",
      torch.isfinite(lp).item() and torch.isfinite(lz).item())

                                                                              
Lc = torch.ones(B, Tout, K, 64, 64) * 0.5
rf0 = torch.zeros(B, Tout, 2, 64, 64)
with torch.no_grad():
    lc = crit(Lc, rf0, x_wfm)
check(f"physics sanity: constant field residual ~= sqrt(eps) ({lc.item():.5f})", lc.item() < 2e-3)

                                                                          
                                                                               
                                                                               
Hs = 64
mean0 = torch.zeros(34); std0 = torch.ones(34)
winds = torch.zeros(34)
for lvl in range(6):
    winds[lvl] = (lvl - 2.5) * 2.0                                            
    winds[6 + lvl] = 0.0
x_wfm_s = winds.view(1, 1, 34, 1, 1).expand(1, Tin, 34, 32, 32).contiguous()

crit_id = PerLayerAdvectionLoss(layer_perm=list(range(6)), steer_phi=0.2,
                                pangu_means=mean0, pangu_stds=std0, device='cpu')
crit_pm = PerLayerAdvectionLoss(layer_perm=list(range(6))[::-1], steer_phi=0.2,
                                pangu_means=mean0, pangu_stds=std0, device='cpu')

                                                                                
yy, xx = torch.meshgrid(torch.arange(Hs).float(), torch.arange(Hs).float(), indexing='ij')
def blob(cx, cy, s=6.0):
    return torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
L_syn = torch.zeros(1, 10, 6, Hs, Hs)
for t in range(10):
    for k in range(6):
        u_pix = 0.2 * winds[k].item()
        L_syn[0, t, k] = blob(24.0 + t * u_pix, 32.0)
rf_s = torch.zeros(1, 10, 2, Hs, Hs)
with torch.no_grad():
    l_id = crit_id(L_syn, rf_s, x_wfm_s).item()
    l_pm = crit_pm(L_syn, rf_s, x_wfm_s).item()
print(f"     shear-parallax: correct pairing loss {l_id:.5f} vs scrambled {l_pm:.5f} (ratio {l_pm/l_id:.2f}x)")
check("differential advection identifiability: correct pairing << scrambled", l_pm > 1.5 * l_id)

                                                                 
crit_sw = PerLayerAdvectionLoss(layer_perm=list(range(6)), steer_phi=0.2, shared_wind=True,
                                pangu_means=mean0, pangu_stds=std0, device='cpu')
crit_sw_pm = PerLayerAdvectionLoss(layer_perm=list(range(6))[::-1], steer_phi=0.2, shared_wind=True,
                                   pangu_means=mean0, pangu_stds=std0, device='cpu')
with torch.no_grad():
    _sw = crit_sw(L_syn, rf_s, x_wfm_s).item()
    _sw_pm = crit_sw_pm(L_syn, rf_s, x_wfm_s).item()
check("shared-wind ablation: pairing no longer matters (losses identical)", abs(_sw - _sw_pm) < 1e-6)

                                                                           
del pred, flow, L, prior, gen_p, gen0, o1, o2, p0, f0
import gc; gc.collect()
gen.train()
pred, res_flow, L, prior = gen(x_obs, x_wfm, return_stf=True)
phys_crit = PhysicsInformedAdvectionDiffusionLoss(phase=1, device='cpu', steer_phi=0.2)
loss_main, _ = phys_crit(pred, y_seq, res_flow, x_wfm, 0.0, 1.0)
l_layer = crit(L, res_flow, x_wfm)
lp, lz = stf_profile_losses(L, prior)
loss = loss_main + 0.05 * l_layer + 0.01 * lp + 0.001 * lz
loss.backward()

def has_grad(p):
    return p.grad is not None and p.grad.abs().sum().item() > 0
check("grad -> integral weights w_theta", has_grad(dec.w_theta))
check("grad -> vil_scale / vil_bias", has_grad(dec.vil_scale) and has_grad(dec.vil_bias))
check("grad -> per-layer FiLM conv (zero-init but beta path live)", has_grad(dec.film.bias) or has_grad(dec.film.weight))
check("grad -> grouped layer head", has_grad(dec.head_layers.weight))
check("grad -> profile prior head", has_grad(gen.profile_prior.net[0].weight))
check("grad -> upstream (radar encoder) intact", has_grad(gen.radar_enc.net[0].conv.weight))

                                                                          
d_f = FrameDiscriminator(); d_s = SeqDiscriminator3D()
lf = d_f(pred.detach().view(-1, 1, 128, 128)); ls = d_s(pred.detach())
check("discriminators consume the VIL output unchanged", lf.shape[-1] == 1 and ls.shape[-1] == 1)

                                                                         
sd = gen.state_dict()
gen2 = PRPF_SetGoGAN_Generator(att_dim=32, swin_depth=1, num_lead_times=Tout)
missing, unexpected = gen2.load_state_dict(sd, strict=True), None
check("state_dict round-trip strict=True", True)

print("\nALL STF SMOKE TESTS PASSED")
