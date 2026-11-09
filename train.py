"""Train the per-level denoisers.   python train.py [key=value ...]

Three things happen in one run:
  1. plain denoising training (predict x0 from x_j), all noise levels in parallel
  2. gradual magnitude pruning of the weights, so the stylesheet has fewer terms to spell out
  3. a final stretch with the weights fake-quantised to the integers the stylesheet will hold
"""
import math
import sys
import time

import numpy as np
import torch

import cssgen
import sprites
from model import Denoisers, sample_float, schedule

cfg = dict(T=12, lam_min=-6.0, lam_max=6.0, g_in=12, mid=64, g_out=20, iters=8000, batch=256,
           lr=3e-3, mix=0.45, mix_from=0.2, seed=0, out='model.pt', data=20000, tag='',
           sp_in=0.0, sp_mid=0.0, sp_out=0.0, sp_start=0.2, sp_end=0.65, fq=0,
           cap_in=127, cap_mid=255, cap_out=255, init='', mix_late=0.15, sp_late=0.5, threads=8)
for a in sys.argv[1:]:
    k, v = a.split('=')
    cfg[k] = type(cfg[k])(v)
print(cfg)

torch.set_num_threads(cfg['threads'])
torch.manual_seed(cfg['seed'])
rng = np.random.default_rng(cfg['seed'])
T = cfg['T']
alpha, sigma = schedule(T, cfg['lam_min'], cfg['lam_max'])
print('alpha', np.round(alpha, 3))

data, _ = sprites.sample_batch(cfg['data'], rng)
data = torch.from_numpy(data)
mean = data.mean(0)

model = Denoisers(T - 1, cfg['g_in'], cfg['mid'], cfg['g_out'], mean=mean)
if cfg['init']:
    model.load_state_dict(torch.load(cfg['init'], weights_only=False)['state'])
opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['iters'], eta_min=cfg['lr'] * 0.03)
a_t = torch.tensor(alpha[1:T], dtype=torch.float32).view(-1, 1, 1)
s_t = torch.tensor(sigma[1:T], dtype=torch.float32).view(-1, 1, 1)
# blend augmentation: off while the sprite is still undecided, full strength through the middle,
# light at the last levels (those mostly need to reproduce a clean sprite exactly)
mix_max = (a_t >= cfg['mix_from']).float() * cfg['mix']
mix_max = torch.where(a_t >= 0.97, torch.full_like(mix_max, cfg['mix_late']), mix_max)
# pruning is heavier at the noisy levels (their output is blurry anyway) and lighter at the last ones
sp_shape = [1.3 if al < 0.2 else (cfg['sp_late'] if al >= 0.97 else 1.0) for al in alpha[1:T]]

# ---- pruning masks and fake quantisation, applied through the model's weight hook
target = {'w_in': cfg['sp_in'], 'w_m1': cfg['sp_mid'], 'w_m2': cfg['sp_mid'], 'w_out': cfg['sp_out']}
caps = {'w_in': cfg['cap_in'], 'w_m1': cfg['cap_mid'], 'w_m2': cfg['cap_mid'], 'w_out': cfg['cap_out']}
masks = {k: torch.ones_like(getattr(model, k)) for k in target}
fq_on = False


def update_masks(frac):
    for k, sp in target.items():
        if sp <= 0:
            continue
        w = getattr(model, k).detach().abs()
        srt = w.flatten(1).sort(dim=1).values
        n = srt.shape[1]
        thr = torch.zeros(w.shape[0])
        for l in range(w.shape[0]):
            kth = int(frac * min(0.92, sp * sp_shape[l]) * n)
            thr[l] = srt[l, kth - 1] if kth >= 1 else -1.0
        masks[k] = (w > thr.view(-1, *([1] * (w.dim() - 1)))).float()


def wq(name, w):
    w = w * masks[name]
    if fq_on:
        mx = w.detach().abs().flatten(1).max(dim=1).values.clamp_min(1e-9)
        scale = 2.0 ** torch.floor(torch.log2(caps[name] / mx)).view(-1, *([1] * (w.dim() - 1)))
        w = w + (torch.round(w * scale) / scale - w).detach()
    return w


i0, i1 = int(cfg['sp_start'] * cfg['iters']), int(cfg['sp_end'] * cfg['iters'])
t0 = time.time()
for it in range(1, cfg['iters'] + 1):
    if i0 <= it <= i1 and (it % 50 == 0 or it == i1):
        u = (it - i0) / max(1, i1 - i0)
        update_masks(1 - (1 - u) ** 3)
    if cfg['fq'] and it == int(0.75 * cfg['iters']):
        fq_on = True
    x0 = data[torch.randint(0, data.shape[0], (cfg['batch'],))]
    eps = torch.randn(T - 1, cfg['batch'], 768)
    # Robustness: at the lower-noise levels, half the inputs are blended with a second sprite and the
    # target stays the dominant one, so those networks learn to snap a muddled state to one sprite.
    other = data[torch.randint(0, data.shape[0], (cfg['batch'],))]
    lam = torch.rand(T - 1, cfg['batch'], 1) * mix_max * (torch.rand(T - 1, cfg['batch'], 1) < 0.5)
    xt = a_t * (x0 + lam * (other - x0)) + s_t * eps
    pred = model(xt, wq)
    per_level = ((pred - x0) ** 2).mean(dim=(1, 2))
    loss = per_level.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if it % 1000 == 0 or it == cfg['iters']:
        rm = (per_level.detach().sqrt() * 127.5).numpy()
        print(f'it {it:5d}  {time.time() - t0:5.0f}s  rmse/level (8-bit):', np.round(rm, 1), flush=True)

with torch.no_grad():                         # bake the masks in; quantisation happens in quant.py
    for k in masks:
        getattr(model, k).mul_(masks[k])
torch.save({'cfg': cfg, 'state': model.state_dict(), 'mean': mean, 'alpha': alpha, 'sigma': sigma},
           cfg['out'])

# ---- sample from the page's own noise and score
seeds = np.random.default_rng(7).integers(0, cssgen.SEED_MAX + 1, 2000)
noise = torch.from_numpy(cssgen.noise_field(seeds) / 256.0).float()
x = sample_float(model, mean, alpha, sigma, noise).numpy()
res, cls = sprites.fit_residual(x)
print('float sampler: residual (8-bit) mean %.2f median %.2f p90 %.2f p99 %.2f max %.1f | >8: %.1f%%'
      % (res.mean(), np.median(res), np.percentile(res, 90), np.percentile(res, 99), res.max(),
         100 * (res > 8).mean()))
print('class counts', np.bincount(cls, minlength=len(sprites.NAMES)).tolist())
sprites.sheet(x[:80], 10, scale=6).save(f'samples{cfg["tag"]}.png')
