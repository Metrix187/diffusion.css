"""Quantise a checkpoint and compare the integer sampler with the float one.

  python evalq.py model.pt [key=value ...]     e.g. prune_in=1 prune_out=1 tag=_q
"""
import sys

import numpy as np
import torch

import cssgen
import quant
import sprites
from model import sample_float

path = sys.argv[1]
opts = dict(cap_in=127, cap_mid=255, cap_out=255, act_target=2000, prune_in=0, prune_out=0)
tag = ''
for a in sys.argv[2:]:
    k, v = a.split('=')
    if k == 'tag':
        tag = v
    else:
        opts[k] = int(v)

ck, model = quant.load(path)
Q = quant.quantize(ck, model, **opts)
seeds = np.random.default_rng(7).integers(0, cssgen.SEED_MAX + 1, 4000)
stats = {}
xi = quant.int_sample(Q, seeds, stats=stats) / 256.0
xf = sample_float(model, ck['mean'], ck['alpha'], ck['sigma'],
                  torch.from_numpy(cssgen.noise_field(seeds) / 256.0).float()).numpy()

for name, x in (('float', xf), ('int  ', xi)):
    res, cls = sprites.fit_residual(x)
    print('%s residual (8-bit): mean %.2f median %.2f p90 %.2f p99 %.2f max %.1f | >8: %.1f%% | classes %s'
          % (name, res.mean(), np.median(res), np.percentile(res, 90), np.percentile(res, 99), res.max(),
             100 * (res > 8).mean(), np.bincount(cls, minlength=10).tolist()))
print('int vs float rms diff (8-bit): %.2f' % (np.sqrt(((xi - xf) ** 2).mean()) * 127.5))
worst = {}
for (s, k), v in stats.items():
    if k != 'h':
        worst[k] = max(worst.get(k, 0), v)
print('worst sum|terms| per layer:', {k: f'{v:,}' for k, v in worst.items()},
      '| float32 limit %s' % f'{cssgen.F32_SAFE:,}')
print('max hidden activation', max(v for (s, k), v in stats.items() if k == 'h'))
nz, tot = quant.count_weights(Q)
print('weights: %d nonzero of %d (%.0f%% zero)' % (nz, tot, 100 * (1 - nz / tot)))
for s, lv in enumerate(Q['levels'][1:], 1):
    print('  level %2d shifts r=%s scales=%s |Wi|max %d |V|max %d nzWi %.0f%% nzV %.0f%%' % (
        s, (lv['r1'], lv['r2'], lv['r3'], lv['r4']), lv['scales'], np.abs(lv['Wi']).max(),
        np.abs(lv['V']).max(), 100 * np.count_nonzero(lv['Wi']) / lv['Wi'].size,
        100 * np.count_nonzero(lv['V']) / lv['V'].size))
sprites.sheet(xi[:80], 10, scale=6).save(f'samples_int{tag}.png')
