"""Checkpoint -> index.html, plus the reference numbers the page must reproduce.

  python export.py model.pt                 write index.html
  python export.py model.pt site=https://you.github.io/diffusion.css/ repo=https://github.com/you/diffusion.css
                                            same, plus link-preview tags and a footer link to the source
  python export.py model.pt --eval SEED     print checksums of every sampler state for that seed
  python export.py model.pt --png SEED      also save the reference trajectory as ref_SEED.png
"""
import sys

import numpy as np

import cssgen
import quant
import sprites
from page import build_page

path = sys.argv[1]
ck, model = quant.load(path)
Q = quant.quantize(ck, model)
g, noise, steps = quant.build_graph(Q)
T = Q['T']

if len(sys.argv) > 3 and sys.argv[2] in ('--eval', '--png'):
    seed = int(sys.argv[3])
    stats = {}
    env = g.evaluate(seed, stats)                       # the slow, literal interpreter of the graph
    fast, traj = quant.int_sample(Q, [seed], keep=True)  # the vectorised sampler
    assert [env[n] for n in steps[-1]] == fast[0].tolist(), 'graph and numpy samplers disagree'
    print('seed', seed, 'digits', env['s1'], env['s2'], env['s3'])
    print('noise', cssgen.checksum([env[n] for n in noise]))
    for k, names in enumerate(steps, 1):
        print('step', k, cssgen.checksum([env[n] for n in names]))
    print('final first 9', [env[n] for n in steps[-1][:9]])
    res, cls = sprites.fit_residual(fast / 256.0)
    print('looks like:', sprites.NAMES[cls[0]], '(residual %.1f)' % res[0])
    print('worst sum|terms|', f'{max(stats.values()):,}', 'limit', f'{cssgen.F32_SAFE:,}')
    if sys.argv[2] == '--png':
        sprites.sheet([t[0] / 256.0 for t in traj], T + 1, scale=8).save(f'ref_{seed}.png')
    sys.exit()

kw = {k: v for k, v in (a.split('=', 1) for a in sys.argv[2:] if '=' in a) if k in ('site', 'repo')}
nz, tot = quant.count_weights(Q)
info = dict(weights=f'{nz:,}', registers=f'{len(g.order) + 768 + 6:,}', size='?', networks=str(T - 1))
html = build_page(g, noise, steps, info, **kw)
mb = len(html.encode()) / 1e6
info['size'] = f'{mb:.1f} MB'
html = build_page(g, noise, steps, info, **kw)
with open('index.html', 'w', encoding='utf-8', newline='\n') as f:
    f.write(html)
print('weights %d nonzero of %d | registers %d | terms %d | %.2f MB' % (
    nz, tot, len(g.order) + 768 + 6, g.n_terms(), len(html.encode()) / 1e6))
