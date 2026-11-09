"""Reference results for checking the page against the integer sampler.

  python verify.py model.pt [n]     write verify_expected.json: [[seed, checksum of the final image], ...]

Then open index.html from a local server and paste verify_snippet.js into the DevTools console.
The snippet is a test harness only: it pins --seed from outside and reads the registers back.
The page itself still contains no script.
"""
import json
import sys

import numpy as np

import cssgen
import quant

ck, model = quant.load(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
Q = quant.quantize(ck, model)
seeds = np.random.default_rng(2026).integers(0, cssgen.SEED_MAX + 1, n)
stats = {}
final = quant.int_sample(Q, seeds, stats=stats)
last = chr(ord('a') + Q['T'] - 1)
rows = [[int(s), cssgen.checksum(x.tolist())] for s, x in zip(seeds, final)]
with open('verify_expected.json', 'w') as f:
    json.dump({'last': last, 'rows': rows}, f)
print('wrote', len(rows), 'seeds; final-state registers are --%s0..--%s767' % (last, last))
