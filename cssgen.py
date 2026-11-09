"""Integer register graphs that can be (a) evaluated exactly in Python and (b) emitted as CSS.

Every register is a CSS custom property registered as <integer>.  A register is either

  lin    round_half_up((bias + sum(coef * src)) / 2**shift), optionally clamped at 0 (ReLU)
  noise  an Irwin-Hall(4) pseudo-Gaussian built from mod() hashes of the three base-251 seed digits

Rounding: CSS rounds a calc() in <integer> context to the nearest integer, but engines disagree on
exact .5 ties.  We never produce one: the divisor is a power of two (exact in binary floating point)
and we add 0.5 to the integer numerator, so the quotient is never k + 0.5.  Python mirrors that with
(s + R/2) // R.
"""

SEED_BASE = 251
SEED_MAX = SEED_BASE ** 3 - 1          # 15,813,250 < 2**24, so the seed itself is float32-exact
NOISE_DRAWS = 4
NOISE_MUL, NOISE_SHIFT = 113, 6        # (sum - 500) * 113 / 64  ->  std 255.9  (x is stored * 256)
F32_SAFE = 1 << 23                     # keep |numerators| below this so float32 engines stay exact


def noise_consts(n=768):
    """The fixed hash constants (c, k1, k2, k3) for every draw of every noise register."""
    import random
    rnd = random.Random(20261108)
    return [[(rnd.randrange(SEED_BASE), rnd.randrange(1, SEED_BASE), rnd.randrange(1, SEED_BASE),
              rnd.randrange(1, SEED_BASE)) for _ in range(NOISE_DRAWS)] for _ in range(n)]


def noise_field(seeds):
    """Vectorised x_T for many seeds -> int array (len(seeds), 768), exactly what the CSS computes."""
    import numpy as np
    seeds = np.asarray(seeds, dtype=np.int64)
    s1 = seeds % SEED_BASE
    sq = (seeds - s1) // SEED_BASE
    s2 = sq % SEED_BASE
    s3 = (sq - s2) // SEED_BASE
    k = np.array(noise_consts(), dtype=np.int64)                      # 768, draws, 4
    h = (k[None, :, :, 0] + k[None, :, :, 1] * s1[:, None, None] + k[None, :, :, 2] * s2[:, None, None]
         + k[None, :, :, 3] * s3[:, None, None]) % SEED_BASE
    tot = (h.sum(-1) - 125 * NOISE_DRAWS) * NOISE_MUL
    r = 1 << NOISE_SHIFT
    return (tot + r // 2) // r


def rshift_round(s, shift):
    if shift == 0:
        return s
    r = 1 << shift
    return (s + r // 2) // r


class Graph:
    def __init__(self):
        self.order = []
        self.defs = {}

    def _add(self, name, d):
        assert name not in self.defs, name
        self.defs[name] = d
        self.order.append(name)

    def lin(self, name, terms, bias=0, shift=0, relu=False):
        terms = [(int(c), s) for c, s in terms if int(c) != 0]
        self._add(name, ('lin', terms, int(bias), int(shift), bool(relu)))

    def noise(self, name, draws):
        assert len(draws) == NOISE_DRAWS
        self._add(name, ('noise', [tuple(int(v) for v in d) for d in draws]))

    # ---------------------------------------------------------------- exact evaluation
    def evaluate(self, seed, stats=None):
        assert 0 <= seed <= SEED_MAX
        s1 = seed % SEED_BASE
        sq = (seed - s1) // SEED_BASE
        s2 = sq % SEED_BASE
        s3 = (sq - s2) // SEED_BASE
        env = {'seed': seed, 's1': s1, 'sq': sq, 's2': s2, 's3': s3}
        for name in self.order:
            d = self.defs[name]
            if d[0] == 'noise':
                tot = sum((c + k1 * s1 + k2 * s2 + k3 * s3) % SEED_BASE for c, k1, k2, k3 in d[1])
                env[name] = rshift_round((tot - 125 * NOISE_DRAWS) * NOISE_MUL, NOISE_SHIFT)
            else:
                _, terms, bias, shift, relu = d
                s = bias
                mag = abs(bias) + 1
                for c, src in terms:
                    p = c * env[src]
                    s += p
                    mag += abs(p)
                if stats is not None and mag > stats.get(name, 0):
                    stats[name] = mag          # worst sum of |terms|: bounds every partial sum
                v = rshift_round(s, shift)
                env[name] = max(0, v) if relu else v
        return env

    # ---------------------------------------------------------------- CSS
    def css_decl(self, name):
        d = self.defs[name]
        if d[0] == 'noise':
            hs = ' + '.join(
                f'mod({c} + {k1}*var(--s1) + {k2}*var(--s2) + {k3}*var(--s3),{SEED_BASE})'
                for c, k1, k2, k3 in d[1])
            return (f'--{name}:calc((({hs} - {125 * NOISE_DRAWS})*{NOISE_MUL} + .5)'
                    f'/{1 << NOISE_SHIFT})')
        _, terms, bias, shift, relu = d
        if shift:
            b = bias + 0.5
            lead = repr(b)
            if lead.endswith('.5') and lead[:-2] in ('0', '-0'):
                lead = lead.replace('0.5', '.5')
        else:
            lead = str(bias)
        parts = [lead]
        for c, src in terms:
            parts.append(' + ' if c > 0 else ' - ')
            if abs(c) != 1:
                parts.append(f'{abs(c)}*')
            parts.append(f'var(--{src})')
        expr = ''.join(parts)
        if shift:
            expr = f'({expr})/{1 << shift}'
        if relu:
            return f'--{name}:max(0,{expr})'
        return f'--{name}:calc({expr})'

    def n_terms(self, names=None):
        tot = 0
        for n in (names or self.order):
            d = self.defs[n]
            tot += len(d[1]) if d[0] == 'lin' else NOISE_DRAWS * 3
        return tot


SEED_DECLS = (
    f'--s1:mod(var(--seed),{SEED_BASE});'
    f'--sq:calc((var(--seed) - var(--s1))/{SEED_BASE});'
    f'--s2:mod(var(--sq),{SEED_BASE});'
    f'--s3:calc((var(--sq) - var(--s2))/{SEED_BASE})'
)
SEED_REGS = ['seed', 's1', 'sq', 's2', 's3']


def at_property(name, inherits=True):
    return (f'@property --{name}{{syntax:"<integer>";inherits:{"true" if inherits else "false"};'
            f'initial-value:0}}')


def checksum(vals):
    acc = 0
    for i, v in enumerate(vals):
        acc = (acc + (i + 1) * (v + 100000)) % 1000000007
    return acc
