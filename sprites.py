"""The training set: ten hand-drawn 16x16 sprites, recoloured to a random hue for every sample.

Legend:  . background   # outline   B body   S shade   H highlight   W white   K dark   P pink
Shade and highlight are affine in the body colour, so each sprite spans a small linear family.
"""
import colorsys

import numpy as np

SPRITES = {
    'heart': '''
................
................
..####....####..
.#BBBB#..#BBBB#.
#BBHHBB##BBBBBB#
#BHHBBBBBBBBBBB#
#BHBBBBBBBBBBBS#
#BBBBBBBBBBBBBS#
.#BBBBBBBBBBBS#.
..#BBBBBBBBBS#..
...#BBBBBBBS#...
....#BBBBBS#....
.....#BBBS#.....
......#BS#......
.......##.......
................''',
    'cat': '''
................
..##........##..
..#B#......#B#..
..#PB#....#BP#..
..#BBB####BBB#..
.#BBBBBBBBBBBB#.
.#BBBBBBBBBBBB#.
.#BBWKBBBBWKBB#.
.#BBWKBBBBWKBB#.
.#BBBBBPPBBBBB#.
.#SBBB#BB#BBBS#.
..#SBBB##BBBS#..
...#SSBBBBSS#...
....########....
................
................''',
    'ghost': '''
................
.....######.....
...##BBBBBB##...
..#BBBBBBBBBB#..
..#BHBBBBBBBB#..
.#BBHBBBBBBBBB#.
.#BBWWBBBBWWBB#.
.#BBWKBBBBWKBB#.
.#BBWKBBBBWKBB#.
.#BBBBBBBBBBBB#.
.#BBBBB##BBBBB#.
.#BBBBBBBBBBBS#.
.#BBBBBBBBBBBS#.
.#BBB#BBBB#BSS#.
..###.####.###..
................''',
    'mushroom': '''
................
.....######.....
...##BBBBBB##...
..#BBWWBBBBBB#..
.#BBBWWBBBWWBB#.
.#BBBBBBBBWWBB#.
#BBWWBBBBBBBBBB#
#BBWWBBBBBBBBBS#
#SBBBBBBBBBBSSS#
.##############.
...#WWWWWWWW#...
...#WKWWWWKW#...
...#WKWWWWKW#...
...#WWWWWWWW#...
....########....
................''',
    'slime': '''
................
................
................
......####......
....##BBBB##....
...#BBHBBBBB#...
..#BBHBBBBBBB#..
..#BBBBBBBBBB#..
.#BBBBBBBBBBBB#.
.#BBWKBBBBWKBB#.
.#BBWKBBBBWKBB#.
.#BBBBBBBBBBBB#.
.#SBBB####BBBS#.
.#SSBBBBBBBBSS#.
..############..
................''',
    'star': '''
................
.......##.......
......#BB#......
......#BB#......
.....#BHBB#.....
.#####BHBB#####.
.#BBBBHBBBBBBB#.
..#BBBBBBBBBB#..
...#BBKBBKBB#...
....#BKBBKB#....
....#BBBBBB#....
...#BBBBBBBB#...
...#BBB##BBB#...
..#BB##..##BB#..
..###......###..
................''',
    'skull': '''
................
....########....
..##BBBBBBBB##..
.#BBHBBBBBBBBB#.
.#BHBBBBBBBBBB#.
.#BBKKKBBKKKBB#.
.#BBKKKBBKKKBB#.
.#BBKKKBBKKKBB#.
.#BBBBBKKBBBBB#.
..#SBBBKKBBBS#..
...##BBBBBB##...
....#B#BB#B#....
....#B#BB#B#....
....########....
................
................''',
    'potion': '''
................
.....######.....
.....#SSSS#.....
......#HH#......
......#HH#......
.....#HHHH#.....
....#HHHHHH#....
...#HHWHHHHH#...
..#BBBBBBBBBB#..
..#BWBBBBBBBB#..
.#BBBBBBBWBBBB#.
.#BBBBBBBBBBBS#.
.#BBBBBBBBBBSS#.
..#BBBBBBBBSS#..
...##########...
................''',
    'invader': '''
................
................
................
................
....B.....B.....
.....B...B......
....BBBBBBB.....
...BB.BBB.BB....
..BBBBBBBBBBB...
..B.BBBBBBB.B...
..B.B.....B.B...
.....BB.BB......
................
................
................
................''',
    'gem': '''
................
................
....########....
...#HHBBBBBS#...
..#HHBBBBBBSS#..
.#HBBBBBBBBBSS#.
.#SSSSSSSSSSSS#.
.#SBBBBBBBBBBS#.
..#SBBBBBBBBS#..
...#SBBBBBBS#...
....#SBBBBS#....
.....#SBBS#.....
......#SS#......
.......##.......
................
................''',
}

NAMES = list(SPRITES)
BG = np.array([0.086, 0.094, 0.157])
OUTLINE = np.array([0.035, 0.035, 0.07])
WHITE = np.array([0.97, 0.96, 0.92])
DARK = np.array([0.07, 0.07, 0.12])
PINK = np.array([1.0, 0.56, 0.69])
SHADE_MUL = 0.60
HI_MUL, HI_ADD = 0.42, 0.58


def _grid(name):
    rows = SPRITES[name].strip('\n').split('\n')
    assert len(rows) == 16 and all(len(r) == 16 for r in rows), (name, [len(r) for r in rows])
    return rows


GRIDS = {n: _grid(n) for n in NAMES}


def _masks(name):
    g = GRIDS[name]
    return {ch: np.array([[c == ch for c in row] for row in g], dtype=np.float64) for ch in '.#BSHWKP'}


MASKS = {n: _masks(n) for n in NAMES}
# per sprite: image = FIXED + BODYW[..., None] * body_rgb   (all in 0..1)
FIXED = {}
BODYW = {}
for _n in NAMES:
    m = MASKS[_n]
    FIXED[_n] = (m['.'][..., None] * BG + m['#'][..., None] * OUTLINE + m['W'][..., None] * WHITE
                 + m['K'][..., None] * DARK + m['P'][..., None] * PINK + m['H'][..., None] * HI_ADD)
    BODYW[_n] = m['B'] + SHADE_MUL * m['S'] + HI_MUL * m['H']


def body_colour(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v))


def render(name, h, s=0.72, v=0.96):
    """-> (16,16,3) float image in 0..1"""
    return FIXED[name] + BODYW[name][..., None] * body_colour(h, s, v)


def sample_batch(n, rng):
    """-> (n, 768) float32 in -1..1, plus the sprite index of each row"""
    idx = rng.integers(0, len(NAMES), n)
    hs = rng.random(n)
    ss = rng.uniform(0.60, 0.85, n)
    vs = rng.uniform(0.90, 1.00, n)
    out = np.empty((n, 16, 16, 3), dtype=np.float32)
    for i in range(n):
        out[i] = render(NAMES[idx[i]], hs[i], ss[i], vs[i])
    return out.reshape(n, 768) * 2.0 - 1.0, idx


def fit_residual(x):
    """How far is each image from the nearest "some sprite in some colour"?

    x: (n, 768) in -1..1.  Returns (rms residual in 8-bit levels, best sprite index)."""
    img = (np.asarray(x, dtype=np.float64).reshape(-1, 16, 16, 3) + 1.0) / 2.0
    best = np.full(img.shape[0], np.inf)
    arg = np.zeros(img.shape[0], dtype=np.int64)
    for k, n in enumerate(NAMES):
        w = BODYW[n][None, ..., None]
        d = img - FIXED[n][None]
        c = (d * w).sum(axis=(1, 2), keepdims=True) / (w * w).sum()
        c = np.clip(c, 0.0, 1.0)
        r = np.sqrt(((d - w * c) ** 2).mean(axis=(1, 2, 3))) * 255.0
        better = r < best
        best[better], arg[better] = r[better], k
    return best, arg


def to_image(x, scale=8):
    """x: (..., 768) in -1..1 -> uint8 array (16*scale, 16*scale, 3)"""
    a = np.clip((np.asarray(x, dtype=np.float64).reshape(16, 16, 3) + 1.0) * 127.5, 0, 255)
    return np.repeat(np.repeat(a.round().astype(np.uint8), scale, 0), scale, 1)


def sheet(xs, cols, scale=6, pad=4, bg=(11, 11, 16)):
    from PIL import Image
    xs = list(xs)
    rows = (len(xs) + cols - 1) // cols
    cell = 16 * scale + pad
    im = Image.new('RGB', (cols * cell + pad, rows * cell + pad), bg)
    for k, x in enumerate(xs):
        im.paste(Image.fromarray(to_image(x, scale)), (pad + (k % cols) * cell, pad + (k // cols) * cell))
    return im


if __name__ == '__main__':
    xs = []
    for row, hue0 in enumerate([0.0, 0.33, 0.58, 0.83]):
        for k, n in enumerate(NAMES):
            xs.append(render(n, hue0 + 0.07 * k).reshape(768) * 2 - 1)
    sheet(xs, len(NAMES), scale=8).save('sprites.png')
    print('sprites', len(NAMES), 'saved sprites.png')
