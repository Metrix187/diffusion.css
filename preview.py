"""Link-preview card (preview.png, 1200x630) built from the integer sampler's own output."""
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cssgen
import quant
import sprites

ck, model = quant.load('model.pt')
Q = quant.quantize(ck, model)
seeds = np.random.default_rng(5).integers(0, cssgen.SEED_MAX + 1, 600)
final, traj = quant.int_sample(Q, seeds, keep=True)
res, cls = sprites.fit_residual(final / 256.0)
best = [int(np.where(cls == k, res, np.inf).argmin()) for k in range(len(sprites.NAMES))]
star = best[sprites.NAMES.index('cat')]


def font(bold, size):
    for name in (('consolab.ttf', 'DejaVuSansMono-Bold.ttf') if bold else ('consola.ttf', 'DejaVuSansMono.ttf')):
        for base in ('C:/Windows/Fonts/', '/usr/share/fonts/truetype/dejavu/'):
            if pathlib.Path(base + name).exists():
                return ImageFont.truetype(base + name, size)
    return ImageFont.load_default()


im = Image.new('RGB', (1200, 630), (10, 10, 17))
d = ImageDraw.Draw(im)
d.text((48, 40), 'diffusion.css', font=font(True, 28), fill=(124, 243, 192))
d.text((48, 84), 'A diffusion model that', font=font(True, 58), fill=(243, 244, 248))
d.text((48, 150), 'runs entirely in CSS.', font=font(True, 58), fill=(243, 244, 248))
nz, _ = quant.count_weights(Q)
d.text((48, 232), f'No JavaScript. {nz:,} weights, each a number in a calc().', font=font(False, 26),
       fill=(144, 147, 168))
x = 48
for i in best:                                         # one generated sprite of each kind
    im.paste(Image.fromarray(sprites.to_image(final[i] / 256.0, 6)), (x, 300))
    x += 96 + 15
x = 48
for t in traj:                                         # one full denoising run
    im.paste(Image.fromarray(sprites.to_image(t[star] / 256.0, 5)), (x, 470))
    x += 80 + 5
d.text((48, 566), 'noise to sprite in 12 steps, computed by the stylesheet', font=font(False, 22),
       fill=(111, 114, 137))
im.save('preview.png')
print('preview.png', 'seeds', [int(seeds[i]) for i in best], 'residuals', np.round(res[best], 1).tolist())
