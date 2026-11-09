"""Assemble the single-file page: registers, the gated network rule, the keyframe timeline, the UI."""
import math

from cssgen import SEED_DECLS, SEED_MAX, SEED_REGS, at_property

N_PIX = 256
N_REG = 768          # 16 x 16 x RGB

UI_CSS = '''
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0a0a11;color:#b9bccb;
font:15px/1.6 ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace}
main{max-width:620px;margin:0 auto;padding:44px 18px 72px}
h1{margin:0 0 10px;font-size:14px;font-weight:600;letter-spacing:.04em;color:#7cf3c0}
.lede{margin:0 0 12px;font-size:27px;line-height:1.2;font-weight:700;color:#f3f4f8;text-wrap:balance}
.sub{margin:0 0 30px;color:#9093a8;text-wrap:pretty}
.app,.net,.stage{display:flex;flex-direction:column;align-items:center;width:100%}
#go{position:absolute;width:1px;height:1px;opacity:0}
.hud,.bar{width:384px;max-width:100%}
.strip{display:flex;flex-wrap:wrap;justify-content:center;gap:6px;width:100%;margin-top:26px;
visibility:hidden}
.app:has(#go:checked) .strip{visibility:visible}
.strip u{position:relative;width:32px;height:32px;overflow:hidden;background:#161828;border-radius:3px;
transition:opacity .25s}
.strip u::before{content:"";position:absolute;left:-2px;top:-2px;width:2px;height:2px}
.bezel{padding:14px;border-radius:16px;background:#11121b;border:1px solid #25273a;
box-shadow:0 30px 80px -30px #000}
.screen{display:grid;grid-template-columns:repeat(16,24px);grid-auto-rows:24px;background:#161828}
.screen i{display:block}
.hud{display:flex;justify-content:space-between;margin:14px 0 8px;font-size:13px;color:#8387a0;
font-variant-numeric:tabular-nums}
.seed::after{counter-reset:s var(--seed);content:"seed " counter(s)}
.step::after{counter-reset:t var(--t);content:"step " counter(t) " / __T__"}
.bar{height:4px;border-radius:2px;background:#1b1d2c;overflow:hidden}
.bar b{display:block;height:100%;width:calc(var(--t)*100%/__T__);background:#7cf3c0}
.row{display:flex;gap:10px;margin-top:22px}
.btn,.replay{padding:12px 26px;border-radius:10px;border:1px solid #7cf3c0;font:inherit;font-weight:700;
font-size:15px;cursor:pointer;user-select:none;-webkit-tap-highlight-color:transparent}
.btn{color:#0a0a11;background:#7cf3c0}
.replay{display:none;padding-inline:18px;background:transparent;color:#8387a0;border-color:#2c2f45}
.replay:hover{color:#f3f4f8;border-color:#8387a0}
.app:has(#go:checked) .replay{display:block}
.btn:hover{background:#a3f8d6}
.btn::after{content:"\\25B6\\00A0 denoise"}
.app:has(#go:checked) .btn{background:transparent;color:#7cf3c0}
.app:has(#go:checked) .btn::after{content:"\\21BB\\00A0 new noise"}
.app:has(#go:focus-visible) .btn{outline:2px solid #f3f4f8;outline-offset:3px}
.note{margin:12px 0 0;font-size:13px;color:#6f7289;text-align:center}
.note::after{content:"the seed is spinning. press to freeze it."}
.app:has(#go:checked) .note::after{content:"same seed, same sprite. press again for new noise."}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:44px 0 0;padding:0;list-style:none}
.stats li{padding:12px 8px;border:1px solid #1f2132;border-radius:10px;text-align:center;font-size:12px;
color:#8387a0}
.stats b{display:block;font-size:18px;color:#f3f4f8}
h2{margin:44px 0 12px;font-size:14px;font-weight:600;letter-spacing:.04em;color:#7cf3c0}
ol{margin:0;padding-left:22px}
ol li{margin:0 0 12px;padding-left:4px;text-wrap:pretty}
ol b{color:#f3f4f8;font-weight:600}
code{color:#d7c9ff}
.foot{margin:40px 0 0;font-size:13px}
.foot a{color:#7cf3c0}
.warn{display:none;margin:0 0 24px;padding:12px 14px;border:1px solid #6b4a1e;border-radius:10px;color:#f0c27a}
@supports not selector(:has(*)){.warn{display:block}}
@media (max-width:470px){
.lede{font-size:22px}
.screen{grid-template-columns:repeat(16,18px);grid-auto-rows:18px}
.hud,.bar{width:288px}
.stats{grid-template-columns:repeat(2,1fr)}
}
'''


def build_page(g, noise, steps, info=None, seconds=6.5, hold=0.55, site=None, repo=None):
    """noise: 768 register names holding x_T.  steps: list of 768-name lists, one per denoising step.
    info: dict of strings for the explainer (weights, registers, size, networks)."""
    T = len(steps)
    info = dict(dict(weights='?', registers='?', size='?', networks=str(T - 1)), **(info or {}))
    noise_set = set(noise)
    gated = [n for n in g.order if n not in noise_set]
    view = [f'v{i}' for i in range(N_REG)]

    out = []
    w = out.append
    w('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
      '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
      '<title>diffusion.css</title>\n<link rel="icon" href="data:,">\n'
      '<meta name="description" content="A denoising diffusion model that runs entirely in CSS. '
      'No JavaScript.">\n')
    if site:      # link-preview metadata only; browsers do not fetch any of it
        w('<meta property="og:type" content="website">\n<meta property="og:title" content="diffusion.css">\n'
          '<meta property="og:description" content="A denoising diffusion model that runs entirely in '
          f'CSS. No JavaScript.">\n<meta property="og:url" content="{site}">\n'
          f'<meta property="og:image" content="{site}preview.png">\n'
          '<meta name="twitter:card" content="summary_large_image">\n')
    w(f'''<!--
  diffusion.css: a denoising diffusion model in a stylesheet.  This page contains no script element.

  Reading the CSS below:
    --seed            a counter that spins until the checkbox pauses it; --s1 --s2 --s3 are its base-251 digits
    --z0 .. --z767    x_T, the starting noise.  Register i is pixel floor(i/3), channel i%3 (r,g,b).
    --a0 .. --a767    the image after sampler step 1, --b* after step 2, and so on up to --{chr(96 + T)}*
    --Ba* --Bb* --Bc* the three hidden layers of the network used by step 2 (step 1 needs no network)
    --v0 .. --v767    what the screen is showing; the "denoise" keyframes point these at each step in turn
  Pixel values are stored times 256 (so -256 is black, 256 is full brightness).
  Every division is by a power of two with .5 added to the numerator, so no result is ever an exact
  tie and every browser rounds it the same way.  Generated by export.py.
-->
''')
    w('<style>\n')
    w('/* ---- registers: every number in the model is an <integer> custom property ---- */\n')
    w('\n'.join(at_property(n) for n in SEED_REGS + ['t'] + list(noise) + gated + view))
    w('\n\n')

    w('/* ---- seed: a counter that spins until the checkbox freezes it ---- */\n')
    w(f'@keyframes spin{{to{{--seed:{SEED_MAX}}}}}\n')
    w('.app{animation:spin 211s linear infinite}\n')
    w('.app:has(#go:checked){animation-play-state:paused}\n\n')

    w('/* ---- x_T: roughly Gaussian noise hashed from the seed, one register per pixel channel ---- */\n')
    w('.net{' + SEED_DECLS + ';\n')
    w(';\n'.join(g.css_decl(n) for n in noise))
    w('}\n\n')

    w('/* ---- the denoiser, unrolled over every sampler step: evaluated once the seed is frozen ---- */\n')
    w('.app:has(#go:checked) .net{\n')
    w(';\n'.join(g.css_decl(n) for n in gated))
    w('}\n\n')

    w('/* ---- the screen shows x_T, then the keyframes walk it through every sampler state ---- */\n')
    w('.stage{' + ';'.join(f'--{v}:var(--{z})' for v, z in zip(view, noise)) + '}\n')
    w(f'.app:has(#go:checked) .stage{{animation:denoise {seconds}s linear .4s both}}\n')
    w('/* holding "replay" drops the animation; letting go starts it again from the noise */\n')
    w('.app:has(#go:checked):has(.replay:active) .stage{animation:none}\n')
    w('@keyframes denoise{\n')
    # The first steps (still static) and the last (final polish) barely change the picture, so the
    # middle steps, where the sprite appears, get the most screen time.
    wts = [0.45 + 0.85 * math.exp(-((k - 0.58 * T) / (0.28 * T)) ** 2) for k in range(T)]
    ends = [sum(wts[:k + 1]) / sum(wts) for k in range(T)]
    for k, names in enumerate(steps, 1):
        decl = f'--t:{k};' + ';'.join(f'--{v}:var(--{n})' for v, n in zip(view, names))
        hi = 100.0 * ends[k - 1]
        lo = hi - 100.0 * hold * wts[k - 1] / sum(wts)
        # each state is reached a little before its mark and then held, so steps read as steps
        w(f'{lo:.2f}%,{hi:.2f}%{{{decl}}}\n')
    w('}\n\n')

    w('/* ---- 256 pixels, each just reads its three registers ---- */\n')
    w('.screen i{background:rgb(calc(127.5 + var(--r)*.498) calc(127.5 + var(--g)*.498) '
      'calc(127.5 + var(--b)*.498))}\n')
    w('\n'.join(
        f'.screen i:nth-child({p + 1}){{--r:var(--v{3 * p});--g:var(--v{3 * p + 1});--b:var(--v{3 * p + 2})}}'
        for p in range(N_PIX)))
    w('\n\n/* ---- filmstrip: every sampler state as a thumbnail, one box-shadow per pixel ---- */\n')

    def shadows(names):
        px = []
        for p in range(N_PIX):
            x, y = (p % 16 + 1) * 2, (p // 16 + 1) * 2
            r, gg, b = (f'calc(var(--{names[3 * p + c]})/2 + 128)' for c in range(3))
            px.append(f'{x}px {y}px rgb({r} {gg} {b})')
        return ','.join(px)

    for k, names in enumerate([list(noise)] + steps):
        w(f'.strip u:nth-child({k + 1})::before{{box-shadow:{shadows(names)}}}\n')
        if k:
            w(f'.strip u:nth-child({k + 1}){{opacity:clamp(0,var(--t) - {k - 1},1)}}\n')
    w('\n/* ---- everything below is ordinary page styling ---- */')
    w(UI_CSS.replace('__T__', str(T)))
    w('</style>\n</head>\n<body>\n<main>\n')
    w('<h1>diffusion.css</h1>\n'
      '<p class="lede">A diffusion model that runs entirely in CSS.</p>\n'
      '<p class="sub">No JavaScript, no WebAssembly, no images. Press the button: the stylesheet freezes '
      f'a seed, turns it into noise, and denoises that into a 16&times;16 sprite in {T} steps.</p>\n'
      '<p class="warn">This browser lacks <code>:has()</code>, which the page needs. '
      'Try a current Chrome, Edge, Firefox or Safari.</p>\n')
    w('<div class="app">\n<input type="checkbox" id="go" aria-label="Freeze the seed and denoise">\n'
      '<div class="net"><div class="stage">\n<div class="bezel"><div class="screen">'
      + '<i></i>' * N_PIX +
      '</div></div>\n<div class="hud"><span class="seed"></span><span class="step"></span></div>\n'
      '<div class="bar"><b></b></div>\n'
      '<div class="row"><label for="go" class="btn"></label>'
      '<button type="button" class="replay">replay</button></div>\n<p class="note"></p>\n'
      '<div class="strip" aria-hidden="true">' + '<u></u>' * (T + 1) + '</div>\n'
      '</div></div>\n</div>\n')
    w(f'''<ul class="stats">
<li><b>{info['weights']}</b>weights</li>
<li><b>{T}</b>sampler steps</li>
<li><b>{info['size']}</b>of CSS</li>
<li><b>0</b>lines of JS</li>
</ul>
<h2>How it works</h2>
<ol>
<li><b>Seed.</b> An <code>@property</code> integer counts up inside a keyframe animation. Ticking the
hidden checkbox pauses it, so the moment you press is the random seed.</li>
<li><b>Noise.</b> Each of the 768 pixel channels hashes that seed with <code>mod()</code> and sums four
draws into roughly Gaussian noise. That is the static on screen before you press.</li>
<li><b>Model.</b> {info['networks']} small neural networks, one per noise level, trained in PyTorch on ten
hand-drawn sprites recoloured to random hues, then pruned and rounded to integers. All
{info['weights']} weights are written out as literals inside <code>calc()</code>.</li>
<li><b>Sampling.</b> {T} DDIM steps, unrolled. Every intermediate value is its own
<code>@property</code> <code>&lt;integer&gt;</code> register, {info['registers']} in all, so the
browser computes each once and rounds it.</li>
<li><b>Timeline.</b> One <code>@keyframes</code> rule points the 256 pixel elements at each sampler
state in turn. Nothing is pre-rendered: a different seed gives a different sprite and colour.</li>
<li><b>Check it.</b> View source. There is no <code>&lt;script&gt;</code>, and a Python reference using
the same integer arithmetic produces identical pixels for the same seed.</li>
</ol>
''')
    if repo:
        w(f'<p class="foot"><a href="{repo}">Source, training code and a reference sampler</a></p>\n')
    w('</main>\n</body>\n</html>\n')
    return ''.join(out)
