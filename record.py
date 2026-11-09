"""Record the page: headless Chrome, real clicks on the real button, back-to-back 2x screenshots.

  python record.py [gens=1] [prefer=cat,heart] [out=demo] [gif=1] [tries=40] [max_res=3.2]
  python record.py url=https://example.org/ check=1 tries=5     only check a hosted copy against the reference

Needs `pip install playwright`, an installed Chrome, and ffmpeg on PATH.  The recorder reads the
frozen seed and the final registers back from the page afterwards (to reject muddled takes and to
check them against the Python reference); it never writes anything into the page.
"""
import asyncio
import base64
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from playwright.async_api import async_playwright

import quant
import sprites

HERE = pathlib.Path(__file__).parent
opts = dict(gens=1, prefer='', out='demo', tries=40, max_res=3.2, gif=1, url='', check=0)
for a in sys.argv[1:]:
    k, v = a.split('=')
    opts[k] = type(opts[k])(v)

ck, model = quant.load(str(HERE / 'model.pt'))
Q = quant.quantize(ck, model)
T = Q['T']
LAST = chr(ord('a') + T - 1)
W, H, SCALE = 560, 900, 2

READ = """() => {
  const cs = getComputedStyle(document.querySelector('.net'));
  const out = [];
  for (let i = 0; i < 768; i++) out.push(parseInt(cs.getPropertyValue('--%s' + i)));
  return { seed: parseInt(cs.getPropertyValue('--seed')), final: out };
}""" % LAST
DONE = ("getComputedStyle(document.querySelector('.stage')).getPropertyValue('--t').trim() === '%d'" % T)


async def take(gens):
    frames = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='chrome', headless=True)
        page = await browser.new_page(viewport={'width': W, 'height': H}, device_scale_factor=SCALE)
        await page.goto(opts['url'] or (HERE / 'index.html').as_uri())
        await page.wait_for_timeout(700)
        client = await page.context.new_cdp_session(page)
        running = True

        async def capture():
            # Screenshots honour the 2x device scale (Chrome's screencast does not), so grab them
            # back to back and keep the time each one was taken.
            loop = asyncio.get_running_loop()
            while running:
                t = loop.time()
                shot = await client.send('Page.captureScreenshot', {
                    'format': 'png', 'optimizeForSpeed': True,
                    'clip': {'x': 0, 'y': 0, 'width': W, 'height': H, 'scale': SCALE}})
                frames.append((t, base64.b64decode(shot['data'])))

        task = asyncio.ensure_future(capture())
        await page.wait_for_timeout(1400)                       # the static, seed spinning
        results = []
        for gi in range(gens):
            await page.click('.btn')                            # freeze the seed, denoise
            await page.wait_for_function(DONE, polling=120, timeout=30000)
            await page.wait_for_timeout(2000)                   # hold on the finished sprite
            results.append(await page.evaluate(READ))
            if gi < gens - 1:
                await page.click('.btn')                        # back to noise
                await page.wait_for_timeout(1100)
        running = False
        await task
        version = browser.version
        await browser.close()
    return frames, results, version


def judge(results):
    seeds = [r['seed'] for r in results]
    ref = quant.int_sample(Q, seeds)
    exact = all(r['final'] == ref[i].tolist() for i, r in enumerate(results))
    res, cls = sprites.fit_residual(ref / 256.0)
    names = [sprites.NAMES[c] for c in cls]
    return seeds, exact, res, names


def encode(frames, out):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='rec_'))
    try:
        lines = []
        for i, (ts, data) in enumerate(frames):
            (tmp / f'f{i:05d}.png').write_bytes(data)
            dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 0.1
            lines += [f"file 'f{i:05d}.png'", f'duration {max(dur, 0.001):.4f}']
        (tmp / 'list.txt').write_text('\n'.join(lines))
        src = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(tmp / 'list.txt')]
        subprocess.run(src + ['-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-crf', '17', '-preset',
                              'slow', '-movflags', '+faststart', str(HERE / f'{out}.mp4')], check=True)
        if opts['gif']:
            # one shared palette, no dithering: the sprite and UI colours recur in every frame so they
            # get exact entries; only the static (700+ random colours) is approximated
            gif = (f'fps=12,scale={W}:-1:flags=area,split[a][b];[a]palettegen=stats_mode=full[p];'
                   '[b][p]paletteuse=dither=none')
            subprocess.run(src + ['-filter_complex', gif, str(HERE / f'{out}.gif')], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def main():
    prefer = [s for s in opts['prefer'].split(',') if s]
    for attempt in range(1, opts['tries'] + 1):
        frames, results, version = await take(opts['gens'])
        seeds, exact, res, names = judge(results)
        span = frames[-1][0] - frames[0][0]
        print(f'take {attempt}: Chrome {version}, {len(frames)} frames over {span:.1f}s, seeds {seeds}, '
              f'{names}, residual {np.round(res, 1).tolist()}, matches reference: {exact}', flush=True)
        if opts['check']:                     # only compare with the reference; write nothing
            continue
        ok = exact and res.max() <= opts['max_res'] and len(set(names)) == len(names)
        if prefer:
            ok = ok and names[0] in prefer
        if ok:
            encode(frames, opts['out'])
            for ext in ('mp4', 'gif') if opts['gif'] else ('mp4',):
                f = HERE / f"{opts['out']}.{ext}"
                print(f'wrote {f.name}: {f.stat().st_size / 1e6:.2f} MB')
            return
    if not opts['check']:
        print('no clean take; nothing written')


asyncio.run(main())
