"""Turn the trained float networks into the integer program the stylesheet runs.

Fixed point everywhere, every divisor a power of two:
    X   pixel registers            x * 256
    H1  in-layer activations       relu((Wi . X + Bi) >> r1)
    H2  mid activations            relu((W1 . H1 + B1) >> r2)
    H3  out-layer activations      relu((W2 . H2 + B2) >> r3)
    X'  next sampler state         (A * X + V . H3 + B) >> r4     <- output layer and DDIM update merged
">>" means divide and round to nearest (see cssgen.rshift_round).
"""
import math

import numpy as np
import torch

from cssgen import F32_SAFE, Graph, noise_consts, noise_field
from model import Denoisers, ddim_coefs, sample_float

SX_BITS = 8


def load(path):
    ck = torch.load(path, weights_only=False)
    c = ck['cfg']
    model = Denoisers(c['T'] - 1, c['g_in'], c['mid'], c['g_out'])
    model.load_state_dict(ck['state'])
    model.eval()
    return ck, model


def _pow2_scale(maxabs, cap):
    """largest b with maxabs * 2**b <= cap"""
    return int(math.floor(math.log2(cap / max(maxabs, 1e-9))))


@torch.no_grad()
def calibrate(ck, model, n=2000, seed=99):
    """Largest activation seen in each layer of each network along real sampler trajectories."""
    alpha, sigma, mean = ck['alpha'], ck['sigma'], ck['mean']
    T = len(alpha) - 1
    seeds = np.random.default_rng(seed).integers(0, 251 ** 3, n)
    x = torch.from_numpy(noise_field(seeds) / 256.0).float()
    k1, k2 = ddim_coefs(alpha, sigma)
    acts = [None]
    x = float(k1[0]) * x + float(k2[0]) * mean
    for j in range(1, T):
        l = j - 1
        xc = x.view(-1, 256, 3).permute(0, 2, 1)
        h1 = torch.relu(torch.einsum('bcp,cgp->bcg', xc, model.w_in[l]).reshape(n, -1) + model.b_in[l])
        h2 = torch.relu(h1 @ model.w_m1[l] + model.b_m1[l])
        h3 = torch.relu(h2 @ model.w_m2[l] + model.b_m2[l])
        acts.append((float(h1.max()), float(h2.max()), float(h3.max())))
        o = torch.einsum('bcg,cgp->bcp', h3.view(n, 3, -1), model.w_out[l]) + model.b_out[l]
        x = float(k1[j]) * x + float(k2[j]) * o.permute(0, 2, 1).reshape(n, 768)
    return acts


def quantize(ck, model, cap_in=127, cap_mid=255, cap_out=255, act_target=2000, r4_max=12,
             prune_in=0, prune_out=0):
    alpha, sigma = ck['alpha'], ck['sigma']
    T = len(alpha) - 1
    k1, k2 = ddim_coefs(alpha, sigma)
    acts = calibrate(ck, model)
    sx = 1 << SX_BITS
    mean = ck['mean'].numpy().astype(np.float64)
    levels = []

    # transition 0: pure noise -> level 1.  x0_hat is just the dataset mean.
    r4 = 10
    levels.append(dict(r4=r4, A=int(round(k1[0] * (1 << r4))),
                       B=np.round(k2[0] * sx * mean * (1 << r4)).astype(np.int64)))
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    for j in range(1, T):
        l = j - 1
        h1m, h2m, h3m = acts[j]
        w_in, w1, w2, w_out = sd['w_in'][l], sd['w_m1'][l], sd['w_m2'][l], sd['w_out'][l]

        sw1 = _pow2_scale(np.abs(w_in).max(), cap_in)
        s1 = _pow2_scale(h1m, act_target)                    # H1 = h1 * 2**s1
        r1 = SX_BITS + sw1 - s1
        sw2 = _pow2_scale(np.abs(w1).max(), cap_mid)
        s2 = _pow2_scale(h2m, act_target)
        r2 = s1 + sw2 - s2
        sw3 = _pow2_scale(np.abs(w2).max(), cap_mid)
        s3 = _pow2_scale(h3m, act_target)
        r3 = s2 + sw3 - s3
        # V = k2 * 256 * w_out * 2**r4 / 2**s3
        r4 = min(r4_max, _pow2_scale(abs(k2[j]) * np.abs(w_out).max(), cap_out) - SX_BITS + s3)
        assert min(r1, r2, r3, r4) >= 1, (j, r1, r2, r3, r4)

        Wi = np.round(w_in * 2.0 ** sw1).astype(np.int64)                         # 3, G, 256
        V = np.round(k2[j] * w_out * 2.0 ** (SX_BITS + r4 - s3)).astype(np.int64)  # 3, G, 256
        if prune_in:
            Wi[np.abs(Wi) <= prune_in] = 0
        if prune_out:
            V[np.abs(V) <= prune_out] = 0
        levels.append(dict(
            r1=r1, r2=r2, r3=r3, r4=r4, Wi=Wi,
            Bi=np.round(sd['b_in'][l] * 2.0 ** (SX_BITS + sw1)).astype(np.int64),
            W1=np.round(w1 * 2.0 ** sw2).astype(np.int64),
            B1=np.round(sd['b_m1'][l] * 2.0 ** (s1 + sw2)).astype(np.int64),
            W2=np.round(w2 * 2.0 ** sw3).astype(np.int64),
            B2=np.round(sd['b_m2'][l] * 2.0 ** (s2 + sw3)).astype(np.int64),
            V=V, A=int(round(k1[j] * 2.0 ** r4)),
            B=np.round(k2[j] * sx * sd['b_out'][l].T.reshape(768) * 2.0 ** r4).astype(np.int64),
            scales=(sw1, s1, sw2, s2, sw3, s3)))
    return dict(T=T, levels=levels)


def _rs(s, r):
    return (s + (1 << (r - 1))) >> r


def int_sample(Q, seeds, keep=False, stats=None):
    """Vectorised integer sampler.  Bit-for-bit the same arithmetic as the CSS."""
    X = noise_field(seeds).astype(np.int64)
    n = X.shape[0]
    traj = [X.copy()]

    def note(key, mag):
        if stats is not None:
            stats[key] = max(stats.get(key, 0), int(mag.max()))

    for s, lv in enumerate(Q['levels']):
        if s == 0:
            S = lv['A'] * X + lv['B'][None]
            note((s, 'out'), np.abs(lv['A'] * X) + np.abs(lv['B'][None]))
        else:
            Xc = X.reshape(n, 256, 3)
            S1 = np.einsum('npc,cgp->ncg', Xc, lv['Wi']).reshape(n, -1) + lv['Bi'][None]
            note((s, 'in'), np.einsum('npc,cgp->ncg', np.abs(Xc), np.abs(lv['Wi'])).reshape(n, -1)
                 + np.abs(lv['Bi'][None]))
            H1 = np.maximum(0, _rs(S1, lv['r1']))
            S2 = H1 @ lv['W1'] + lv['B1'][None]
            note((s, 'm1'), H1 @ np.abs(lv['W1']) + np.abs(lv['B1'][None]))
            H2 = np.maximum(0, _rs(S2, lv['r2']))
            S3 = H2 @ lv['W2'] + lv['B2'][None]
            note((s, 'm2'), H2 @ np.abs(lv['W2']) + np.abs(lv['B2'][None]))
            H3 = np.maximum(0, _rs(S3, lv['r3']))
            g = lv['V'].shape[1]
            H3c = H3.reshape(n, 3, g)
            O = np.einsum('ncg,cgp->npc', H3c, lv['V']).reshape(n, 768)
            S = lv['A'] * X + O + lv['B'][None]
            note((s, 'out'), np.abs(lv['A'] * X) + np.einsum('ncg,cgp->npc', H3c, np.abs(lv['V'])
                                                              ).reshape(n, 768) + np.abs(lv['B'][None]))
            note((s, 'h'), np.maximum(H1.max(), np.maximum(H2.max(), H3.max())) * np.ones(1))
        X = _rs(S, lv['r4'])
        if keep:
            traj.append(X.copy())
    return (X, traj) if keep else X


def build_graph(Q):
    g = Graph()
    consts = noise_consts()
    noise = [f'z{i}' for i in range(768)]
    for i, n in enumerate(noise):
        g.noise(n, consts[i])
    steps, prev = [], 'z'
    for s, lv in enumerate(Q['levels']):
        L = chr(ord('a') + s)
        U = L.upper()
        if s > 0:
            G = lv['Wi'].shape[1]
            for c in range(3):
                for k in range(G):
                    w = lv['Wi'][c, k]
                    g.lin(f'{U}a{c * G + k}', [(int(w[p]), f'{prev}{p * 3 + c}') for p in range(256)],
                          bias=int(lv['Bi'][c * G + k]), shift=lv['r1'], relu=True)
            n1, n2 = lv['W1'].shape
            for o in range(n2):
                g.lin(f'{U}b{o}', [(int(lv['W1'][i, o]), f'{U}a{i}') for i in range(n1)],
                      bias=int(lv['B1'][o]), shift=lv['r2'], relu=True)
            n2, n3 = lv['W2'].shape
            for o in range(n3):
                g.lin(f'{U}c{o}', [(int(lv['W2'][i, o]), f'{U}b{i}') for i in range(n2)],
                      bias=int(lv['B2'][o]), shift=lv['r3'], relu=True)
        names = []
        for i in range(768):
            p, c = divmod(i, 3)
            terms = [(lv['A'], f'{prev}{i}')]
            if s > 0:
                Go = lv['V'].shape[1]
                terms += [(int(lv['V'][c, k, p]), f'{U}c{c * Go + k}') for k in range(Go)]
            g.lin(f'{L}{i}', terms, bias=int(lv['B'][i]), shift=lv['r4'])
            names.append(f'{L}{i}')
        steps.append(names)
        prev = L
    return g, noise, steps


def count_weights(Q):
    """(nonzero weights and biases, total before dropping zeros)"""
    nz = tot = 0
    for s, lv in enumerate(Q['levels']):
        arrs = [lv['B']] + ([] if s == 0 else [lv[k] for k in ('Wi', 'Bi', 'W1', 'B1', 'W2', 'B2', 'V')])
        for a in arrs:
            nz += int(np.count_nonzero(a))
            tot += a.size
    return nz, tot
