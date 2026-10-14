"""Independent checks of the headline results, written WITHOUT importing the xfmr package.

1. Data sanity: yearly Swiss end-user energy and peak.
2. ONAF, 2024, peak = nameplate: re-solve the IEC ODEs with scipy LSODA and compare with results/summary.json.
3. The 1.384 pu headline (and the 6-month-shift case): brute-force scan over the full three years.
4. Daily ratings of 8 random days: brute-force scan instead of bisection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / 'results' / 'summary.json').read_text())
df = pd.read_csv(ROOT / 'data' / 'load_ambient_15min.csv.gz', parse_dates=['time_utc'])

# medium/large ONAF typical values, typed again here on purpose
x, y, R, k11, k21, k22, to, tw, dor, dhr = 0.8, 1.3, 6.0, 0.5, 2.0, 2.0, 150.0, 7.0, 52.0, 26.0


def ss(k, a):
    return (a + dor * ((1 + k * k * R) / (1 + R)) ** x, k21 * dhr * k ** y, (k21 - 1) * dhr * k ** y)


def sim(K, A, s0, dt=15.0):
    o, h1, h2 = s0
    e1, e2, e3 = np.exp(-dt / (k11 * to)), np.exp(-dt / (k22 * tw)), np.exp(-dt / (to / k22))
    O, H = np.empty(len(K)), np.empty(len(K))
    for i, (k, a) in enumerate(zip(K, A)):
        uo = a + dor * ((1 + k * k * R) / (1 + R)) ** x
        o = uo + (o - uo) * e1
        u1, u2 = k21 * dhr * k ** y, (k21 - 1) * dhr * k ** y
        h1 = u1 + (h1 - u1) * e2
        h2 = u2 + (h2 - u2) * e3
        O[i], H[i] = o, o + h1 - h2
    return O, H, (o, h1, h2)


def ageing_days(h):
    return np.exp(15000 / 383 - 15000 / (h + 273)).sum() * 15 / 1440


print('1. data')
for yr in (2023, 2024, 2025):
    d = df[df.time_utc.dt.year == yr]
    print(f'   {yr}: {d.load_mw.sum() * 0.25 / 1e6:.2f} TWh, peak {d.load_mw.max() / 1e3:.2f} GW')

print('2. LSODA re-solve, ONAF 2024, peak = nameplate')
d24 = df[df.time_utc.dt.year == 2024].reset_index(drop=True)
K = d24.load_mw.values / d24.load_mw.values.max()
A = d24.ambient_c.values


def rhs(t, s):
    i = min(int(t // 15), len(K) - 1)
    k, a = K[i], A[i]
    o, h1, h2 = s
    return [(dor * ((1 + k * k * R) / (1 + R)) ** x - (o - a)) / (k11 * to),
            (k21 * dhr * k ** y - h1) / (k22 * tw), ((k21 - 1) * dhr * k ** y - h2) / (to / k22)]


def lsoda(s0, n):
    return solve_ivp(rhs, [0, n * 15], s0, method='LSODA', t_eval=np.arange(1, n + 1) * 15.0,
                     rtol=1e-9, atol=1e-9, max_step=15.0)


warm = lsoda(ss(K[:192].mean(), A[:192].mean()), 192)
sol = lsoda(warm.y[:, -1], len(K))
hs = sol.y[0] + sol.y[1] - sol.y[2]
ref = [r for r in S['annual_peak_equals_nameplate']
       if r['cooling'] == 'Medium/large ONAF' and r['ambient'] == 'real' and r['year'] == 2024][0]
print(f'   independent: hot-spot max {hs.max():.3f} degC, ageing {ageing_days(hs):.3f} d')
print(f'   pipeline   : hot-spot max {ref["hot_spot_max"]:.3f} degC, ageing {ref["loss_of_life_days"]:.3f} d')

print('3. brute-force maximum uniform peak (three years, real ambient)')
Kall = df.load_mw.values / df.load_mw.values.max()
Aall = df.ambient_c.values
for label, KK in [('as measured', Kall), ('shifted 6 months', np.roll(Kall, len(Kall) // 2))]:
    for s in np.arange(1.0, 1.6, 0.001):
        Ks = s * KK
        _, _, st = sim(Ks[:192], Aall[:192], ss(Ks[:192].mean(), Aall[:192].mean()))
        _, H, _ = sim(Ks, Aall, st)
        if H.max() > 120:
            break
        best = s
    ref = [r for r in S['max_uniform_peak'] if r['cooling'] == 'Medium/large ONAF' and
           r['ambient'] == ('real' if label == 'as measured' else 'real, load shifted 6 months')][0]
    print(f'   {label}: brute force {best:.3f} pu, pipeline {ref["max_peak_pu"]:.4f} pu')

print('4. daily rating, 8 random days (brute force step 0.0005 pu)')
d = pd.read_csv(ROOT / 'results' / 'daily_rating.csv', parse_dates=['day'])
df['day'] = df.time_utc.dt.tz_convert('Europe/Zurich').dt.floor('D').dt.tz_localize(None)
for i in np.random.default_rng(7).choice(len(d), 8, replace=False):
    day = d.day.iloc[i]
    idx = np.where(df.day.values == np.datetime64(day))[0]
    seg = slice(idx[0] - 192, idx[-1] + 1)
    L, AA = df.load_mw.values[seg], df.ambient_c.values[seg]
    shape = L / L[192:].max()
    best = 0.0
    for s in np.arange(1.0, 1.6, 0.0005):
        Ks = shape * s
        O, H, _ = sim(Ks, AA, ss(Ks[:96].mean(), AA[:96].mean()))
        h = H[192:]
        if h.max() <= 120 and O[192:].max() <= 105 and s <= 1.5 and ageing_days(h) <= 1:
            best = s
        else:
            break
    print(f'   {day.date()}: pipeline {d.rating_pu.iloc[i]:.4f}, brute force {best:.4f} ({d.binding.iloc[i]})')
