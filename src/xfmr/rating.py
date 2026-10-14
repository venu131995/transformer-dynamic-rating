"""Static and dynamic ratings of a transformer that serves a real load shape under real ambient."""
import numpy as np
import pandas as pd

from .iec60076_7 import LIMITS, initial_state, loss_of_life_days, simulate_exact

DT = 15.0          # min, data resolution
PER_DAY = 96


def load_table(path):
    df = pd.read_csv(path, parse_dates=['time_utc'])
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    # calendar days in Swiss local time (what an operator plans by)
    df['day'] = df['time_utc'].dt.tz_convert('Europe/Zurich').dt.floor('D').dt.tz_localize(None)
    return df


def per_unit(load_mw, peak_pu=1.0):
    """Scale a load series so that its overall maximum equals peak_pu (transformer sized for the peak)."""
    load_mw = np.asarray(load_mw, float)
    return peak_pu * load_mw / load_mw.max()


def simulate_series(p, K, theta_a, warmup_steps=2 * PER_DAY):
    """Run the exact solver on a long series; the first `warmup_steps` are repeated in front to settle."""
    K = np.asarray(K, float)
    theta_a = np.asarray(theta_a, float)
    s0 = initial_state(p, K[:warmup_steps].mean(), theta_a[:warmup_steps].mean())
    warm = simulate_exact(p, K[:warmup_steps], theta_a[:warmup_steps], DT, s0)
    return simulate_exact(p, K, theta_a, DT, warm['state'])


def _daily_blocks(df):
    """Arrays (3 * 96, n_days): previous two days + the day itself, for every day with full data."""
    days = df['day'].values
    uniq, start = np.unique(days, return_index=True)
    counts = np.diff(np.append(start, len(days)))
    ok = np.where(counts == PER_DAY)[0]
    ok = ok[ok >= 2]
    ok = ok[(counts[ok - 1] == PER_DAY) & (counts[ok - 2] == PER_DAY)]
    idx = np.stack([np.arange(start[i - 2], start[i - 2] + 3 * PER_DAY) for i in ok], axis=1)
    return uniq[ok], idx


def daily_dynamic_rating(p, df, limits=LIMITS, ageing_limit_days=1.0, iters=40, hi=3.0):
    """Largest daily peak load (pu of nameplate) that keeps every limit on that day.

    The day's own load shape is scaled so that its peak equals the candidate value; the two preceding
    days carry the same scaling so the oil starts from a consistent thermal state. Bisection is done
    for all days in parallel. Returns a DataFrame with the rating and which limit binds.
    """
    day, idx = _daily_blocks(df)
    L = df['load_mw'].values[idx]
    A = df['ambient_c'].values[idx]
    shape = L / L[2 * PER_DAY:].max(axis=0)          # target day peak -> 1
    lo_s = np.zeros(len(day))
    hi_s = np.full(len(day), hi)

    def check(s):
        K = shape * s
        s0 = initial_state(p, K[:PER_DAY].mean(axis=0), A[:PER_DAY].mean(axis=0))
        r = simulate_exact(p, K, A, DT, s0)
        h = r['hot_spot'][2 * PER_DAY:]
        o = r['top_oil'][2 * PER_DAY:]
        lol = loss_of_life_days(h, DT, p.paper)
        return dict(hot=h.max(axis=0) <= limits['hot_spot'], oil=o.max(axis=0) <= limits['top_oil'],
                    cur=s <= limits['current'], age=lol <= ageing_limit_days,
                    h=h.max(axis=0), lol=lol)

    for _ in range(iters):
        mid = 0.5 * (lo_s + hi_s)
        c = check(mid)
        ok = c['hot'] & c['oil'] & c['cur'] & c['age']
        lo_s = np.where(ok, mid, lo_s)
        hi_s = np.where(ok, hi_s, mid)
    # which constraint fails first just above the rating
    c = check(lo_s * (1 + 1e-4) + 1e-6)
    binding = np.select([~c['cur'], ~c['age'], ~c['hot'], ~c['oil']],
                        ['current 1.5 pu', 'ageing 1 day/day', 'hot-spot 120 C', 'top-oil 105 C'], 'none')
    amb = df.groupby('day')['ambient_c'].agg(['mean', 'max']).loc[day]
    return pd.DataFrame({'day': day, 'rating_pu': lo_s, 'binding': binding,
                         'ambient_mean': amb['mean'].values, 'ambient_max': amb['max'].values})


def max_uniform_scaling(p, K_shape, theta_a, years, limits=LIMITS, iters=40):
    """Largest factor s such that s*K_shape keeps hot-spot, top-oil and current limits at all times
    and the insulation ageing of every calendar year at or below one year.

    Returns (s, binding constraint, details at s).
    """
    K_shape = np.asarray(K_shape, float)
    theta_a = np.asarray(theta_a, float)
    years = np.asarray(years)
    uy = np.unique(years)

    def run(s):
        r = simulate_series(p, s * K_shape, theta_a)
        lol = np.array([loss_of_life_days(r['hot_spot'][years == y], DT, p.paper) for y in uy])
        days_in = np.array([(years == y).sum() * DT / 1440.0 for y in uy])
        return dict(hot=r['hot_spot'].max(), oil=r['top_oil'].max(), cur=s * K_shape.max(),
                    age=(lol / days_in).max(), lol=lol, days=days_in)

    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = run(mid)
        ok = d['hot'] <= limits['hot_spot'] and d['oil'] <= limits['top_oil'] and \
            d['cur'] <= limits['current'] and d['age'] <= 1.0
        lo, hi = (mid, hi) if ok else (lo, mid)
    d = run(lo * (1 + 1e-4))
    binding = ('current' if d['cur'] > limits['current'] else 'ageing' if d['age'] > 1.0 else
               'hot-spot' if d['hot'] > limits['hot_spot'] else 'top-oil')
    return lo, binding, run(lo)
