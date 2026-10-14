import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from xfmr import iec60076_7 as m   # noqa: E402
from xfmr import rating as rt      # noqa: E402


@pytest.mark.parametrize('p', m.ALL, ids=lambda p: p.name)
def test_rated_steady_state_is_98C(p):
    # Table 4 values are built so that rated load at 20 degC ambient gives the 98 degC reference hot-spot
    _, h = m.steady_state(p, 1.0, 20.0)
    assert h == pytest.approx(98.0, abs=1e-9)


def test_exact_integrator_matches_closed_form():
    p = m.MEDIUM_ONAF
    for dt in (1.0, 15.0):
        n = int(900 / dt)
        t = np.arange(1, n + 1) * dt
        _, h_cf = m.step_response_exact(p, 1.4, 25.0, t, K0=0.3)
        r = m.simulate_exact(p, np.full(n, 1.4), np.full(n, 25.0), dt, m.initial_state(p, 0.3, 25.0))
        assert np.abs(r['hot_spot'] - h_cf).max() < 1e-9


def test_difference_equations_first_order():
    p = m.MEDIUM_ONAN
    errs = []
    for dt in (2.0, 1.0, 0.5):
        n = int(600 / dt)
        rd = m.simulate_difference(p, np.full(n, 1.5), np.full(n, 20.0), dt, m.initial_state(p, 0.5, 20.0))
        _, h = m.step_response_exact(p, 1.5, 20.0, np.arange(1, n + 1) * dt, K0=0.5)
        errs.append(np.abs(rd['hot_spot'] - h).max())
    order = np.log(errs[0] / errs[2]) / np.log(4.0)
    assert 0.9 < order < 1.1


def test_oil_energy_balance_closes():
    p = m.DISTRIBUTION_ONAN
    rng = np.random.default_rng(0)
    K = np.clip(0.8 + 0.3 * rng.standard_normal(500), 0, 1.6)
    A = 10 + 8 * np.sin(np.arange(500) / 30)
    s0 = m.initial_state(p, 0.8, 10.0)
    r = m.simulate_exact(p, K, A, 15.0, s0)
    eb = m.oil_energy_balance(p, K, A, 15.0, r['top_oil'], s0[0])
    assert eb['step_resid_rel'].max() < 1e-12
    assert abs(eb['cumulative_resid']) < 1e-9 * eb['cumulative_in']


def test_ageing_rates():
    assert m.relative_ageing_rate(98.0, 'normal') == pytest.approx(1.0)
    assert m.relative_ageing_rate(104.0, 'normal') == pytest.approx(2.0)
    assert m.relative_ageing_rate(110.0, 'upgraded') == pytest.approx(1.0)
    assert m.loss_of_life_days(np.full(96, 110.0), 15.0, 'upgraded') == pytest.approx(1.0)


def test_batched_runs_equal_single_runs():
    p = m.MEDIUM_ONAF
    rng = np.random.default_rng(1)
    K = rng.uniform(0.2, 1.4, size=(200, 3))
    A = rng.uniform(-5, 30, size=(200, 3))
    s0 = m.initial_state(p, K[0], A[0])
    rb = m.simulate_exact(p, K, A, 15.0, s0)
    for j in range(3):
        r1 = m.simulate_exact(p, K[:, j], A[:, j], 15.0, m.initial_state(p, K[0, j], A[0, j]))
        assert np.allclose(rb['hot_spot'][:, j], r1['hot_spot'], atol=1e-12)


def _synthetic_df(days=6, amb=15.0):
    t = pd.date_range('2024-06-01', periods=days * 96, freq='15min', tz='Europe/Zurich').tz_convert('UTC')
    hour = np.arange(days * 96) / 4 % 24
    load = 1000 + 400 * np.exp(-((hour - 18) / 3) ** 2)
    df = pd.DataFrame({'time_utc': t, 'load_mw': load, 'ambient_c': amb})
    df['day'] = df.time_utc.dt.tz_convert('Europe/Zurich').dt.floor('D').dt.tz_localize(None)
    return df


def test_daily_rating_respects_limits():
    p = m.MEDIUM_ONAF
    df = _synthetic_df()
    d = rt.daily_dynamic_rating(p, df)
    assert len(d) == 4
    s = float(d.rating_pu.iloc[-1])
    # re-simulate the last day at the rating: within limits; 1 % above: a limit is broken
    for f, ok in [(1.0, True), (1.01, False)]:
        sub = df.iloc[-3 * 96:]
        shape = sub.load_mw.values / sub.load_mw.values[-96:].max()
        K = shape * s * f
        r = m.simulate_exact(p, K, sub.ambient_c.values, 15.0,
                             m.initial_state(p, K[:96].mean(), 15.0))
        h = r['hot_spot'][-96:]
        within = h.max() <= 120.0 + 1e-6 and m.loss_of_life_days(h, 15.0, p.paper) <= 1.0 + 1e-9 and s * f <= 1.5
        assert within == ok


def test_colder_days_rate_higher():
    p = m.MEDIUM_ONAN
    cold = rt.daily_dynamic_rating(p, _synthetic_df(amb=0.0)).rating_pu.mean()
    warm = rt.daily_dynamic_rating(p, _synthetic_df(amb=30.0)).rating_pu.mean()
    assert cold > warm


def test_prepared_data_is_continuous_utc():
    path = ROOT / 'data' / 'load_ambient_15min.csv.gz'
    if not path.exists():
        pytest.skip('run scripts/prepare_data.py first')
    df = rt.load_table(path)
    dt = df.time_utc.diff().dropna().unique()
    assert len(dt) == 1 and dt[0] == pd.Timedelta(minutes=15)
    assert df.load_mw.notna().all() and df.ambient_c.notna().all()
