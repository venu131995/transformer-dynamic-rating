"""All results of the project: verification, annual simulations, static vs dynamic rating, heatwave.

Writes results/summary.json, results/daily_rating.csv and the figures in figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from xfmr import iec60076_7 as m          # noqa: E402
from xfmr import rating as rt             # noqa: E402

FIG = ROOT / 'figures'
RES = ROOT / 'results'
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
BASE = m.MEDIUM_ONAF


def verification():
    out = {}
    p = BASE
    # 1. steady state at rated load, 20 degC: 20 + 52 + 26 = 98 degC by construction of Table 4
    out['steady_hot_spot_rated_20C'] = {q.name: round(m.steady_state(q, 1.0, 20.0)[1], 6) for q in m.ALL}
    # 2. exact integrator vs closed-form step response 0.5 -> 1.5 pu
    t = np.arange(1, 601) * 1.0
    _, h_cf = m.step_response_exact(p, 1.5, 20.0, t, K0=0.5)
    r = m.simulate_exact(p, np.full(600, 1.5), np.full(600, 20.0), 1.0, m.initial_state(p, 0.5, 20.0))
    out['exact_vs_closed_form_max_K'] = float(np.abs(r['hot_spot'] - h_cf).max())
    # 3. IEC difference equations converge to the closed form at first order
    dts = [4.0, 2.0, 1.0, 0.5, 0.25]
    errs = []
    for dt in dts:
        n = int(600 / dt)
        rd = m.simulate_difference(p, np.full(n, 1.5), np.full(n, 20.0), dt, m.initial_state(p, 0.5, 20.0))
        _, h2 = m.step_response_exact(p, 1.5, 20.0, np.arange(1, n + 1) * dt, K0=0.5)
        errs.append(float(np.abs(rd['hot_spot'] - h2).max()))
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    out['difference_eq_errors_K'] = dict(zip([str(d) for d in dts], errs))
    out['difference_eq_order'] = float(order)
    # 4. energy balance of the oil equation on real data
    df = rt.load_table(ROOT / 'data' / 'load_ambient_15min.csv.gz')
    y24 = df[df.time_utc.dt.year == 2024]
    K = rt.per_unit(y24.load_mw)
    s0 = m.initial_state(p, K[:96].mean(), y24.ambient_c.values[:96].mean())
    r = m.simulate_exact(p, K, y24.ambient_c.values, rt.DT, s0)
    eb = m.oil_energy_balance(p, K, y24.ambient_c.values, rt.DT, r['top_oil'], s0[0])
    out['oil_energy_balance_max_step_rel'] = float(eb['step_resid_rel'].max())
    out['oil_energy_balance_cumulative_rel'] = abs(eb['cumulative_resid']) / eb['cumulative_in']
    # 5. production solver (15 min, exact) vs IEC difference equations on 1-min sub-steps
    sub = 15
    Kf = np.repeat(K, sub)
    Af = np.repeat(y24.ambient_c.values, sub)
    rd = m.simulate_difference(p, Kf, Af, 1.0, s0)
    diff = np.abs(rd['hot_spot'][sub - 1::sub] - r['hot_spot'])
    out['exact15_vs_difference1min_2024'] = dict(max_K=float(diff.max()), mean_K=float(diff.mean()))
    return out, (dts, errs, t, h_cf, p)


def annual(df):
    """Every cooling type, peak load = nameplate (the static sizing rule)."""
    rows = []
    for p in m.ALL:
        for amb_case in ['real', 'constant 20 C']:
            for y in sorted(df.time_utc.dt.year.unique()):
                d = df[df.time_utc.dt.year == y]
                K = rt.per_unit(d.load_mw)
                A = d.ambient_c.values if amb_case == 'real' else np.full(len(d), 20.0)
                r = rt.simulate_series(p, K, A)
                rows.append(dict(cooling=p.name, ambient=amb_case, year=int(y),
                                 hot_spot_max=float(r['hot_spot'].max()), top_oil_max=float(r['top_oil'].max()),
                                 loss_of_life_days=float(m.loss_of_life_days(r['hot_spot'], rt.DT, p.paper)),
                                 days=len(d) * rt.DT / 1440))
    return pd.DataFrame(rows)


def uniform_scaling(df):
    years = df.time_utc.dt.year.values
    K = rt.per_unit(df.load_mw)
    rows = []
    for p in m.ALL:
        # 'load shifted 6 months': same load shape, but its winter peak now meets summer weather
        half = len(K) // 2
        for amb_case, A, KK in [('real', df.ambient_c.values, K), ('constant 20 C', np.full(len(df), 20.0), K),
                                ('real + 3 K', df.ambient_c.values + 3.0, K),
                                ('real, load shifted 6 months', df.ambient_c.values, np.roll(K, half))]:
            s, binding, d = rt.max_uniform_scaling(p, KK, A, years)
            rows.append(dict(cooling=p.name, ambient=amb_case, max_peak_pu=round(s, 4), binding=binding,
                             hot_spot_max=round(d['hot'], 2), worst_year_ageing=round(d['age'], 4)))
    return pd.DataFrame(rows)


def figures(df, ver_data, daily, ann):
    dts, errs, t, h_cf, p = ver_data
    # fig1 verification
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for dt, ls in [(4.0, ':'), (1.0, '--')]:
        n = int(600 / dt)
        rd = m.simulate_difference(p, np.full(n, 1.5), np.full(n, 20.0), dt, m.initial_state(p, 0.5, 20.0))
        ax[0].plot(np.arange(1, n + 1) * dt / 60, rd['hot_spot'], ls, label=f'IEC difference eq., dt = {dt:g} min')
    r = m.simulate_exact(p, np.full(40, 1.5), np.full(40, 20.0), 15.0, m.initial_state(p, 0.5, 20.0))
    ax[0].plot(t / 60, h_cf, 'k', lw=1, label='closed form')
    ax[0].plot(np.arange(1, 41) * 0.25, r['hot_spot'], 'o', ms=3, label='exact integrator, dt = 15 min')
    ax[0].set(xlabel='time after load step 0.5 -> 1.5 pu (h)', ylabel='hot-spot (degC)', xlim=(0, 10))
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].loglog(dts, errs, 'o-', label='IEC difference eq.')
    ax[1].loglog(dts, errs[2] * np.array(dts) / dts[2], 'k:', label='slope 1')
    ax[1].set(xlabel='time step (min)', ylabel='max hot-spot error (K)')
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_verification.png')
    plt.close(fig)

    # fig2 a winter and a summer week, peak = 1.0 pu
    y24 = df[df.time_utc.dt.year == 2024].reset_index(drop=True)
    K = rt.per_unit(y24.load_mw)
    r = rt.simulate_series(BASE, K, y24.ambient_c.values)
    fig, axs = plt.subplots(2, 2, figsize=(9, 5), sharey='row')
    for j, (lab, start) in enumerate([('winter week (Jan 2024)', '2024-01-15'), ('summer week (Aug 2024)', '2024-08-05')]):
        sel = (y24.time_utc >= pd.Timestamp(start, tz='UTC')) & (y24.time_utc < pd.Timestamp(start, tz='UTC') + pd.Timedelta(days=7))
        tt = y24.time_utc[sel].dt.tz_convert('Europe/Zurich').dt.tz_localize(None)
        axs[0, j].plot(tt, K[sel], color='C0')
        axs[0, j].set(title=lab, ylabel='load (pu of nameplate)' if j == 0 else None)
        axs[1, j].plot(tt, r['hot_spot'][sel], color='C3', label='hot-spot')
        axs[1, j].plot(tt, r['top_oil'][sel], color='C1', label='top oil')
        axs[1, j].plot(tt, y24.ambient_c[sel], color='C2', label='ambient (MeteoSwiss BUS)')
        axs[1, j].set(ylabel='temperature (degC)' if j == 0 else None)
        for a in axs[:, j]:
            a.tick_params(axis='x', labelrotation=30, labelsize=7)
    axs[1, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_weeks.png')
    plt.close(fig)

    # fig3 daily rating vs ambient
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for b, c in [('hot-spot 120 C', 'C3'), ('ageing 1 day/day', 'C0'), ('current 1.5 pu', 'C2'), ('top-oil 105 C', 'C1')]:
        s = daily[daily.binding == b]
        if len(s):
            ax[0].scatter(s.ambient_mean, s.rating_pu, s=5, color=c, label=f'limited by {b} ({len(s)} days)')
    fit = np.polyfit(daily.ambient_mean, daily.rating_pu, 1)
    xx = np.linspace(daily.ambient_mean.min(), daily.ambient_mean.max(), 10)
    ax[0].plot(xx, np.polyval(fit, xx), 'k--', lw=1, label=f'fit: {100 * fit[0]:.2f} % per K')
    ax[0].axhline(1.0, color='grey', lw=0.8)
    ax[0].set(xlabel='daily mean ambient (degC)', ylabel='permissible daily peak (pu of nameplate)')
    ax[0].legend(frameon=False, fontsize=7)
    daily = daily.assign(month=pd.to_datetime(daily.day).dt.month)
    ax[1].boxplot([daily.rating_pu[daily.month == k] for k in range(1, 13)], showfliers=False)
    ax[1].axhline(1.0, color='grey', lw=0.8)
    ax[1].set(xlabel='month', ylabel='permissible daily peak (pu)', title='2023-2025, ONAF')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_daily_rating.png')
    plt.close(fig)

    # fig4 annual ageing by cooling type
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    piv = (ann.assign(r=ann.loss_of_life_days / ann.days * 365).groupby(['cooling', 'ambient']).r.mean().unstack())
    piv.plot.bar(ax=ax, rot=0, color=['C0', 'C7'])
    ax.set(ylabel='insulation ageing (days per year)', xlabel='', title='peak load = nameplate, 2023-2025 mean')
    ax.legend(frameon=False, fontsize=7, title='ambient', title_fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_annual_ageing.png')
    plt.close(fig)


def heatwave(df, daily):
    amb_day = df.groupby('day').ambient_c.mean()
    roll = amb_day.rolling(7).mean()
    end = roll.idxmax()
    start = end - pd.Timedelta(days=6)
    sel = daily[(pd.to_datetime(daily.day) >= start) & (pd.to_datetime(daily.day) <= end)]
    return dict(start=str(start.date()), end=str(end.date()), mean_ambient=float(roll.max()),
                rating_min=float(sel.rating_pu.min()), rating_mean=float(sel.rating_pu.mean()))


def main():
    ver, ver_data = verification()
    df = rt.load_table(ROOT / 'data' / 'load_ambient_15min.csv.gz')
    ann = annual(df)
    uni = uniform_scaling(df)
    daily = rt.daily_dynamic_rating(BASE, df)
    daily.to_csv(RES / 'daily_rating.csv', index=False, float_format='%.4f')
    figures(df, ver_data, daily, ann)
    hw = heatwave(df, daily)

    d = daily.assign(month=pd.to_datetime(daily.day).dt.month)
    season = {'winter (DJF)': [12, 1, 2], 'spring (MAM)': [3, 4, 5], 'summer (JJA)': [6, 7, 8], 'autumn (SON)': [9, 10, 11]}
    fit = np.polyfit(daily.ambient_mean, daily.rating_pu, 1)
    summary = dict(
        data=dict(load='Swissgrid EnergieUebersichtCH 2023-2025, total end-user consumption, 15 min',
                  ambient='MeteoSwiss SwissMetNet BUS (Buchs/Aarau), hourly mean air temperature',
                  rows=len(df), days_rated=len(daily),
                  load_peak_mw=float(df.load_mw.max()), load_mean_mw=float(df.load_mw.mean()),
                  load_factor=float(df.load_mw.mean() / df.load_mw.max())),
        thermal_model='IEC 60076-7:2018 typical parameters (Table 4); base case ' + BASE.name,
        verification=ver,
        annual_peak_equals_nameplate=ann.round(3).to_dict(orient='records'),
        max_uniform_peak=uni.to_dict(orient='records'),
        daily_rating=dict(
            mean=float(daily.rating_pu.mean()), min=float(daily.rating_pu.min()), max=float(daily.rating_pu.max()),
            by_season={k: float(d.rating_pu[d.month.isin(v)].mean()) for k, v in season.items()},
            slope_pu_per_K=float(fit[0]), intercept_pu_at_0C=float(fit[1]),
            binding_counts=daily.binding.value_counts().to_dict(),
            share_days_above_1p2=float((daily.rating_pu > 1.2).mean())),
        heatwave=hw,
    )
    (RES / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['verification'], indent=1))
    print(uni.to_string())
    print(ann.groupby(['cooling', 'ambient'])[['hot_spot_max', 'loss_of_life_days']].mean().round(2).to_string())
    print(json.dumps(summary['daily_rating'], indent=1))
    print(hw)


if __name__ == '__main__':
    main()
