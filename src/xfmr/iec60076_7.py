"""Transformer top-oil and winding hot-spot temperatures after IEC 60076-7:2018 (loading guide).

Two solvers for the same thermal model:

* ``simulate_difference`` - the standard's own difference equations (clause 8.2.3), explicit Euler.
* ``simulate_exact``      - the same three linear ODEs integrated exactly over each step for a
  load factor and ambient that are constant within the step (exponential integrator). Unconditionally
  stable, so it can run directly on 15-minute data; used as the production solver.

State variables (all in K or degC):
    theta_o   top-oil temperature
    dth1, dth2  the two hot-spot rise components, hot-spot rise = dth1 - dth2
The ODEs (K = load current / rated current, theta_a = ambient):
    k11 tau_o   d theta_o/dt = dtheta_or * ((1 + K^2 R)/(1 + R))^x - (theta_o - theta_a)
    k22 tau_w   d dth1/dt    = k21 dtheta_hr K^y - dth1
    tau_o/k22   d dth2/dt    = (k21 - 1) dtheta_hr K^y - dth2
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ThermalParams:
    name: str
    x: float           # oil exponent
    y: float           # winding exponent
    R: float           # ratio of load losses at rated current to no-load losses
    k11: float
    k21: float
    k22: float
    tau_o: float       # oil time constant, min
    tau_w: float       # winding time constant, min
    dtheta_or: float   # top-oil rise over ambient at rated load, K
    dtheta_hr: float   # hot-spot to top-oil gradient at rated load (H * g_r), K
    paper: str = 'upgraded'   # 'normal' (Kraft) or 'upgraded' (thermally upgraded)

    def with_(self, **kw):
        return replace(self, **kw)


# Typical values of IEC 60076-7:2018 Table 4. Real units should use their own heat-run test values.
DISTRIBUTION_ONAN = ThermalParams('Distribution ONAN', 0.8, 1.6, 5.0, 1.0, 1.0, 2.0, 180.0, 4.0, 55.0, 23.0,
                                  paper='normal')
MEDIUM_ONAN = ThermalParams('Medium/large ONAN', 0.8, 1.3, 6.0, 0.5, 2.0, 2.0, 210.0, 10.0, 52.0, 26.0)
MEDIUM_ONAF = ThermalParams('Medium/large ONAF', 0.8, 1.3, 6.0, 0.5, 2.0, 2.0, 150.0, 7.0, 52.0, 26.0)
MEDIUM_OF = ThermalParams('Medium/large OF', 1.0, 1.3, 6.0, 1.0, 1.3, 1.0, 90.0, 7.0, 56.0, 22.0)
ALL = [DISTRIBUTION_ONAN, MEDIUM_ONAN, MEDIUM_ONAF, MEDIUM_OF]

# Normal cyclic loading limits, IEC 60076-7 Table 6 (medium power transformers)
LIMITS = dict(hot_spot=120.0, top_oil=105.0, current=1.5)


def ultimate_top_oil_rise(p, K):
    return p.dtheta_or * ((1.0 + K ** 2 * p.R) / (1.0 + p.R)) ** p.x


def steady_state(p, K, theta_a):
    """Steady top-oil and hot-spot temperatures for constant load and ambient."""
    th_o = theta_a + ultimate_top_oil_rise(p, K)
    return th_o, th_o + p.dtheta_hr * K ** p.y


def initial_state(p, K0, theta_a0):
    """Steady state as (theta_o, dth1, dth2) - arrays allowed."""
    K0 = np.asarray(K0, float)
    th_o = theta_a0 + ultimate_top_oil_rise(p, K0)
    return th_o, p.k21 * p.dtheta_hr * K0 ** p.y, (p.k21 - 1.0) * p.dtheta_hr * K0 ** p.y


def simulate_difference(p, K, theta_a, dt_min, state0=None):
    """IEC 60076-7 clause 8.2.3 difference equations (explicit Euler).

    Stable and accurate only if dt_min is well below the smallest time constant (the standard
    asks for at most half of it). Returns dict of arrays with values at the END of each step.
    """
    K = np.asarray(K, float)
    theta_a = np.asarray(theta_a, float)
    if state0 is None:
        state0 = initial_state(p, K[0], theta_a[0])
    th_o, d1, d2 = (float(v) for v in state0)
    n = len(K)
    out_o, out_h = np.empty(n), np.empty(n)
    for i in range(n):
        Ki, ta = K[i], theta_a[i]
        d_o = dt_min / (p.k11 * p.tau_o) * (ultimate_top_oil_rise(p, Ki) - (th_o - ta))
        d_1 = dt_min / (p.k22 * p.tau_w) * (p.k21 * p.dtheta_hr * Ki ** p.y - d1)
        d_2 = dt_min / (p.tau_o / p.k22) * ((p.k21 - 1.0) * p.dtheta_hr * Ki ** p.y - d2)
        th_o, d1, d2 = th_o + d_o, d1 + d_1, d2 + d_2
        out_o[i], out_h[i] = th_o, th_o + d1 - d2
    return dict(top_oil=out_o, hot_spot=out_h, state=(th_o, d1, d2))


def _exp_factors(p, dt_min):
    return (np.exp(-dt_min / (p.k11 * p.tau_o)),
            np.exp(-dt_min / (p.k22 * p.tau_w)),
            np.exp(-dt_min / (p.tau_o / p.k22)))


def simulate_exact(p, K, theta_a, dt_min, state0=None):
    """Exact integration of the IEC ODEs with K and theta_a held constant over each step.

    K and theta_a may be 1-D (time) or 2-D (time, cases): every column is an independent run,
    which lets the rating search evaluate many candidate loadings at once.
    """
    K = np.asarray(K, float)
    theta_a = np.broadcast_to(np.asarray(theta_a, float), K.shape) if np.ndim(theta_a) < K.ndim \
        else np.asarray(theta_a, float)
    if state0 is None:
        state0 = initial_state(p, K[0], theta_a[0])
    th_o, d1, d2 = (np.array(v, float) for v in state0)
    e_o, e_1, e_2 = _exp_factors(p, dt_min)
    a1 = p.k21 * p.dtheta_hr
    a2 = (p.k21 - 1.0) * p.dtheta_hr
    out_o = np.empty_like(K)
    out_h = np.empty_like(K)
    for i in range(K.shape[0]):
        Ki = K[i]
        Ky = Ki ** p.y
        ult_o = theta_a[i] + ultimate_top_oil_rise(p, Ki)
        th_o = ult_o + (th_o - ult_o) * e_o
        d1 = a1 * Ky + (d1 - a1 * Ky) * e_1
        d2 = a2 * Ky + (d2 - a2 * Ky) * e_2
        out_o[i] = th_o
        out_h[i] = th_o + d1 - d2
    return dict(top_oil=out_o, hot_spot=out_h, state=(th_o, d1, d2))


def step_response_exact(p, K, theta_a, t_min, K0=0.0):
    """Closed-form response to a load step K0 -> K at t = 0 from steady state (clause 8.2.2 form)."""
    t = np.asarray(t_min, float)
    o0, h1_0, h2_0 = initial_state(p, K0, theta_a)
    ult_o = theta_a + ultimate_top_oil_rise(p, K)
    th_o = ult_o + (o0 - ult_o) * np.exp(-t / (p.k11 * p.tau_o))
    a1 = p.k21 * p.dtheta_hr * K ** p.y
    a2 = (p.k21 - 1.0) * p.dtheta_hr * K ** p.y
    d1 = a1 + (h1_0 - a1) * np.exp(-t / (p.k22 * p.tau_w))
    d2 = a2 + (h2_0 - a2) * np.exp(-t / (p.tau_o / p.k22))
    return th_o, th_o + d1 - d2


# ---------------------------------------------------------------- energy accounting of the oil model
def oil_energy_balance(p, K, theta_a, dt_min, top_oil, th_o0):
    """Discrete energy balance of the top-oil equation, written as a lumped heat balance.

    With C = k11 tau_o (in 'rated-loss minutes per kelvin of rise', i.e. normalised by P_rated/dtheta_or):
        C d theta_o/dt = Q_in(K) - Q_out(theta_o - theta_a)
    where Q_in = ultimate rise (the heat input expressed in kelvin of steady rise) and Q_out = actual rise.
    For the exact integrator the step-averaged Q_out is known in closed form, so
    stored energy - (heat in - heat out) must vanish to round-off, step by step and cumulatively.
    """
    K = np.asarray(K, float)
    theta_a = np.asarray(theta_a, float)
    tau = p.k11 * p.tau_o
    prev = np.concatenate([[th_o0], top_oil[:-1]])
    ult = theta_a + ultimate_top_oil_rise(p, K)
    # exact time average of (theta_o - theta_a) over a step for an exponential relaxation
    e = np.exp(-dt_min / tau)
    mean_theta = ult + (prev - ult) * tau / dt_min * (1.0 - e)
    q_in = ultimate_top_oil_rise(p, K) * dt_min
    q_out = (mean_theta - theta_a) * dt_min
    stored = tau * (top_oil - prev)
    resid = stored - (q_in - q_out)
    scale = np.maximum(np.abs(q_in), 1e-12)
    return dict(step_resid_rel=np.abs(resid) / scale, cumulative_resid=float(resid.sum()),
                cumulative_in=float(q_in.sum()))


# ---------------------------------------------------------------- insulation ageing (IEC 60076-7 clause 6.3)
def relative_ageing_rate(theta_h, paper='upgraded'):
    th = np.asarray(theta_h, float)
    if paper == 'normal':
        return 2.0 ** ((th - 98.0) / 6.0)
    return np.exp(15000.0 / (110.0 + 273.0) - 15000.0 / (th + 273.0))


def loss_of_life_days(theta_h, dt_min, paper='upgraded'):
    """Equivalent ageing in days at the reference hot-spot temperature (98 or 110 degC)."""
    return relative_ageing_rate(theta_h, paper).sum(axis=0) * dt_min / 1440.0
