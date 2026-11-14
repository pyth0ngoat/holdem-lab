"""Hold'em Lab - From-Scratch Statistics Toolkit.

Pure NumPy and Python standard library implementation.
No dependency on scipy or statsmodels (which are restricted to test and verification cells).
Every statistical formula is documented in its docstring.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


# =========================================================================
# 1. Special Functions
# =========================================================================
def loggamma(x: float) -> float:
    """Natural logarithm of the Gamma function ln(Gamma(x))."""
    return math.lgamma(x)


def digamma(x: float) -> float:
    """Digamma (psi) function psi(x) = d/dx ln(Gamma(x)).

    Computed via recurrence psi(x+1) = psi(x) + 1/x for x < 6,
    combined with the asymptotic expansion:
    psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6).
    """
    if x <= 0 and int(x) == x:
        raise ValueError(f"digamma is singular at non-positive integers, got {x}")

    res = 0.0
    # Reflection formula for negative x if needed
    if x < 0.5:
        # psi(1-x) - psi(x) = pi * cot(pi * x)
        return digamma(1.0 - x) - math.pi / math.tan(math.pi * x)

    while x < 6.0:
        res -= 1.0 / x
        x += 1.0

    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    res += (
        math.log(x)
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0)))
    )
    return res


def trigamma(x: float) -> float:
    """Trigamma function psi_1(x) = d^2/dx^2 ln(Gamma(x)).

    Computed via recurrence psi_1(x) = psi_1(x+1) + 1/x^2 for x < 6,
    combined with the asymptotic expansion:
    psi_1(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7).
    """
    res = 0.0
    while x < 6.0:
        res += 1.0 / (x * x)
        x += 1.0

    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    res += inv_x + 0.5 * inv_x2 + inv_x * inv_x2 * (1.0 / 6.0 - inv_x2 * (1.0 / 30.0))
    return res


def _betacf(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """Modified Lentz continued fraction helper for incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < 1e-15:
            break
    return h


def incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) = B(x; a, b) / B(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def incomplete_gamma(s: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(s, x) = gamma(s, x) / Gamma(s)."""
    if x <= 0.0:
        return 0.0
    if x < s + 1.0:
        sum_val = 1.0 / s
        term = sum_val
        for n in range(1, 250):
            term *= x / (s + n)
            sum_val += term
            if abs(term) < 1e-15 * abs(sum_val):
                break
        return sum_val * math.exp(-x + s * math.log(x) - math.lgamma(s))
    else:
        # Lentz method for upper incomplete gamma Q(s, x)
        b = x + 1.0 - s
        c = 1.0 / 1e-30
        d = 1.0 / b
        f = d
        for i in range(1, 250):
            a = -i * (i - s)
            b += 2.0
            d = a * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + a / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        q = f * math.exp(-x + s * math.log(x) - math.lgamma(s))
        return 1.0 - q


# =========================================================================
# 2. Probability Distributions (PDF, CDF, PPF)
# =========================================================================
def norm_pdf(x: float | np.ndarray, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
    """Standard Normal Probability Density Function."""
    z = (x - loc) / scale
    return np.exp(-0.5 * z * z) / (scale * math.sqrt(2.0 * math.pi))


def norm_cdf(x: float | np.ndarray, loc: float = 0.0, scale: float = 1.0) -> float | np.ndarray:
    """Standard Normal Cumulative Distribution Function via math.erf."""
    z = (x - loc) / (scale * math.sqrt(2.0))
    if isinstance(z, np.ndarray):
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(z))
    return 0.5 * (1.0 + math.erf(z))


def norm_ppf(p: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """Inverse standard normal CDF (Acklam rational approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients in rational approximations
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    q = min(p, 1.0 - p)
    if q > 0.02425:
        r = p - 0.5
        r2 = r * r
        z = (((((a[0]*r2 + a[1])*r2 + a[2])*r2 + a[3])*r2 + a[4])*r2 + a[5])*r / (((((b[0]*r2 + b[1])*r2 + b[2])*r2 + b[3])*r2 + b[4])*r2 + 1.0)
    else:
        r = math.sqrt(-2.0 * math.log(q))
        z = (((((c[0]*r + c[1])*r + c[2])*r + c[3])*r + c[4])*r + c[5]) / ((((d[0]*r + d[1])*r + d[2])*r + d[3])*r + 1.0)
        if p > 0.5:
            z = -z
    return loc + scale * z


def t_pdf(x: float | np.ndarray, df: float) -> float | np.ndarray:
    """Student-t probability density function."""
    coef = math.exp(math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)) / math.sqrt(math.pi * df)
    return coef * (1.0 + (x * x) / df) ** (-0.5 * (df + 1.0))


def t_cdf(x: float, df: float) -> float:
    """Student-t cumulative distribution function."""
    if x == 0.0:
        return 0.5
    z = df / (df + x * x)
    ib = incomplete_beta(z, 0.5 * df, 0.5)
    return 1.0 - 0.5 * ib if x > 0 else 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF via bisection / Newton."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    # Initial estimate from normal
    z = norm_ppf(p)
    low, high = -100.0, 100.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        val = t_cdf(mid, df)
        if abs(val - p) < 1e-8:
            return mid
        if val < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def beta_pdf(x: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """Beta probability density function."""
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=float)
        valid = (x > 0.0) & (x < 1.0)
        log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        res[valid] = np.exp((a - 1.0) * np.log(x[valid]) + (b - 1.0) * np.log(1.0 - x[valid]) - log_b)
        return res
    if x <= 0.0 or x >= 1.0:
        return 0.0
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_b)


def beta_cdf(x: float, a: float, b: float) -> float:
    """Beta cumulative distribution function."""
    return incomplete_beta(x, a, b)


def gamma_pdf(x: float | np.ndarray, shape: float, scale: float = 1.0) -> float | np.ndarray:
    """Gamma probability density function."""
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=float)
        valid = x > 0.0
        log_k = -math.lgamma(shape) - shape * math.log(scale)
        res[valid] = np.exp((shape - 1.0) * np.log(x[valid]) - x[valid] / scale + log_k)
        return res
    if x <= 0.0:
        return 0.0
    log_k = -math.lgamma(shape) - shape * math.log(scale)
    return math.exp((shape - 1.0) * math.log(x) - x / scale + log_k)


def gamma_cdf(x: float, shape: float, scale: float = 1.0) -> float:
    """Gamma cumulative distribution function."""
    return incomplete_gamma(shape, x / scale)


def f_pdf(x: float | np.ndarray, df1: float, df2: float) -> float | np.ndarray:
    """Snedecor's F-distribution probability density function."""
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=float)
        valid = x > 0.0
        c = (
            math.lgamma(0.5 * (df1 + df2))
            - math.lgamma(0.5 * df1)
            - math.lgamma(0.5 * df2)
            + 0.5 * df1 * math.log(df1 / df2)
        )
        res[valid] = np.exp(c + (0.5 * df1 - 1.0) * np.log(x[valid]) - 0.5 * (df1 + df2) * np.log(1.0 + (df1 / df2) * x[valid]))
        return res
    if x <= 0.0:
        return 0.0
    c = (
        math.lgamma(0.5 * (df1 + df2))
        - math.lgamma(0.5 * df1)
        - math.lgamma(0.5 * df2)
        + 0.5 * df1 * math.log(df1 / df2)
    )
    return math.exp(c + (0.5 * df1 - 1.0) * math.log(x) - 0.5 * (df1 + df2) * math.log(1.0 + (df1 / df2) * x))


def f_cdf(x: float, df1: float, df2: float) -> float:
    """Snedecor's F-distribution cumulative distribution function."""
    if x <= 0.0:
        return 0.0
    z = (df1 * x) / (df1 * x + df2)
    return incomplete_beta(z, 0.5 * df1, 0.5 * df2)


def chi2_pdf(x: float | np.ndarray, df: float) -> float | np.ndarray:
    """Chi-square probability density function."""
    return gamma_pdf(x, shape=0.5 * df, scale=2.0)


def chi2_cdf(x: float, df: float) -> float:
    """Chi-square cumulative distribution function."""
    if x <= 0.0:
        return 0.0
    return incomplete_gamma(0.5 * df, 0.5 * x)


def chi2_ppf(p: float, df: float) -> float:
    """Inverse chi-square CDF via bisection."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    low, high = 0.0, max(100.0, df * 10.0)
    for _ in range(60):
        mid = 0.5 * (low + high)
        val = chi2_cdf(mid, df)
        if abs(val - p) < 1e-8:
            return mid
        if val < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# =========================================================================
# 3. Parameter Fitting (MoM and MLE)
# =========================================================================
def fit_beta_mom(data: np.ndarray) -> tuple[float, float]:
    """Fit Beta distribution parameters (alpha, beta) via Method of Moments.

    Formula:
    mean = x_bar, var = s^2
    term = (mean * (1 - mean) / var) - 1
    alpha = mean * term
    beta = (1 - mean) * term
    """
    mean = float(np.mean(data))
    var = float(np.var(data, ddof=1))
    if var <= 0 or var >= mean * (1.0 - mean):
        raise ValueError(f"Variance {var} invalid for Beta MoM fit with mean {mean}")
    term = (mean * (1.0 - mean) / var) - 1.0
    alpha = max(1e-4, mean * term)
    beta = max(1e-4, (1.0 - mean) * term)
    return alpha, beta


def fit_beta_mle(data: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> tuple[float, float]:
    """Fit Beta distribution parameters (alpha, beta) via Maximum Likelihood (Newton-Raphson)."""
    # Initialize from Method of Moments
    try:
        a, b = fit_beta_mom(data)
    except Exception:
        a, b = 1.0, 1.0

    n = len(data)
    # Clip data slightly away from 0 and 1 for log stability
    x = np.clip(data, 1e-8, 1.0 - 1e-8)
    sum_log_x = float(np.sum(np.log(x)))
    sum_log_1_minus_x = float(np.sum(np.log(1.0 - x)))

    for _ in range(max_iter):
        psi_a = digamma(a)
        psi_b = digamma(b)
        psi_ab = digamma(a + b)

        # Gradients
        g_a = n * (psi_ab - psi_a) + sum_log_x
        g_b = n * (psi_ab - psi_b) + sum_log_1_minus_x

        # Hessian components
        tri_a = trigamma(a)
        tri_b = trigamma(b)
        tri_ab = trigamma(a + b)

        h_aa = n * (tri_ab - tri_a)
        h_bb = n * (tri_ab - tri_b)
        h_ab = n * tri_ab

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break

        # Newton step: delta = H^{-1} * g
        delta_a = (h_bb * g_a - h_ab * g_b) / det
        delta_b = (-h_ab * g_a + h_aa * g_b) / det

        a_new = a - delta_a
        b_new = b - delta_b

        if a_new <= 0 or b_new <= 0:
            # Damped step if out of bounds
            delta_a *= 0.5
            delta_b *= 0.5
            a_new = max(1e-4, a - delta_a)
            b_new = max(1e-4, b - delta_b)

        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return a, b


def fit_gamma_mom(data: np.ndarray) -> tuple[float, float]:
    """Fit Gamma distribution (shape k, scale theta) via Method of Moments."""
    mean = float(np.mean(data))
    var = float(np.var(data, ddof=1))
    shape = (mean * mean) / var
    scale = var / mean
    return shape, scale


def fit_gamma_mle(data: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> tuple[float, float]:
    """Fit Gamma distribution (shape k, scale theta) via MLE."""
    shape, _ = fit_gamma_mom(data)
    mean = float(np.mean(data))
    log_mean = math.log(mean)
    mean_log = float(np.mean(np.log(np.clip(data, 1e-12, None))))
    s = log_mean - mean_log

    # Newton-Raphson for shape
    for _ in range(max_iter):
        g = math.log(shape) - digamma(shape) - s
        h = 1.0 / shape - trigamma(shape)
        if abs(h) < 1e-12:
            break
        shape_new = shape - g / h
        if shape_new <= 0:
            shape_new = 0.5 * shape
        if abs(shape_new - shape) < tol:
            shape = shape_new
            break
        shape = shape_new

    scale = mean / shape
    return shape, scale


def fit_geometric_mom(data: np.ndarray) -> float:
    """Fit Geometric distribution parameter p via Method of Moments (support {1, 2, ...})."""
    mean = float(np.mean(data))
    return 1.0 / mean if mean > 0 else 1.0


def fit_geometric_mle(data: np.ndarray) -> float:
    """Fit Geometric distribution parameter p via MLE (support {1, 2, ...})."""
    return fit_geometric_mom(data)


def fit_poisson_mle(data: np.ndarray) -> float:
    """Fit Poisson distribution parameter lambda via MLE."""
    return float(np.mean(data))


def fit_negative_binomial_mom(data: np.ndarray) -> tuple[float, float]:
    """Fit Negative Binomial parameters (r, p) via Method of Moments (support {0, 1, ...})."""
    mean = float(np.mean(data))
    var = float(np.var(data, ddof=1))
    if var <= mean:
        # Fallback to Poisson-like regime
        return 1.0, 1.0 / (1.0 + mean)
    p = mean / var
    r = (mean * mean) / (var - mean)
    return r, p


# =========================================================================
# 4. Hypothesis Testing & Goodness of Fit
# =========================================================================
def ks_statistic(data: np.ndarray, cdf_func: Callable[[float], float]) -> float:
    """Compute one-sample Kolmogorov-Smirnov statistic D = sup |F_n(x) - F_0(x)|."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_vals = np.array([cdf_func(x) for x in sorted_data])
    d_plus = np.max(np.arange(1, n + 1) / n - cdf_vals)
    d_minus = np.max(cdf_vals - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def chi2_goodness_of_fit(
    observed: Sequence[float],
    expected: Sequence[float],
    params_estimated: int = 0,
    min_expected: float = 5.0,
) -> dict[str, Any]:
    """Perform chi-square goodness-of-fit test with sparse bin pooling.

    Parameters
    ----------
    observed : sequence of float
        Observed category counts.
    expected : sequence of float
        Expected category counts (or expected probabilities scaled by N).
    params_estimated : int
        Number of parameters fitted from data (subtracted from degrees of freedom).
    min_expected : float
        Minimum expected frequency per bin; bins with expected < min_expected are pooled.

    Returns
    -------
    dict
        Dictionary containing statistic, p_value, df, pooled_observed, pooled_expected.
    """
    obs = list(observed)
    exp = list(expected)

    # Scale expected to match sum of observed if probabilities were provided
    total_obs = sum(obs)
    if abs(sum(exp) - 1.0) < 1e-4:
        exp = [e * total_obs for e in exp]

    # Pool bins where expected < min_expected
    pooled_obs: list[float] = []
    pooled_exp: list[float] = []

    cur_obs = 0.0
    cur_exp = 0.0

    for o, e in zip(obs, exp):
        cur_obs += o
        cur_exp += e
        if cur_exp >= min_expected:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)
            cur_obs = 0.0
            cur_exp = 0.0

    # If remaining tail did not reach min_expected, pool into last bin
    if cur_exp > 0.0:
        if len(pooled_exp) > 0:
            pooled_obs[-1] += cur_obs
            pooled_exp[-1] += cur_exp
        else:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)

    k_pooled = len(pooled_obs)
    df = k_pooled - 1 - params_estimated
    if df <= 0:
        df = 1  # Guard against zero degrees of freedom

    chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(pooled_obs, pooled_exp))
    p_val = max(0.0, 1.0 - chi2_cdf(chi2_stat, df))

    return {
        "statistic": chi2_stat,
        "p_value": p_val,
        "df": df,
        "k_pooled": k_pooled,
        "pooled_observed": pooled_obs,
        "pooled_expected": pooled_exp,
    }


# =========================================================================
# 5. Regression & Confidence Intervals
# =========================================================================
def ols_regression(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ordinary Least Squares regression with analytical standard errors."""
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    # beta = (X^T X)^{-1} X^T y
    xtx = X.T @ X
    xty = X.T @ y
    beta = np.linalg.solve(xtx, xty)

    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    df_resid = max(1, n - p)
    s2 = ss_res / df_resid
    inv_xtx = np.linalg.inv(xtx)
    se = np.sqrt(np.maximum(0.0, np.diag(s2 * inv_xtx)))

    t_stats = beta / np.where(se > 0, se, 1e-12)
    p_values = [2.0 * (1.0 - t_cdf(abs(t), df_resid)) for t in t_stats]

    return {
        "beta": beta,
        "se": se,
        "r_squared": r_squared,
        "p_values": p_values,
        "residuals": residuals,
        "df_resid": df_resid,
    }


def wls_regression(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Weighted Least Squares regression with analytical standard errors."""
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape

    w = np.sqrt(weights)
    X_w = X * w[:, None]
    y_w = y * w

    return ols_regression(X_w, y_w)


def t_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Student-t confidence interval for sample mean."""
    n = len(data)
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1) / math.sqrt(n))
    t_crit = t_ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2_000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval (iid or moving block)."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_boot)

    if block_size is None or block_size <= 1:
        # Standard iid bootstrap
        for b in range(n_boot):
            sample = rng.choice(data, size=n, replace=True)
            boot_stats[b] = stat_func(sample)
    else:
        # Moving block bootstrap
        num_blocks = math.ceil(n / block_size)
        for b in range(n_boot):
            start_indices = rng.integers(0, n - block_size + 1, size=num_blocks)
            blocks = [data[idx : idx + block_size] for idx in start_indices]
            sample = np.concatenate(blocks)[:n]
            boot_stats[b] = stat_func(sample)

    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(boot_stats, 100.0 * alpha))
    high = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return low, high


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 2_000,
    seed: int = 42,
) -> float:
    """Two-sample permutation test for difference in means."""
    rng = np.random.default_rng(seed)
    obs_diff = abs(float(np.mean(group1) - np.mean(group2)))
    n1 = len(group1)
    combined = np.concatenate([group1, group2])
    n_total = len(combined)

    count_extreme = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        diff = abs(float(np.mean(perm[:n1]) - np.mean(perm[n1:])))
        if diff >= obs_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


# =========================================================================
# 6. Analysis of Variance (ANOVA)
# =========================================================================
def one_way_anova(*groups: np.ndarray) -> dict[str, Any]:
    """One-way Analysis of Variance (ANOVA)."""
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand_mean = sum(np.sum(g) for g in groups) / n_total

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    p_val = max(0.0, 1.0 - f_cdf(f_stat, df_between, df_within))

    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0.0

    return {
        "f_stat": f_stat,
        "p_value": p_val,
        "eta_squared": eta_sq,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "df_between": df_between,
        "df_within": df_within,
    }


def two_way_anova_interaction(
    y: np.ndarray,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    n_permutations: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Two-way balanced ANOVA with interaction F-test and permutation calibration."""
    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)
    I = len(levels_a)
    J = len(levels_b)

    grand_mean = float(np.mean(y))
    # Cell means
    cell_means = {}
    for a in levels_a:
        for b in levels_b:
            mask = (factor_a == a) & (factor_b == b)
            cell_means[(a, b)] = float(np.mean(y[mask]))

    # Row and col means
    row_means = {a: float(np.mean(y[factor_a == a])) for a in levels_a}
    col_means = {b: float(np.mean(y[factor_b == b])) for b in levels_b}

    # Sum of squares
    ss_ab = 0.0
    ss_e = 0.0
    k_rep = len(y) // (I * J)

    for (a, b), cmean in cell_means.items():
        inter = cmean - row_means[a] - col_means[b] + grand_mean
        ss_ab += k_rep * (inter**2)
        mask = (factor_a == a) & (factor_b == b)
        ss_e += float(np.sum((y[mask] - cmean) ** 2))

    df_ab = (I - 1) * (J - 1)
    df_e = I * J * (k_rep - 1)

    ms_ab = ss_ab / df_ab
    ms_e = ss_e / df_e if df_e > 0 else 1.0

    f_stat = ms_ab / ms_e if ms_e > 0 else 0.0
    p_analytical = max(0.0, 1.0 - f_cdf(f_stat, df_ab, df_e))

    # Permutation-calibrated p-value
    rng = np.random.default_rng(seed)
    perm_count = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_cell_means = {}
        for a in levels_a:
            for b in levels_b:
                mask = (factor_a == a) & (factor_b == b)
                perm_cell_means[(a, b)] = float(np.mean(y_perm[mask]))
        p_row_means = {a: float(np.mean(y_perm[factor_a == a])) for a in levels_a}
        p_col_means = {b: float(np.mean(y_perm[factor_b == b])) for b in levels_b}
        p_grand = float(np.mean(y_perm))

        p_ss_ab = sum(
            k_rep * (perm_cell_means[(a, b)] - p_row_means[a] - p_col_means[b] + p_grand) ** 2
            for a in levels_a
            for b in levels_b
        )
        p_ms_ab = p_ss_ab / df_ab
        p_f = p_ms_ab / ms_e
        if p_f >= f_stat:
            perm_count += 1

    p_permutation = (perm_count + 1) / (n_permutations + 1)
    ss_tot = float(np.sum((y - grand_mean) ** 2))
    eta_squared = ss_ab / ss_tot if ss_tot > 0 else 0.0

    return {
        "f_interaction": f_stat,
        "p_analytical": p_analytical,
        "p_permutation": p_permutation,
        "eta_squared": eta_squared,
        "df_ab": df_ab,
        "df_e": df_e,
    }


# =========================================================================
# 7. Rank Correlation
# =========================================================================
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Compute average fractional ranks for array."""
    sorter = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    x_sorted = x[sorter]

    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        avg_rank = 1.0 + (i + j - 1) / 2.0
        ranks[sorter[i:j]] = avg_rank
        i = j
    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    cov = float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))))
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def kendall_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall rank correlation coefficient tau-b."""
    n = len(x)
    concordant = 0
    discordant = 0
    extra_x = 0
    extra_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0 and dy != 0:
                    extra_x += 1
                elif dy == 0 and dx != 0:
                    extra_y += 1

    denom = math.sqrt((concordant + discordant + extra_x) * (concordant + discordant + extra_y))
    return (concordant - discordant) / denom if denom > 0 else 0.0


# =========================================================================
# 8. Order Statistics
# =========================================================================
def order_stat_max_cdf(x: float, cdf_func: Callable[[float], float], n: int) -> float:
    """Continuous order statistic: CDF of max of n iid draws: F_max(x) = F(x)^n."""
    return cdf_func(x) ** n


def order_stat_min_cdf(x: float, cdf_func: Callable[[float], float], n: int) -> float:
    """Continuous order statistic: CDF of min of n iid draws: F_min(x) = 1 - (1 - F(x))^n."""
    return 1.0 - (1.0 - cdf_func(x)) ** n


def discrete_order_stat_max(probs: np.ndarray, n: int) -> dict[str, Any]:
    """Discrete tie-aware order statistics for max of n iid draws.

    Parameters
    ----------
    probs : np.ndarray
        PMF array over discrete categories 0..K.
    n : int
        Number of independent draws.

    Returns
    -------
    dict
        cdf_max: P(max <= k)
        pmf_max: P(max = k)
        prob_tie_for_max: P(two or more draws tied for strictly maximum value)
    """
    k_len = len(probs)
    F = np.cumsum(probs)

    cdf_max = F**n
    pmf_max = np.empty(k_len)
    pmf_max[0] = cdf_max[0]
    pmf_max[1:] = cdf_max[1:] - cdf_max[:-1]

    # Probability of tie for the maximum
    prob_tie = 0.0
    for k in range(k_len):
        f_k = F[k]
        f_prev = F[k - 1] if k > 0 else 0.0
        p_k = probs[k]
        # P(max = k and sole winner) = n * p_k * F(k-1)^{n-1}
        p_sole = n * p_k * (f_prev ** (n - 1))
        prob_tie += (pmf_max[k] - p_sole)

    return {
        "cdf_max": cdf_max,
        "pmf_max": pmf_max,
        "prob_tie_for_max": prob_tie,
    }


# =========================================================================
# 9. Probability Generating Functions (PGF) & Brownian Ruin
# =========================================================================
def pgf_moments(pgf_func: Callable[[float], float], h: float = 1e-4) -> dict[str, float]:
    """Derive numerical mean and variance from a Probability Generating Function G(s).

    Formulas:
    E[X] = G'(1)
    Var(X) = G''(1) + G'(1) - (G'(1))^2
    Using central difference stencils at s = 1.
    """
    g_plus = pgf_func(1.0 + h)
    g_one = pgf_func(1.0)
    g_minus = pgf_func(1.0 - h)

    # First derivative G'(1)
    g1 = (g_plus - g_minus) / (2.0 * h)
    # Second derivative G''(1)
    g2 = (g_plus - 2.0 * g_one + g_minus) / (h * h)

    mean = g1
    variance = g2 + g1 - (g1 * g1)

    return {
        "mean": mean,
        "variance": variance,
        "first_derivative": g1,
        "second_derivative": g2,
    }


def brownian_ruin_prob(mu: float, sigma: float, bankroll: float) -> float:
    """Brownian motion approximation to probability of ruin with positive drift.

    Formula:
    P(ruin) = exp(-2 * mu * B / sigma^2)
    """
    if bankroll <= 0:
        return 1.0
    if mu <= 0:
        return 1.0
    if sigma <= 0:
        return 0.0
    exponent = -2.0 * mu * bankroll / (sigma * sigma)
    return math.exp(exponent)
