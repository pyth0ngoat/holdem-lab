"""Hold'em Lab - Statistics Toolkit Unit & Verification Tests.

Verifies pure-NumPy holdem_lab.stats against SciPy and Statsmodels reference libraries.
"""

import math

import numpy as np
import pandas as pd
import scipy.special as sc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from holdem_lab.stats import (
    brownian_ruin_prob,
    chi2_cdf,
    chi2_goodness_of_fit,
    digamma,
    discrete_order_stat_max,
    f_cdf,
    fit_beta_mle,
    fit_beta_mom,
    incomplete_beta,
    incomplete_gamma,
    kendall_rank_correlation,
    norm_cdf,
    norm_ppf,
    ols_regression,
    pgf_moments,
    spearman_rank_correlation,
    t_cdf,
    two_way_anova_interaction,
    wls_regression,
)


def test_special_functions_against_scipy():
    """Verify digamma, incomplete_beta, and incomplete_gamma against SciPy."""
    for x in [0.5, 1.0, 2.5, 7.0, 15.0]:
        assert abs(digamma(x) - float(sc.digamma(x))) < 1e-7

    for a in [0.5, 1.5, 4.0]:
        for b in [0.5, 2.0, 5.0]:
            for x in [0.1, 0.5, 0.85]:
                assert abs(incomplete_beta(x, a, b) - float(sc.betainc(a, b, x))) < 1e-7

    for s in [0.5, 2.0, 5.0]:
        for x in [0.2, 1.5, 6.0]:
            assert abs(incomplete_gamma(s, x) - float(sc.gammainc(s, x))) < 1e-7


def test_distribution_cdfs_against_scipy():
    """Verify normal, t, f, and chi2 CDFs against scipy.stats."""
    # Normal
    for x in [-2.0, -0.5, 0.0, 1.2, 2.8]:
        assert abs(norm_cdf(x) - stats.norm.cdf(x)) < 1e-7
        assert abs(norm_ppf(stats.norm.cdf(x)) - x) < 1e-5

    # Student-t
    for df in [2, 5, 15, 30]:
        for x in [-2.5, 0.0, 1.8]:
            assert abs(t_cdf(x, df) - stats.t.cdf(x, df)) < 1e-7

    # F-distribution
    for df1, df2 in [(2, 10), (4, 25)]:
        for x in [0.5, 1.5, 3.5]:
            assert abs(f_cdf(x, df1, df2) - stats.f.cdf(x, df1, df2)) < 1e-7

    # Chi-square
    for df in [1, 3, 8, 14]:
        for x in [0.5, 2.5, 7.0]:
            assert abs(chi2_cdf(x, df) - stats.chi2.cdf(x, df)) < 1e-7


def test_beta_fits_against_scipy():
    """Verify Beta MoM and MLE fits on simulated data."""
    rng = np.random.default_rng(42)
    true_a, true_b = 2.5, 5.0
    data = rng.beta(true_a, true_b, size=5_000)

    # MoM
    mom_a, mom_b = fit_beta_mom(data)
    assert abs(mom_a - true_a) < 0.3
    assert abs(mom_b - true_b) < 0.5

    # MLE
    mle_a, mle_b = fit_beta_mle(data)
    sp_a, sp_b, _, _ = stats.beta.fit(data, floc=0, fscale=1)
    assert abs(mle_a - sp_a) < 0.05
    assert abs(mle_b - sp_b) < 0.05


def test_chi2_goodness_of_fit_against_scipy():
    """Verify chi-square goodness of fit against scipy.stats.chisquare."""
    obs = [48, 52, 45, 55]
    exp = [50, 50, 50, 50]
    res = chi2_goodness_of_fit(obs, exp)

    sp_stat, sp_pval = stats.chisquare(obs, f_exp=exp)
    assert abs(res["statistic"] - sp_stat) < 1e-7
    assert abs(res["p_value"] - sp_pval) < 1e-6


def test_ols_and_wls_against_statsmodels():
    """Verify OLS and WLS regression coefficients and SEs against Statsmodels."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(1.0, 10.0, n)
    X = np.column_stack([np.ones(n), x])
    y = 3.0 + 2.5 * x + rng.normal(0, 1.0, n)
    w = rng.uniform(0.5, 2.0, n)

    # OLS
    our_ols = ols_regression(X, y)
    sm_ols = sm.OLS(y, X).fit()
    assert np.allclose(our_ols["beta"], sm_ols.params, atol=1e-6)
    assert np.allclose(our_ols["se"], sm_ols.bse, atol=1e-6)
    assert abs(our_ols["r_squared"] - sm_ols.rsquared) < 1e-6

    # WLS
    our_wls = wls_regression(X, y, w)
    sm_wls = sm.WLS(y, X, weights=w).fit()
    assert np.allclose(our_wls["beta"], sm_wls.params, atol=1e-6)
    assert np.allclose(our_wls["se"], sm_wls.bse, atol=1e-6)


def test_two_way_anova_interaction_against_statsmodels():
    """Verify two-way ANOVA interaction F-statistic against Statsmodels OLS."""
    rng = np.random.default_rng(42)
    n_rep = 20
    a_levels = ["A1", "A2"]
    b_levels = ["B1", "B2", "B3"]

    data = []
    for a in a_levels:
        for b in b_levels:
            # Plant interaction
            base = 10.0 + (3.0 if a == "A2" and b == "B3" else 0.0)
            vals = rng.normal(base, 1.0, n_rep)
            for v in vals:
                data.append({"y": v, "A": a, "B": b})

    df = pd.DataFrame(data)
    our_res = two_way_anova_interaction(df["y"].values, df["A"].values, df["B"].values, n_permutations=200, seed=42)

    # Statsmodels reference
    model = smf.ols("y ~ C(A) + C(B) + C(A):C(B)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    sm_interaction_f = anova_table.loc["C(A):C(B)", "F"]

    assert abs(our_res["f_interaction"] - sm_interaction_f) < 1e-5
    assert our_res["p_analytical"] < 0.05
    assert our_res["p_permutation"] < 0.05


def test_rank_correlation_against_scipy():
    """Verify Spearman and Kendall correlations against SciPy."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    y = x + rng.normal(0, 0.5, 50)

    sp_rho, _ = stats.spearmanr(x, y)
    our_rho = spearman_rank_correlation(x, y)
    assert abs(our_rho - sp_rho) < 1e-6

    sp_tau, _ = stats.kendalltau(x, y)
    our_tau = kendall_rank_correlation(x, y)
    assert abs(our_tau - sp_tau) < 1e-6


def test_discrete_order_stat_max_toy():
    """Verify discrete tie-aware order statistic formula against exhaustive enumeration."""
    p = np.array([0.2, 0.5, 0.3])
    n = 3

    # Exact brute-force enumeration on 3 draws
    grid = np.array(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2])).T.reshape(-1, 3)
    probs = p[grid[:, 0]] * p[grid[:, 1]] * p[grid[:, 2]]
    max_vals = np.max(grid, axis=1)
    is_tie = np.sum(grid == max_vals[:, None], axis=1) > 1

    brute_tie = float(np.sum(probs[is_tie]))
    brute_cdf = [float(np.sum(probs[max_vals <= k])) for k in range(3)]

    res = discrete_order_stat_max(p, n)
    assert abs(res["prob_tie_for_max"] - brute_tie) < 1e-12
    assert np.allclose(res["cdf_max"], brute_cdf, atol=1e-12)


def test_pgf_moments_against_closed_form():
    """Verify numerical PGF moments against closed-form geometric distribution."""
    p = 0.2
    # PGF of Geometric(p) on {1, 2, ...} is G(s) = p*s / (1 - (1-p)*s)
    def geom_pgf(s: float) -> float:
        return (p * s) / (1.0 - (1.0 - p) * s)

    moments = pgf_moments(geom_pgf, h=1e-4)
    true_mean = 1.0 / p
    true_var = (1.0 - p) / (p * p)

    assert abs(moments["mean"] - true_mean) < 1e-4
    assert abs(moments["variance"] - true_var) < 1e-3


def test_brownian_ruin_prob_sanity():
    """Verify Brownian ruin formula properties."""
    # Negative or zero drift gives certainty of ruin
    assert brownian_ruin_prob(-0.1, 1.0, 10.0) == 1.0
    assert brownian_ruin_prob(0.0, 1.0, 10.0) == 1.0
    # Positive drift
    p_ruin = brownian_ruin_prob(mu=0.5, sigma=1.0, bankroll=5.0)
    # exp(-2 * 0.5 * 5 / 1) = exp(-5) ≈ 0.0067379
    assert abs(p_ruin - math.exp(-5.0)) < 1e-7
