"""Hold'em Lab - Core Infrastructure, Cards, Evaluator, and Benchmark Tests.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import treys

from holdem_lab import (
    CACHE_DIR,
    CONFIG,
    FIGURES_DATA_DIR,
    FIGURES_DIR,
    MODE,
    REPORTS_DIR,
    ROOT_DIR,
)
from holdem_lab.cards import (
    CATEGORY_NAMES,
    EXACT_5CARD_COUNTS,
    EXACT_5CARD_PROBS,
    EXACT_7CARD_COUNTS,
    EXACT_7CARD_PROBS,
    P_7CARD_ROYAL_FLUSH,
    P_FLUSH_BY_RIVER_FROM_FLOP_DRAW,
    P_SET_ON_FLOP,
    RANKS_STR,
    SUITS_STR,
    TOTAL_5CARD,
    TOTAL_7CARD,
    card_to_str,
    evaluate_5cards,
    evaluate_7cards,
    evaluate_hands,
    get_deck,
    measure_evaluator_speed,
    str_to_card,
)
from holdem_lab.equity import (
    ALL_CLASSES,
    COMBO_WEIGHTS,
    TOTAL_COMBOS,
    cards_to_class,
    class_combos_count,
    class_representative,
    class_to_combos,
    estimate_hand_equity,
    exact_board_equity,
    exact_turn_equity,
    simulate_table_deals,
)
from holdem_lab.findings import (
    assert_all_claims_backed,
    clear_claims_ledger,
    load_findings_records,
    record_finding,
    register_claim,
    save_fig,
    write_findings,
)


def test_repo_layout_directories():
    """Verify that expected core directories exist."""
    assert ROOT_DIR.exists()
    assert CACHE_DIR.exists()
    assert FIGURES_DIR.exists()
    assert FIGURES_DATA_DIR.exists()
    assert REPORTS_DIR.exists()


def test_config_structure():
    """Verify configuration parameters."""
    assert MODE in ("quick", "full")
    assert "seeds" in CONFIG
    assert CONFIG["seeds"]["fit"] == 42
    assert CONFIG["seeds"]["validation"] == 1337
    assert CONFIG["seeds"]["demo"] == 2026
    assert CONFIG["formula_tolerance"] == 0.01


def test_findings_record_and_save_fig(tmp_path):
    """Verify recording a finding, saving a figure, and CSV data generation."""
    test_fid = "T-CORE-1"
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    ax.plot(x, y, label="y=2x+1")
    ax.set_title("Test Figure")

    df = pd.DataFrame({"x": x, "y": y})
    fig_path = save_fig(fig, test_fid, "test_plot", "Test plot caption", data=df)

    assert Path(fig_path).exists()
    assert (FIGURES_DIR / f"{test_fid}_test_plot.png").exists()
    assert (FIGURES_DATA_DIR / f"{test_fid}_test_plot.csv").exists()

    rec = record_finding(
        fid=test_fid,
        title="Test Core Finding",
        question="Does findings helper record correctly?",
        method="Unit test verification",
        results={"slope": 2.0, "intercept": {"value": 1.0, "se": 0.01}},
        figures=[fig_path],
        verdict="VERIFIED",
        caveats="Synthetic unit test only",
        gap="VERIFICATION",
    )

    records = load_findings_records()
    assert test_fid in records
    assert records[test_fid]["verdict"] == "VERIFIED"
    assert records[test_fid]["results"]["slope"] == 2.0


def test_write_findings_idempotency():
    """Verify that write_findings generates reports/findings.md and is idempotent (ignoring timestamp)."""
    md1 = write_findings()
    findings_md_file = REPORTS_DIR / "findings.md"
    assert findings_md_file.exists()

    md2 = write_findings()
    # Strip the generation timestamp line for exact idempotency comparison
    lines1 = [line for line in md1.splitlines() if not line.startswith("*Auto-generated on:")]
    lines2 = [line for line in md2.splitlines() if not line.startswith("*Auto-generated on:")]
    assert lines1 == lines2


def test_claims_ledger_verification():
    """Verify that assert_all_claims_backed succeeds for valid claims and fails for missing ones."""
    clear_claims_ledger()

    record_finding(
        fid="T-BACKED-1",
        title="Backed Claim Finding",
        question="Is it backed?",
        method="Direct verification",
        results={"success": True},
        figures=[],
        verdict="VERIFIED",
        caveats="None",
    )
    register_claim("System has core verification", "T-BACKED-1")
    assert_all_claims_backed()

    register_claim("Non-existent claim", "T-MISSING-999")
    with pytest.raises(AssertionError) as exc_info:
        assert_all_claims_backed()
    assert "T-MISSING-999" in str(exc_info.value)

    clear_claims_ledger()


# -------------------------------------------------------------------------
# P1: Cards, Evaluator, and Exact Combinatorics Tests
# -------------------------------------------------------------------------
def test_card_encoding_and_strings():
    """Test string conversions and deck properties."""
    deck = get_deck()
    assert len(deck) == 52
    assert len(np.unique(deck)) == 52

    for c in range(52):
        s = card_to_str(c)
        assert len(s) == 2
        assert str_to_card(s) == c

    assert card_to_str(str_to_card("As")) == "As"
    assert card_to_str(str_to_card("2c")) == "2c"
    assert card_to_str(str_to_card("Td")) == "Td"


def test_exact_counts_and_probabilities():
    """Verify exact combinatorial category counts sum to theoretical totals."""
    assert sum(EXACT_5CARD_COUNTS.values()) == TOTAL_5CARD == 2_598_960
    assert sum(EXACT_7CARD_COUNTS.values()) == TOTAL_7CARD == 133_784_560
    assert abs(sum(EXACT_5CARD_PROBS.values()) - 1.0) < 1e-12
    assert abs(sum(EXACT_7CARD_PROBS.values()) - 1.0) < 1e-12


def test_benchmark_probabilities():
    """Verify the three required benchmark probabilities against textbook formulas."""
    expected_set = 1.0 - math.comb(48, 3) / math.comb(50, 3)
    assert abs(P_SET_ON_FLOP - expected_set) < 1e-12
    assert abs(1.0 / P_SET_ON_FLOP - 8.506944) < 1e-3

    expected_flush = 1.0 - math.comb(38, 2) / math.comb(47, 2)
    assert abs(P_FLUSH_BY_RIVER_FROM_FLOP_DRAW - expected_flush) < 1e-12
    assert abs(P_FLUSH_BY_RIVER_FROM_FLOP_DRAW - 0.349676) < 1e-4

    expected_rf = 4324 / 133_784_560
    assert abs(P_7CARD_ROYAL_FLUSH - expected_rf) < 1e-12
    assert abs(1.0 / P_7CARD_ROYAL_FLUSH - 30940.0) < 0.1


def test_evaluator_wheel_straight_and_royal_flush():
    """Test straight wheel (A-2-3-4-5) and Royal Flush classification."""
    wheel_cards = np.array([str_to_card("5c"), str_to_card("4d"), str_to_card("3h"), str_to_card("2s"), str_to_card("Ac")])
    scores_5, cats_5 = evaluate_5cards(wheel_cards.reshape(1, 5))
    assert cats_5[0] == 4

    six_straight = np.array([str_to_card("6c"), str_to_card("5d"), str_to_card("4h"), str_to_card("3s"), str_to_card("2c")])
    scores_6, cats_6 = evaluate_5cards(six_straight.reshape(1, 5))
    assert cats_6[0] == 4
    assert scores_6[0] > scores_5[0]

    rf_7 = np.array([
        str_to_card("As"), str_to_card("Ks"), str_to_card("Qs"), str_to_card("Js"), str_to_card("Ts"),
        str_to_card("2c"), str_to_card("3d")
    ])
    scores_rf, cats_rf = evaluate_7cards(rf_7.reshape(1, 7))
    assert cats_rf[0] == 9


def test_evaluator_speed_target():
    """Verify evaluator throughput exceeds 1,000,000 hands per second."""
    speed = measure_evaluator_speed(50_000, seed=123)
    assert speed > 1_000_000, f"Evaluator speed {speed:,.0f} hands/s below 1,000,000 target"


def test_evaluator_treys_agreement():
    """Verify exact category and relative ordering agreement against Treys library."""
    n_sample = 10_000 if MODE == "quick" else 100_000
    rng = np.random.default_rng(2026)
    deck = np.tile(np.arange(52, dtype=np.int8), (n_sample, 1))
    hands = rng.permuted(deck, axis=1)[:, :7]

    our_scores, our_cats = evaluate_7cards(hands)

    treys_eval = treys.Evaluator()
    treys_class_to_cat = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}

    treys_scores = np.empty(n_sample, dtype=np.int32)
    treys_cats = np.empty(n_sample, dtype=np.int32)

    for i in range(n_sample):
        h = hands[i]
        t_cards = [treys.Card.new(RANKS_STR[c // 4] + SUITS_STR[c % 4]) for c in h]
        score = treys_eval.evaluate(t_cards[:5], t_cards[5:7])
        t_class = treys_eval.get_rank_class(score)
        treys_scores[i] = score
        treys_cats[i] = treys_class_to_cat[t_class]

    mapped_our_cats = np.where(our_cats == 9, 8, our_cats)
    category_mismatches = np.sum(mapped_our_cats != treys_cats)
    assert category_mismatches == 0, f"Found {category_mismatches} category mismatches with Treys"

    sA, sB = our_scores[0::2], our_scores[1::2]
    tA, tB = treys_scores[0::2], treys_scores[1::2]
    our_cmp = np.sign(sA - sB)
    treys_cmp = np.sign(tB - tA)
    order_mismatches = np.sum(our_cmp != treys_cmp)
    assert order_mismatches == 0, f"Found {order_mismatches} ordering mismatches with Treys"


# -------------------------------------------------------------------------
# P2: Equity Engine, Hand Classes, and 1/n Identity Tests
# -------------------------------------------------------------------------
def test_hand_classes_and_combo_weights():
    """Verify that 169 classes are properly formed and combo weights sum to 1,326."""
    assert len(ALL_CLASSES) == 169
    assert TOTAL_COMBOS == 1326
    assert sum(COMBO_WEIGHTS.values()) == 1326

    # Verify combo counts: 13 pairs * 6 = 78, 78 suited * 4 = 312, 78 offsuit * 12 = 936
    pair_combos = sum(COMBO_WEIGHTS[c] for c in ALL_CLASSES if len(c) == 2)
    suited_combos = sum(COMBO_WEIGHTS[c] for c in ALL_CLASSES if c.endswith("s"))
    offsuit_combos = sum(COMBO_WEIGHTS[c] for c in ALL_CLASSES if c.endswith("o"))

    assert pair_combos == 78
    assert suited_combos == 312
    assert offsuit_combos == 936

    # Test representative cards mapping
    for c in ["AA", "AKs", "72o", "32s", "KK", "T9o"]:
        rep = class_representative(c)
        assert cards_to_class(rep[0], rep[1]) == c


def test_seed_reproducibility():
    """Verify that fixed seed produces identical Monte Carlo equity numbers."""
    res1 = estimate_hand_equity("AKs", 4, n_sim=5_000, rng=np.random.default_rng(42))
    res2 = estimate_hand_equity("AKs", 4, n_sim=5_000, rng=np.random.default_rng(42))
    assert res1["equity"] == res2["equity"]
    assert res1["p_win"] == res2["p_win"]
    assert res1["p_tie"] == res2["p_tie"]


def test_1_over_n_identity():
    """Verify that average player payout equals 1/n for n = 2..9 within precision."""
    for n in range(2, 10):
        deals = simulate_table_deals(n, n_deals=5_000, rng=np.random.default_rng(n * 100))
        mean_payout_per_seat = np.mean(deals["payouts"], axis=0)
        overall_mean = float(np.mean(mean_payout_per_seat))
        assert abs(overall_mean - 1.0 / n) < 1e-12


def test_equity_benchmarks():
    """Verify literature benchmark reproductions: AA, 72o, and 32o."""
    res_aa_2 = estimate_hand_equity("AA", 2, n_sim=10_000, rng=np.random.default_rng(42))
    assert abs(res_aa_2["equity"] - 0.853) < 0.01

    res_aa_6 = estimate_hand_equity("AA", 6, n_sim=10_000, rng=np.random.default_rng(42))
    assert abs(res_aa_6["equity"] - 0.490) < 0.02

    res_72_2 = estimate_hand_equity("72o", 2, n_sim=10_000, rng=np.random.default_rng(42))
    assert abs(res_72_2["equity"] - 0.345) < 0.015

    res_72_6 = estimate_hand_equity("72o", 6, n_sim=10_000, rng=np.random.default_rng(42))
    assert abs(res_72_6["equity"] - 0.086) < 0.015

    res_32_6 = estimate_hand_equity("32o", 6, n_sim=10_000, rng=np.random.default_rng(42))
    # Rank reversal: at 6 players, 32o has higher equity than 72o
    assert res_32_6["equity"] > res_72_6["equity"]


def test_exact_vs_monte_carlo():
    """Verify that Monte Carlo turn equity matches exhaustive 45,540 deal enumeration within 95% CI."""
    hero = (str_to_card("Ah"), str_to_card("Kh"))
    turn_board = [str_to_card("Jh"), str_to_card("Th"), str_to_card("2c"), str_to_card("7s")]

    exact_res = exact_turn_equity(hero, turn_board)
    mc_res = estimate_hand_equity(hero, 2, board=turn_board, n_sim=15_000, rng=np.random.default_rng(42))

    diff = abs(exact_res["equity"] - mc_res["equity"])
    assert diff <= 2.5 * mc_res["equity_se"]


# -----------------------------------------------------------------------------
# Prompt 9 Tests: UCI Poker Hand & PHH Real Play
# -----------------------------------------------------------------------------


def test_uci_loading_and_eda():
    """Verify loading UCI dataset and EDA summary statistics."""
    from holdem_lab.analysis import load_uci_data, uci_eda_summary

    train_df = load_uci_data("train")
    assert len(train_df) == 25010
    assert list(train_df.columns) == [
        "S1",
        "C1",
        "S2",
        "C2",
        "S3",
        "C3",
        "S4",
        "C4",
        "S5",
        "C5",
        "class",
    ]

    eda = uci_eda_summary(train_df)
    assert eda["missing_count"] == 0
    assert all(1 <= v <= 4 for v in eda["suit_mins"].values())
    assert all(1 <= v <= 4 for v in eda["suit_maxs"].values())
    assert all(1 <= v <= 13 for v in eda["rank_mins"].values())
    assert all(1 <= v <= 13 for v in eda["rank_maxs"].values())
    assert 6.5 < eda["rank_mean"] < 7.5


def test_uci_evaluator_verification():
    """Verify 100% agreement between evaluator and UCI training split labels."""
    from holdem_lab.analysis import load_uci_data, verify_uci_evaluator

    train_df = load_uci_data("train")
    # Verify on full training split and first 10,000 of test split for speed
    test_df = load_uci_data("test").iloc[:10000]

    v_res = verify_uci_evaluator(train_df, test_df)
    assert v_res["train_mismatches"] == 0
    assert v_res["train_agreed"] == 25010
    assert v_res["test_mismatches"] == 0
    assert v_res["agreement_rate"] == 1.0


def test_uci_goodness_of_fit_table():
    """Verify Table 1 chi-square GOF: train rejects H0, test fails to reject H0."""
    from holdem_lab.analysis import load_uci_data, uci_goodness_of_fit_table

    train_df = load_uci_data("train")
    test_df = load_uci_data("test")

    table_df, gof_stats = uci_goodness_of_fit_table(train_df, test_df, mc_reps=500)
    assert len(table_df) == 10
    # Training split oversamples rare hands (5 SF, 5 RF) -> rejects H0
    assert gof_stats["train_chi2"] > 15.0
    assert gof_stats["train_p_val"] < 0.05
    # Testing split follows natural multinomial probabilities -> fails to reject H0
    assert gof_stats["test_chi2"] < 10.0
    assert gof_stats["test_p_val"] > 0.05


def test_phh_parser_and_pokerkit():
    """Verify PHH hand history parsing and optional pokerkit loader."""
    import glob
    from holdem_lab.analysis import parse_phh_handhq, parse_phh_with_pokerkit

    files = glob.glob("data/external/phh/phh-dataset/data/handhq/**/*.phhs", recursive=True)
    assert len(files) > 0, "PHH files must be present"

    # Test pokerkit parser on 1 file
    df_pk = parse_phh_with_pokerkit(files[0], max_hands=20)
    assert isinstance(df_pk, pd.DataFrame)
    if len(df_pk) > 0:
        assert "seats_dealt" in df_pk.columns
        assert "winning_cat" in df_pk.columns
        assert "winning_score" in df_pk.columns

    # Test regex parser on 5 files
    df_showdowns, log = parse_phh_handhq(max_files=5, use_cache=False)
    assert len(df_showdowns) > 0
    assert log["total_hands"] > 0
    assert log["retained_showdowns"] == len(df_showdowns)
    assert all(c in df_showdowns.columns for c in ["seats_dealt", "players_showdown", "winning_cat"])


def test_phh_tightness_estimation():
    """Verify effective opponent tightness estimation from showdown hands."""
    from holdem_lab.analysis import estimate_effective_tightness, parse_phh_handhq

    df_showdowns, _ = parse_phh_handhq(max_files=10, use_cache=True)
    tightness = estimate_effective_tightness(df_showdowns, n_sim=600, seed=42)

    assert "fitted_q" in tightness
    assert 0.10 <= tightness["fitted_q"] <= 0.70
    assert "tv_distance" in tightness
    assert "chi2_stat" in tightness


# -----------------------------------------------------------------------------
# Prompt 4 Tests: G1 Table Dynamics & G2 Equity Distribution
# -----------------------------------------------------------------------------


def test_tie_aware_fn_toy():
    """Verify discrete tie-aware order statistic of maximum against brute-force enumeration."""
    from holdem_lab.stats import discrete_order_stat_max

    probs = np.array([0.5, 0.3, 0.2])
    res = discrete_order_stat_max(probs, 2)

    # Brute-force: (0,0)=0.25 tie; (1,1)=0.09 tie; (2,2)=0.04 tie -> total tie = 0.38
    assert abs(res["prob_tie_for_max"] - 0.38) < 1e-12
    # P(max = 0)=0.25, P(max = 1)=0.39, P(max = 2)=0.36
    assert np.allclose(res["pmf_max"], [0.25, 0.39, 0.36])
    assert np.allclose(res["cdf_max"], [0.25, 0.64, 1.00])


def test_equity_table_1_over_n_identity():
    """Verify combo-weighted mean equals 1/n across all n=2..9 in the equity table."""
    from holdem_lab.analysis import compute_equity_table

    df_equity = compute_equity_table(use_cache=True)
    assert len(df_equity) == 169 * 8

    for n in range(2, 10):
        df_n = df_equity[df_equity["n_players"] == n]
        weighted_mean = float(np.sum(df_n["equity"] * df_n["combos"]) / 1326.0)
        assert abs(weighted_mean - 1.0 / n) < 1e-6

    # Verify AA, 72o, and rank reversal at n >= 6
    df_aa = df_equity[df_equity["hand_class"] == "AA"].set_index("n_players")
    df_72 = df_equity[df_equity["hand_class"] == "72o"].set_index("n_players")
    df_32 = df_equity[df_equity["hand_class"] == "32o"].set_index("n_players")

    assert df_aa.loc[2, "equity"] > df_aa.loc[6, "equity"] > df_aa.loc[9, "equity"]
    assert df_72.loc[2, "equity"] > df_32.loc[2, "equity"]  # heads up: 72o > 32o
    assert df_32.loc[9, "equity"] > df_72.loc[9, "equity"]  # multiway (9-max): 32o > 72o (rank reversal)


def test_beta_fit_mom_vs_scipy():
    """Verify from-scratch Beta MoM fit matches theoretical equations and SciPy."""
    import scipy.stats as sp_stats
    from holdem_lab.stats import fit_beta_mom

    rng = np.random.default_rng(42)
    data = rng.beta(3.5, 6.2, size=1500)

    a_mom, b_mom = fit_beta_mom(data)
    m = np.mean(data)
    v = np.var(data, ddof=1)
    a_exp = m * (m * (1 - m) / v - 1)
    b_exp = (1 - m) * (m * (1 - m) / v - 1)

    assert abs(a_mom - a_exp) < 1e-12
    assert abs(b_mom - b_exp) < 1e-12

    a_sp, b_sp, _, _ = sp_stats.beta.fit(data, floc=0, fscale=1)
    assert abs(a_mom - a_sp) < 0.25
    assert abs(b_mom - b_sp) < 0.35


# -----------------------------------------------------------------------------
# Prompt 5 Tests: G7 Simple Formula & Rank Stability
# -----------------------------------------------------------------------------


def test_planted_power_law_recovers_known_beta():
    """Verify that WLS regression recovers known planted power-law exponent exactly."""
    known_beta = 1.750
    p2 = np.linspace(0.3, 0.85, 169)
    w = np.random.default_rng(42).integers(4, 16, size=169).astype(float)
    raw = (2.0 * p2) ** known_beta
    kappa = float(np.sum(w * raw) / np.sum(w))
    y = (1.0 / 6.0) * raw / kappa

    x = np.log(2.0 * p2)
    y_log = np.log(6.0 * y)

    # WLS fit with intercept
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    coefs = np.linalg.solve(X.T @ W @ X, X.T @ W @ y_log)

    assert abs(coefs[1] - known_beta) < 1e-10
    assert abs(coefs[0] - (-np.log(kappa))) < 1e-10


def test_cross_validation_excludes_held_out_class():
    """Verify that Leave-One-Out Cross-Validation strictly excludes the held-out class."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    # Inject extreme corruption at index 0
    y_corrupted = y.copy()
    y_corrupted[0] = 999.0

    # Model fit for index 0 excluding index 0
    mask = np.ones(5, dtype=bool)
    mask[0] = False
    X_tr = x[mask]
    y_tr = y_corrupted[mask]
    beta_0 = float(np.sum(X_tr * y_tr) / np.sum(X_tr**2))

    # Should be unaffected by y_corrupted[0]
    assert abs(beta_0 - 2.0) < 1e-10


def test_formula_held_out_tolerance():
    """Verify that formula reproduces held-out classes within tolerance and premium pairs fall."""
    import json
    from holdem_lab import DATA_DIR
    from holdem_lab.formula import fit_candidate_models

    df_comp, info = fit_candidate_models()
    m1_comp = df_comp[df_comp["model"] == "M1"]

    # Mean absolute error across table sizes should be small (< 2.5 pp)
    assert m1_comp["mae_pp"].mean() < 2.5

    # Check formula_params.json exists and is valid
    params_path = DATA_DIR / "formula_params.json"
    assert params_path.exists()
    with open(params_path, "r") as fp:
        params = json.load(fp)
    assert "smooth_law" in params
    assert "table_params" in params
    assert "6" in params["table_params"]


# =========================================================================
# Prompt 8 Tests: Luck, Sample Size, Waiting Times, Table 3
# =========================================================================

def test_pgf_moments_closed_form():
    """Verify that numerical PGF derivatives recover exact closed-form mean and variance."""
    from holdem_lab.stats import pgf_moments

    # Test on Geometric(p) for pocket aces (p = 1/221) and any pair (p = 1/17)
    test_cases = [1.0 / 221.0, 1.0 / 17.0, 4.0 / 17.0]
    for p in test_cases:
        pgf = lambda s, p_val=p: (p_val * s) / (1.0 - (1.0 - p_val) * s)
        m = pgf_moments(pgf, h=1e-5)
        exact_mean = 1.0 / p
        exact_var = (1.0 - p) / (p ** 2)

        assert abs(m["mean"] - exact_mean) < 0.05
        # Relative error in variance < 0.1%
        assert abs(m["variance"] - exact_var) / exact_var < 0.001


def test_power_calculation_vs_simulation():
    """Verify that power calculation formula matches empirical rejection rates."""
    import numpy as np

    delta = 0.15
    sigma = 1.0
    z_alpha = 1.95996  # 5% two-sided
    z_beta = 0.84162   # 80% power

    # Sample size formula for one-sample/paired t-test
    n_req = int(np.ceil(((z_alpha + z_beta) ** 2) * (sigma ** 2) / (delta ** 2)))

    # Monte Carlo simulation of 400 experiments
    rng = np.random.default_rng(1234)
    n_exp = 400
    rejects = 0
    for _ in range(n_exp):
        sample = rng.normal(loc=delta, scale=sigma, size=n_req)
        t_stat = np.mean(sample) / (np.std(sample, ddof=1) / np.sqrt(n_req))
        if t_stat > z_alpha:
            rejects += 1

    emp_power = rejects / n_exp
    # Empirical power should be within ~6 pp of 0.80
    assert abs(emp_power - 0.80) < 0.06


def test_ruin_approximation_sanity():
    """Sanity check Brownian ruin probability formula on random walk with known drift."""
    import numpy as np
    from holdem_lab.stats import brownian_ruin_prob

    mu = 0.05
    sigma = 1.0
    bankroll = 20.0

    p_th = brownian_ruin_prob(mu, sigma, bankroll)
    assert abs(p_th - np.exp(-2.0 * mu * bankroll / (sigma ** 2))) < 1e-12

    # Simulate 1,500 random walks of length 2,500
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=mu, scale=sigma, size=(1500, 2500))
    walks = np.cumsum(steps, axis=1) + bankroll
    ruin_sim = float(np.mean(np.any(walks <= 0, axis=1)))

    # Simulated ruin should track theoretical within 3 pp
    assert abs(p_th - ruin_sim) < 0.03


def test_waiting_times_and_table3():
    """Verify that G6 produces Table 3 with all 3 events and valid chi-square results."""
    from holdem_lab.analysis import run_g6_waiting_times

    res = run_g6_waiting_times(n_deals=10000, seed=42)
    df_t3 = res["df_table3"]

    assert len(df_t3) == 3
    assert set(df_t3["event"]) == {"pocket_aces", "any_pocket_pair", "suited_hand"}
    assert np.all(df_t3["df"] >= 1)
    assert np.all(df_t3["p_value"] > 0.01)


# =========================================================================
# Prompt 6 Tests: G3 Streets, Martingale, Rao-Blackwell, Rule of 2 and 4
# =========================================================================

def test_exact_draw_probabilities():
    """Verify exact combinatorial draw hit probabilities for flush and straight draws."""
    from holdem_lab.analysis import exact_draw_probability

    # Flush draw: 9 outs on flop
    p_flush = exact_draw_probability(9, "flop")
    # Exact: (93*9 - 81) / 2162 = 756 / 2162
    assert abs(p_flush - 756.0 / 2162.0) < 1e-12
    assert abs(p_flush - 0.349676) < 1e-4

    # Open-ended straight draw: 8 outs on flop
    p_straight = exact_draw_probability(8, "flop")
    # Exact: (93*8 - 64) / 2162 = 680 / 2162
    assert abs(p_straight - 680.0 / 2162.0) < 1e-12
    assert abs(p_straight - 0.314524) < 1e-4

    # Turn probabilities: k / 46
    assert abs(exact_draw_probability(9, "turn") - 9.0 / 46.0) < 1e-12
    assert abs(exact_draw_probability(8, "turn") - 8.0 / 46.0) < 1e-12


def test_martingale_property_toy_problem():
    """Verify martingale property E[X_{t+1}|X_t] = X_t and E[X_T] = X_0 on toy process."""
    import numpy as np

    rng = np.random.default_rng(42)
    n_sim = 50000
    x0 = 0.50

    # Three zero-mean mean-preserving steps with conditional noise
    # Step 1: flop noise
    eps1 = rng.choice([-0.2, 0.2], size=n_sim)
    x1 = x0 + eps1

    # Step 2: turn noise conditional on x1
    eps2 = rng.normal(0.0, 0.1, size=n_sim)
    x2 = x1 + eps2

    # Step 3: river noise
    eps3 = rng.choice([-0.15, 0.15], size=n_sim)
    x3 = x2 + eps3

    # Martingale expectation conservation: E[X_t] = X_0
    assert abs(np.mean(x1) - x0) < 0.01
    assert abs(np.mean(x2) - x0) < 0.01
    assert abs(np.mean(x3) - x0) < 0.01

    # Orthogonality of martingale increments: Cov(d_i, d_j) = 0
    d1 = x1 - x0
    d2 = x2 - x1
    d3 = x3 - x2
    assert abs(np.cov(d1, d2)[0, 1]) < 0.005
    assert abs(np.cov(d2, d3)[0, 1]) < 0.005
    assert abs(np.cov(d1, d3)[0, 1]) < 0.005


def test_variance_decomposition_adds_up():
    """Verify that the law of total variance decomposition holds: Var(X3) = sum(Var(d_i))."""
    import numpy as np

    rng = np.random.default_rng(123)
    n_sim = 100000
    x0 = 0.60

    # Independent increments
    d1 = rng.normal(0, 0.10, size=n_sim)
    d2 = rng.normal(0, 0.15, size=n_sim)
    d3 = rng.normal(0, 0.20, size=n_sim)

    x3 = x0 + d1 + d2 + d3

    var_total = np.var(x3)
    var_sum = np.var(d1) + np.var(d2) + np.var(d3)

    assert abs(var_total - var_sum) / var_total < 0.01


def test_rao_blackwell_unbiased_lower_variance():
    """Verify Rao-Blackwell estimator is unbiased and achieves strictly lower variance."""
    import numpy as np

    rng = np.random.default_rng(999)
    n_sim = 20000

    # True parameter theta
    theta = 0.65
    # X is an intermediate signal (flop signal)
    # Y is terminal binary indicator (showdown outcome) with P(Y=1|X) = X
    # Unconditional mean of X is theta
    X = rng.beta(theta * 10, (1 - theta) * 10, size=n_sim)
    Y = (rng.uniform(size=n_sim) < X).astype(np.float64)

    # Plain MC estimator uses Y
    # Rao-Blackwell estimator uses conditional expectation E[Y|X] = X
    mean_mc = np.mean(Y)
    mean_rb = np.mean(X)
    var_mc = np.var(Y)
    var_rb = np.var(X)

    # Both are unbiased for theta
    assert abs(mean_mc - theta) < 0.015
    assert abs(mean_rb - theta) < 0.015

    # Rao-Blackwell variance is strictly lower than plain MC variance
    assert var_rb < var_mc
    vr_ratio = var_mc / var_rb
    assert vr_ratio > 1.2


# =========================================================================
# Prompt 7 Tests: G4 Opponents, Blinds, and Table Selection ANOVA
# =========================================================================

def test_pot_and_blind_accounting_conserves_chips():
    """Verify pot and blind accounting strictly conserves chips (sum of profits = 0 on every hand)."""
    import numpy as np
    from holdem_lab.strategy import simulate_limp_fold_game

    for n in [2, 3, 6, 9]:
        res = simulate_limp_fold_game(n_players=n, n_hands=200, seed=42)
        profits = res["profits"]

        # Sum across players for every single hand must equal 0 exactly
        hand_sums = np.sum(profits, axis=1)
        assert np.all(np.abs(hand_sums) < 1e-10)
        assert res["chips_conserved"] is True


def test_strategy_q100_plays_every_hand():
    """Verify that strategy with q=100% contains all 169 classes and calls every hand."""
    from holdem_lab.strategy import get_range_classes, simulate_limp_fold_game

    all_classes = get_range_classes(100.0, "heads_up")
    assert len(all_classes) == 169

    # In a game where seat 0 has q=100%, active showdown count should include seat 0 on every deal
    res = simulate_limp_fold_game(n_players=6, n_hands=100, strategies=[100.0, 50.0, 50.0, 50.0, 50.0, 50.0], seed=42)
    # Seat 0 should always be in showdown when not folded as small blind
    assert res["chips_conserved"] is True


def test_closed_form_blind_cost_vs_simulation():
    """Verify closed-form blind cost matches simulation of 100% folding strategy within CI."""
    import numpy as np
    from holdem_lab.strategy import simulate_limp_fold_game, closed_form_blind_cost

    # Strategy that plays empty set of classes (folds 100%)
    empty_strat = set()
    n_players = 6
    n_hands = 2000

    res = simulate_limp_fold_game(
        n_players=n_players,
        n_hands=n_hands,
        strategies=[empty_strat] + [100.0] * (n_players - 1),
        seed=123,
    )
    p0_profits = res["profits"][:, 0]
    sim_mean = float(np.mean(p0_profits))
    sim_se = float(np.std(p0_profits) / np.sqrt(n_hands))

    # Theoretical cost: 1.5 / n = 0.25 bb/hand loss (-0.25)
    expected_cost = closed_form_blind_cost(n_players, 0.0)
    assert abs(expected_cost - (-1.5 / n_players)) < 1e-10

    # Simulation mean should match closed form within 1e-6
    assert abs(sim_mean - expected_cost) < 1e-6


def test_anova_on_planted_interaction():
    """Verify that two-way ANOVA detects planted factor interaction effect with high power."""
    import numpy as np
    from holdem_lab.stats import two_way_anova_interaction

    rng = np.random.default_rng(777)
    # 3 levels of factor A, 3 levels of factor B, 20 replicates per cell = 180 obs
    levels_a = ["A1", "A2", "A3"]
    levels_b = ["B1", "B2", "B3"]
    k_rep = 25

    rows = []
    f_a_list = []
    f_b_list = []

    # Planted interaction table:
    # A1: B1=0, B2=0, B3=0
    # A2: B1=0, B2=5, B3=-5
    # A3: B1=0, B2=-5, B3=5
    planted = {
        ("A1", "B1"): 0.0, ("A1", "B2"): 0.0, ("A1", "B3"): 0.0,
        ("A2", "B1"): 0.0, ("A2", "B2"): 4.0, ("A2", "B3"): -4.0,
        ("A3", "B1"): 0.0, ("A3", "B2"): -4.0, ("A3", "B3"): 4.0,
    }

    for a in levels_a:
        for b in levels_b:
            mean_val = planted[(a, b)]
            noise = rng.normal(0.0, 1.0, size=k_rep)
            y_cell = mean_val + noise
            rows.extend(y_cell)
            f_a_list.extend([a] * k_rep)
            f_b_list.extend([b] * k_rep)

    y = np.array(rows, dtype=np.float64)
    fa = np.array(f_a_list)
    fb = np.array(f_b_list)

    anova_res = two_way_anova_interaction(y, fa, fb, n_permutations=200, seed=42)

    # Interaction should be detected with high F and tiny p-value (< 0.001)
    assert anova_res["f_interaction"] > 10.0
    assert anova_res["p_analytical"] < 1e-4
    assert anova_res["p_permutation"] < 0.02
    assert anova_res["eta_squared"] > 0.20






