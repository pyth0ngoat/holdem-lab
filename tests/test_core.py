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
    """Verify that write_findings generates reports/findings.md and is idempotent."""
    md1 = write_findings()
    findings_md_file = REPORTS_DIR / "findings.md"
    assert findings_md_file.exists()

    md2 = write_findings()
    assert md1 == md2


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
