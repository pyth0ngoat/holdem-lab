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
    # 1. Set on the flop with a pocket pair = 1 - C(48,3)/C(50,3)
    expected_set = 1.0 - math.comb(48, 3) / math.comb(50, 3)
    assert abs(P_SET_ON_FLOP - expected_set) < 1e-12
    assert abs(1.0 / P_SET_ON_FLOP - 8.506944) < 1e-3

    # 2. Flush by river from 4 suited cards on flop = 1 - C(38,2)/C(47,2) ≈ 34.97%
    expected_flush = 1.0 - math.comb(38, 2) / math.comb(47, 2)
    assert abs(P_FLUSH_BY_RIVER_FROM_FLOP_DRAW - expected_flush) < 1e-12
    assert abs(P_FLUSH_BY_RIVER_FROM_FLOP_DRAW - 0.349676) < 1e-4

    # 3. Seven-card royal flush = 4,324 / 133,784,560 (1 in 30,940)
    expected_rf = 4324 / 133_784_560
    assert abs(P_7CARD_ROYAL_FLUSH - expected_rf) < 1e-12
    assert abs(1.0 / P_7CARD_ROYAL_FLUSH - 30940.0) < 0.1


def test_evaluator_wheel_straight_and_royal_flush():
    """Test straight wheel (A-2-3-4-5) and Royal Flush classification."""
    # Wheel straight: 5c, 4d, 3h, 2s, Ac
    wheel_cards = np.array([str_to_card("5c"), str_to_card("4d"), str_to_card("3h"), str_to_card("2s"), str_to_card("Ac")])
    scores_5, cats_5 = evaluate_5cards(wheel_cards.reshape(1, 5))
    assert cats_5[0] == 4  # Straight
    # Compare with 6-high straight: 6c, 5d, 4h, 3s, 2c
    six_straight = np.array([str_to_card("6c"), str_to_card("5d"), str_to_card("4h"), str_to_card("3s"), str_to_card("2c")])
    scores_6, cats_6 = evaluate_5cards(six_straight.reshape(1, 5))
    assert cats_6[0] == 4
    assert scores_6[0] > scores_5[0]  # 6-high strictly beats 5-high wheel

    # Royal Flush in 7 cards: As Ks Qs Js Ts 2c 3d
    rf_7 = np.array([
        str_to_card("As"), str_to_card("Ks"), str_to_card("Qs"), str_to_card("Js"), str_to_card("Ts"),
        str_to_card("2c"), str_to_card("3d")
    ])
    scores_rf, cats_rf = evaluate_7cards(rf_7.reshape(1, 7))
    assert cats_rf[0] == 9  # Royal Flush


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

    # Map our Royal Flush (9) to 8 for comparison with Treys Straight Flush class
    mapped_our_cats = np.where(our_cats == 9, 8, our_cats)
    category_mismatches = np.sum(mapped_our_cats != treys_cats)
    assert category_mismatches == 0, f"Found {category_mismatches} category mismatches with Treys"

    # Pairwise ordering comparison
    sA, sB = our_scores[0::2], our_scores[1::2]
    tA, tB = treys_scores[0::2], treys_scores[1::2]
    our_cmp = np.sign(sA - sB)
    treys_cmp = np.sign(tB - tA)  # Treys is lower-is-better
    order_mismatches = np.sum(our_cmp != treys_cmp)
    assert order_mismatches == 0, f"Found {order_mismatches} ordering mismatches with Treys"
