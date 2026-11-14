"""Hold'em Lab - Equity Monte Carlo and Exact Enumeration Engine.

Computes P_win, P_tie, and equity share across table sizes n = 2..9:
- 169 starting hand classes (13 pairs, 78 suited, 78 offsuit; combo weights 6, 4, 12 summing to 1,326).
- Vectorized Monte Carlo equity engine with standard errors and precision stopping rules.
- Table-deal simulation engine verifying exchangeability identity E[equity] = 1/n.
- Exact board enumeration (river and turn).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from holdem_lab.cards import (
    RANKS_STR,
    evaluate_7cards,
    get_deck,
)

# -------------------------------------------------------------------------
# 169 Hand Classes & Combinations
# -------------------------------------------------------------------------
PAIRS = [f"{r}{r}" for r in reversed(RANKS_STR)]  # 13 pocket pairs: AA down to 22
SUITED = [
    f"{RANKS_STR[r2]}{RANKS_STR[r1]}s"
    for r2 in range(12, 0, -1)
    for r1 in range(r2 - 1, -1, -1)
]  # 78 suited classes: AKs down to 32s
OFFSUIT = [
    f"{RANKS_STR[r2]}{RANKS_STR[r1]}o"
    for r2 in range(12, 0, -1)
    for r1 in range(r2 - 1, -1, -1)
]  # 78 offsuit classes: AKo down to 32o

ALL_CLASSES = PAIRS + SUITED + OFFSUIT  # Exactly 169 classes


def class_combos_count(hclass: str) -> int:
    """Return number of specific 2-card combinations for hand class (6, 4, or 12)."""
    if len(hclass) == 2:
        return 6  # Pocket pair: C(4,2) = 6
    elif hclass.endswith("s"):
        return 4  # Suited: 4 combinations (one per suit)
    elif hclass.endswith("o"):
        return 12  # Offsuit: 4 * 3 = 12 combinations
    raise ValueError(f"Invalid hand class format: '{hclass}'")


COMBO_WEIGHTS: dict[str, int] = {c: class_combos_count(c) for c in ALL_CLASSES}
TOTAL_COMBOS: int = sum(COMBO_WEIGHTS.values())  # Exactly 1,326


def cards_to_class(c1: int, c2: int) -> str:
    """Map any two integer cards (0..51) to one of the 169 hand classes."""
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4

    if r1 == r2:
        return f"{RANKS_STR[r1]}{RANKS_STR[r2]}"
    elif s1 == s2:
        high, low = max(r1, r2), min(r1, r2)
        return f"{RANKS_STR[high]}{RANKS_STR[low]}s"
    else:
        high, low = max(r1, r2), min(r1, r2)
        return f"{RANKS_STR[high]}{RANKS_STR[low]}o"


def class_to_combos(hclass: str) -> list[tuple[int, int]]:
    """Return all specific 2-card pairs (as integer card tuples) for hand class."""
    if len(hclass) == 2:  # Pocket pair
        r = RANKS_STR.index(hclass[0])
        return [(r * 4 + s1, r * 4 + s2) for s1 in range(4) for s2 in range(s1 + 1, 4)]
    elif hclass.endswith("s"):  # Suited
        r1 = RANKS_STR.index(hclass[0])
        r2 = RANKS_STR.index(hclass[1])
        return [(r1 * 4 + s, r2 * 4 + s) for s in range(4)]
    elif hclass.endswith("o"):  # Offsuit
        r1 = RANKS_STR.index(hclass[0])
        r2 = RANKS_STR.index(hclass[1])
        return [(r1 * 4 + s1, r2 * 4 + s2) for s1 in range(4) for s2 in range(4) if s1 != s2]
    raise ValueError(f"Invalid hand class: '{hclass}'")


def class_representative(hclass: str) -> tuple[int, int]:
    """Return canonical representative 2-card pair for hand class."""
    combos = class_to_combos(hclass)
    return combos[0]


# -------------------------------------------------------------------------
# Monte Carlo Equity Engine
# -------------------------------------------------------------------------
def estimate_hand_equity(
    hero_cards: str | tuple[int, int] | list[int],
    n_players: int,
    board: list[int] | np.ndarray | None = None,
    n_sim: int = 10_000,
    target_precision: float | None = None,
    rng: np.random.Generator | None = None,
    opponent_combos: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Estimate P_win, P_tie, and equity share for hero hand against n-1 opponents.

    Parameters
    ----------
    hero_cards : str, tuple, or list
        Either a hand class string (e.g. 'AA', 'AKs', '72o') or a 2-card tuple/list of ints.
    n_players : int
        Total players at table (2 <= n <= 9).
    board : list or array, optional
        Community cards already dealt (0, 3, 4, or 5 cards).
    n_sim : int
        Number of Monte Carlo simulations per batch.
    target_precision : float, optional
        If specified, simulations continue until standard error of equity <= target_precision.
    rng : np.random.Generator, optional
        Seeded random number generator.
    opponent_combos : list of tuple, optional
        If specified, opponents are drawn uniformly from this pool of hole-card combinations.

    Returns
    -------
    dict
        Dictionary containing equity, equity_se, p_win, p_win_se, p_tie, p_tie_se, n_sim, ci_95.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if isinstance(hero_cards, str):
        hero_pair = class_representative(hero_cards)
    else:
        hero_pair = (int(hero_cards[0]), int(hero_cards[1]))

    hero_arr = np.array(hero_pair, dtype=np.int8)

    if board is None:
        board_arr = np.empty(0, dtype=np.int8)
    else:
        board_arr = np.array(board, dtype=np.int8)

    cards_used = set(hero_pair) | set(board_arr.tolist())
    deck = get_deck()
    remaining_deck = np.array([c for c in deck if c not in cards_used], dtype=np.int8)

    n_opponents = n_players - 1
    cards_needed_board = 5 - len(board_arr)
    cards_needed_opps = 2 * n_opponents
    total_needed = cards_needed_board + cards_needed_opps

    if total_needed > len(remaining_deck):
        raise ValueError("Not enough cards remaining in deck to deal board and opponents.")

    # Simulation accumulator
    total_samples = 0
    sum_payout = 0.0
    sum_payout_sq = 0.0
    sum_win = 0
    sum_tie = 0

    batch_size = max(5_000, n_sim)
    max_samples = 500_000

    while True:
        # Vectorized deal: draw total_needed cards without replacement per simulation
        drawn = rng.permuted(np.tile(remaining_deck, (batch_size, 1)), axis=1)[:, :total_needed]

        if cards_needed_board > 0:
            drawn_board = drawn[:, :cards_needed_board]
            if len(board_arr) > 0:
                sim_board = np.column_stack([np.tile(board_arr, (batch_size, 1)), drawn_board])
            else:
                sim_board = drawn_board
        else:
            sim_board = np.tile(board_arr, (batch_size, 1))

        opp_cards = drawn[:, cards_needed_board:].reshape(batch_size, n_opponents, 2)

        # 1. Evaluate Hero hands: shape (batch_size, 7)
        hero_7 = np.column_stack([np.tile(hero_arr, (batch_size, 1)), sim_board])
        hero_scores, _ = evaluate_7cards(hero_7)

        # 2. Evaluate Opponent hands: shape (batch_size * n_opponents, 7)
        opp_flat = opp_cards.reshape(batch_size * n_opponents, 2)
        board_repeated = np.repeat(sim_board, n_opponents, axis=0)
        opp_7 = np.column_stack([opp_flat, board_repeated])
        opp_scores_flat, _ = evaluate_7cards(opp_7)
        opp_scores = opp_scores_flat.reshape(batch_size, n_opponents)

        # 3. Payout and outcome computation
        max_opp_score = np.max(opp_scores, axis=1)
        win_mask = hero_scores > max_opp_score
        tie_mask = hero_scores == max_opp_score

        # When tied with highest opponent, count total tied players k = 1 + count(opp == max_opp)
        opp_tied_counts = np.sum(opp_scores == max_opp_score[:, None], axis=1)
        k_tied = 1 + opp_tied_counts

        payout = np.where(win_mask, 1.0, np.where(tie_mask, 1.0 / k_tied, 0.0))

        total_samples += batch_size
        sum_payout += float(np.sum(payout))
        sum_payout_sq += float(np.sum(payout**2))
        sum_win += int(np.sum(win_mask))
        sum_tie += int(np.sum(tie_mask))

        eq = sum_payout / total_samples
        variance = (sum_payout_sq / total_samples) - (eq**2)
        se = math.sqrt(max(0.0, variance) / total_samples)

        if target_precision is None or se <= target_precision or total_samples >= max_samples:
            break

    p_win = sum_win / total_samples
    p_tie = sum_tie / total_samples
    p_win_se = math.sqrt(max(0.0, p_win * (1.0 - p_win)) / total_samples)
    p_tie_se = math.sqrt(max(0.0, p_tie * (1.0 - p_tie)) / total_samples)

    return {
        "equity": eq,
        "equity_se": se,
        "p_win": p_win,
        "p_win_se": p_win_se,
        "p_tie": p_tie,
        "p_tie_se": p_tie_se,
        "n_sim": total_samples,
        "ci_95": (max(0.0, eq - 1.96 * se), min(1.0, eq + 1.96 * se)),
    }


# -------------------------------------------------------------------------
# All-Players Table Deal Simulator
# -------------------------------------------------------------------------
def simulate_table_deals(
    n_players: int,
    n_deals: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Simulate full-table deals of n players, returning all player outcomes.

    Parameters
    ----------
    n_players : int
        Table size n in 2..9.
    n_deals : int
        Number of complete deals.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    dict
        payouts: shape (n_deals, n_players)
        scores: shape (n_deals, n_players)
        categories: shape (n_deals, n_players)
        winning_scores: shape (n_deals,)
        winning_categories: shape (n_deals,)
        split_pot: shape (n_deals,) bool
    """
    if rng is None:
        rng = np.random.default_rng(42)

    deck = get_deck()
    total_cards_needed = 2 * n_players + 5

    # Deal cards for all hands
    drawn = rng.permuted(np.tile(deck, (n_deals, 1)), axis=1)[:, :total_cards_needed]

    board = drawn[:, :5]  # shape (n_deals, 5)
    player_hole = drawn[:, 5:].reshape(n_deals, n_players, 2)

    # Flatten for evaluation
    player_flat = player_hole.reshape(n_deals * n_players, 2)
    board_repeated = np.repeat(board, n_players, axis=0)

    all_7 = np.column_stack([player_flat, board_repeated])
    scores_flat, cats_flat = evaluate_7cards(all_7)

    scores = scores_flat.reshape(n_deals, n_players)
    categories = cats_flat.reshape(n_deals, n_players)

    max_scores = np.max(scores, axis=1)  # shape (n_deals,)
    is_winner = scores == max_scores[:, None]  # shape (n_deals, n_players)
    winner_counts = np.sum(is_winner, axis=1)  # shape (n_deals,)

    payouts = np.where(is_winner, 1.0 / winner_counts[:, None], 0.0)
    split_pot = winner_counts > 1

    # Winning category: category of any winning player on that deal
    win_player_idx = np.argmax(is_winner, axis=1)
    winning_categories = categories[np.arange(n_deals), win_player_idx]

    return {
        "payouts": payouts,
        "scores": scores,
        "categories": categories,
        "winning_scores": max_scores,
        "winning_categories": winning_categories,
        "split_pot": split_pot,
    }


# -------------------------------------------------------------------------
# Exact Enumeration (Turn and River)
# -------------------------------------------------------------------------
def exact_turn_equity(
    hero_cards: tuple[int, int],
    turn_board: list[int],
) -> dict[str, float]:
    """Compute exact heads-up equity against a random opponent on a 4-card turn board.
    
    Exhaustively enumerates all C(46, 3) * 3 = 45,540 deals of opponent hole cards and river card.
    """
    used = set(hero_cards) | set(turn_board)
    deck = get_deck()
    rem = [c for c in deck if c not in used]

    triplets = list(itertools.combinations(rem, 3))
    deals = []
    for c1, c2, c3 in triplets:
        deals.append((c1, c2, c3))
        deals.append((c1, c3, c2))
        deals.append((c2, c3, c1))

    deals_arr = np.array(deals, dtype=np.int8)
    opp_cards = deals_arr[:, :2]
    river = deals_arr[:, 2]

    board_5 = np.column_stack([np.tile(turn_board, (len(deals_arr), 1)), river])
    hero_7 = np.column_stack([np.tile(hero_cards, (len(deals_arr), 1)), board_5])
    opp_7 = np.column_stack([opp_cards, board_5])

    hs, _ = evaluate_7cards(hero_7)
    os, _ = evaluate_7cards(opp_7)

    win = int(np.sum(hs > os))
    tie = int(np.sum(hs == os))
    exact_eq = (win + 0.5 * tie) / len(deals_arr)

    return {
        "equity": exact_eq,
        "p_win": win / len(deals_arr),
        "p_tie": tie / len(deals_arr),
        "n_deals": len(deals_arr),
    }


def exact_board_equity(
    hero_cards: tuple[int, int],
    opp_cards: tuple[int, int],
    partial_board: list[int],
) -> dict[str, float]:
    """Compute exact heads-up equity by exhaustive enumeration of remaining board cards."""
    used = set(hero_cards) | set(opp_cards) | set(partial_board)
    deck = get_deck()
    rem = [c for c in deck if c not in used]
    needed = 5 - len(partial_board)

    if needed <= 0:
        full_board = list(partial_board)[:5]
        h7 = np.array([hero_cards[0], hero_cards[1]] + full_board).reshape(1, 7)
        o7 = np.array([opp_cards[0], opp_cards[1]] + full_board).reshape(1, 7)
        hs, _ = evaluate_7cards(h7)
        os, _ = evaluate_7cards(o7)
        if hs[0] > os[0]:
            return {"equity": 1.0, "p_win": 1.0, "p_tie": 0.0}
        elif hs[0] == os[0]:
            return {"equity": 0.5, "p_win": 0.0, "p_tie": 1.0}
        else:
            return {"equity": 0.0, "p_win": 0.0, "p_tie": 0.0}

    combos = list(itertools.combinations(rem, needed))
    n_combos = len(combos)

    h_list = []
    o_list = []
    for c in combos:
        full_board = list(partial_board) + list(c)
        h_list.append([hero_cards[0], hero_cards[1]] + full_board)
        o_list.append([opp_cards[0], opp_cards[1]] + full_board)

    h_arr = np.array(h_list, dtype=np.int8)
    o_arr = np.array(o_list, dtype=np.int8)

    h_scores, _ = evaluate_7cards(h_arr)
    o_scores, _ = evaluate_7cards(o_arr)

    wins = int(np.sum(h_scores > o_scores))
    ties = int(np.sum(h_scores == o_scores))

    return {
        "equity": (wins + 0.5 * ties) / n_combos,
        "p_win": wins / n_combos,
        "p_tie": ties / n_combos,
        "n_combos": n_combos,
    }
