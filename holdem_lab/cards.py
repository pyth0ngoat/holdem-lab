"""Hold'em Lab - Card Encoding, Deck Manipulation, and Vectorized Hand Evaluator.

Pure NumPy and standard library implementation.
Evaluates 5-card and 7-card hands into:
- Category (0 to 9): High Card, One Pair, Two Pair, Three of a Kind,
  Straight, Flush, Full House, Four of a Kind, Straight Flush, Royal Flush.
- Integer strength score (strictly monotonic: higher is better, ties equal).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Card constants
RANKS_STR = "23456789TJQKA"
SUITS_STR = "cdhs"  # Clubs=0, Diamonds=1, Hearts=2, Spades=3

RANK_2, RANK_3, RANK_4, RANK_5, RANK_6 = 0, 1, 2, 3, 4
RANK_7, RANK_8, RANK_9, RANK_T, RANK_J = 5, 6, 7, 8, 9
RANK_Q, RANK_K, RANK_A = 10, 11, 12

SUIT_C, SUIT_D, SUIT_H, SUIT_S = 0, 1, 2, 3

CATEGORY_NAMES = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",
]

BASE = 15  # Base 15 radix for non-overlapping kicker scoring

# -------------------------------------------------------------------------
# Precomputed Lookup Tables
# -------------------------------------------------------------------------
# Straight lookup table for 13-bit rank masks (0..8191)
# Returns high rank of straight (3 for 5-high wheel A-2-3-4-5, up to 12 for A-high), or 0 if none.
STRAIGHT_LOOKUP = np.zeros(8192, dtype=np.int32)
for _m in range(8192):
    _p = [(_m & (1 << _r)) != 0 for _r in range(13)]
    _top = 0
    # Wheel: A, 2, 3, 4, 5
    if _p[12] and _p[0] and _p[1] and _p[2] and _p[3]:
        _top = 3  # 5-high straight
    for _s in range(9):
        if _p[_s] and _p[_s + 1] and _p[_s + 2] and _p[_s + 3] and _p[_s + 4]:
            _top = _s + 4
    STRAIGHT_LOOKUP[_m] = _top

# Flush lookup table for 13-bit rank masks with >= 5 cards of the same suit
FLUSH_LOOKUP_CAT = np.full(8192, -1, dtype=np.int32)
FLUSH_LOOKUP_SCORE = np.full(8192, -1, dtype=np.int32)

for _m in range(8192):
    _top = int(STRAIGHT_LOOKUP[_m])
    _ranks_in_mask = [_r for _r in range(12, -1, -1) if (_m & (1 << _r))]
    if len(_ranks_in_mask) >= 5:
        if _top > 0:
            if _top == 12:
                _cat = 9  # Royal Flush
                _score = 9 * (BASE**5) + 12
            else:
                _cat = 8  # Straight Flush
                _score = 8 * (BASE**5) + _top
        else:
            _cat = 5  # Flush
            _top5 = _ranks_in_mask[:5]
            _tb = (
                _top5[0] * (BASE**4)
                + _top5[1] * (BASE**3)
                + _top5[2] * (BASE**2)
                + _top5[3] * BASE
                + _top5[4]
            )
            _score = 5 * (BASE**5) + _tb
        FLUSH_LOOKUP_CAT[_m] = _cat
        FLUSH_LOOKUP_SCORE[_m] = _score


# -------------------------------------------------------------------------
# Card and Deck Functions
# -------------------------------------------------------------------------
def get_deck() -> np.ndarray:
    """Return standard 52-card deck as integers 0..51."""
    return np.arange(52, dtype=np.int8)


def card_to_str(c: int) -> str:
    """Convert integer card (0..51) to two-character string (e.g. 'As', 'Td')."""
    return RANKS_STR[c // 4] + SUITS_STR[c % 4]


def str_to_card(s: str) -> int:
    """Convert two-character string (e.g. 'As', 'Td') to integer card (0..51)."""
    s = s.strip()
    r = RANKS_STR.index(s[0].upper())
    suit_char = s[1].lower()
    su = SUITS_STR.index(suit_char)
    return r * 4 + su


def hand_to_str(cards: Union[List[int], np.ndarray]) -> str:
    """Convert sequence of integer cards to space-separated string representation."""
    return " ".join(card_to_str(int(c)) for c in cards)


# -------------------------------------------------------------------------
# Exact Counts and Probabilities
# -------------------------------------------------------------------------
EXACT_5CARD_COUNTS: Dict[int, int] = {
    0: 1_302_540,  # High card
    1: 1_098_240,  # One pair
    2: 123_552,    # Two pair
    3: 54_912,     # Three of a kind
    4: 10_200,     # Straight
    5: 5_108,      # Flush
    6: 3_744,      # Full house
    7: 624,        # Four of a kind
    8: 36,         # Straight flush (excluding royal flush)
    9: 4,          # Royal flush
}
TOTAL_5CARD = 2_598_960
EXACT_5CARD_PROBS: Dict[int, float] = {k: v / TOTAL_5CARD for k, v in EXACT_5CARD_COUNTS.items()}

EXACT_7CARD_COUNTS: Dict[int, int] = {
    0: 23_294_460,  # High card
    1: 58_627_800,  # One pair
    2: 31_433_400,  # Two pair
    3: 6_461_620,   # Three of a kind
    4: 6_180_020,   # Straight
    5: 4_047_644,   # Flush
    6: 3_473_184,   # Full house
    7: 224_848,     # Four of a kind
    8: 37_260,      # Straight flush (excluding royal flush)
    9: 4_324,       # Royal flush
}
TOTAL_7CARD = 133_784_560
EXACT_7CARD_PROBS: Dict[int, float] = {k: v / TOTAL_7CARD for k, v in EXACT_7CARD_COUNTS.items()}

# Benchmark analytical probabilities
P_SET_ON_FLOP = 1.0 - math.comb(48, 3) / math.comb(50, 3)  # 2304 / 19600 ≈ 0.117551 (1 in 8.51)
P_FLUSH_BY_RIVER_FROM_FLOP_DRAW = 1.0 - math.comb(38, 2) / math.comb(47, 2)  # 378 / 1081 ≈ 0.349676
P_7CARD_ROYAL_FLUSH = 4_324 / TOTAL_7CARD  # 1 in 30,940


# -------------------------------------------------------------------------
# Vectorized Hand Evaluators (NumPy Only)
# -------------------------------------------------------------------------
def evaluate_5cards(hands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an array of 5-card hands in NumPy.

    Parameters
    ----------
    hands : np.ndarray
        Array of shape (N, 5) with card integers in 0..51.

    Returns
    -------
    scores : np.ndarray
        Array of shape (N,) with integer scores (higher is better, ties equal).
    categories : np.ndarray
        Array of shape (N,) with integer categories 0..9.
    """
    if hands.ndim == 1:
        hands = hands.reshape(1, -1)
    N = hands.shape[0]

    ranks = (hands // 4).astype(np.int32)
    suits = (hands % 4).astype(np.int32)

    scores = np.zeros(N, dtype=np.int32)
    categories = np.zeros(N, dtype=np.int32)

    is_flush = (
        (suits[:, 0] == suits[:, 1])
        & (suits[:, 1] == suits[:, 2])
        & (suits[:, 2] == suits[:, 3])
        & (suits[:, 3] == suits[:, 4])
    )
    rank_mask = np.bitwise_or.reduce(1 << ranks, axis=1)
    straight_top = STRAIGHT_LOOKUP[rank_mask]

    counts = np.zeros((N, 13), dtype=np.int32)
    rows = np.arange(N)
    for i in range(5):
        counts[rows, ranks[:, i]] += 1

    c4 = counts == 4
    c3 = counts == 3
    c2 = counts == 2
    has_4 = np.any(c4, axis=1)
    num_3 = np.sum(c3, axis=1)
    num_2 = np.sum(c2, axis=1)

    # 1. Flush hands
    f_idx = np.where(is_flush)[0]
    if len(f_idx) > 0:
        f_top = straight_top[f_idx]
        rf = f_idx[f_top == 12]
        sf = f_idx[(f_top > 0) & (f_top < 12)]
        fl = f_idx[f_top == 0]

        categories[rf] = 9
        scores[rf] = 9 * (BASE**5) + 12

        categories[sf] = 8
        scores[sf] = 8 * (BASE**5) + straight_top[sf]

        categories[fl] = 5
        sr = np.sort(ranks[fl], axis=1)[:, ::-1]
        scores[fl] = (
            5 * (BASE**5)
            + sr[:, 0] * (BASE**4)
            + sr[:, 1] * (BASE**3)
            + sr[:, 2] * (BASE**2)
            + sr[:, 3] * BASE
            + sr[:, 4]
        )

    # 2. Non-flush hands
    nf_idx = np.where(~is_flush)[0]
    if len(nf_idx) > 0:
        # Four of a kind
        idx_4 = nf_idx[has_4[nf_idx]]
        if len(idx_4) > 0:
            q = np.argmax(c4[idx_4], axis=1)
            k = np.argmax(counts[idx_4] == 1, axis=1)
            categories[idx_4] = 7
            scores[idx_4] = 7 * (BASE**5) + q * BASE + k

        # Full house
        fh_mask = (num_3 == 1) & (num_2 == 1)
        idx_fh = nf_idx[fh_mask[nf_idx]]
        if len(idx_fh) > 0:
            t = np.argmax(c3[idx_fh], axis=1)
            p = np.argmax(c2[idx_fh], axis=1)
            categories[idx_fh] = 6
            scores[idx_fh] = 6 * (BASE**5) + t * BASE + p

        # Straight
        str_mask = (~has_4) & (~fh_mask) & (straight_top > 0)
        idx_str = nf_idx[str_mask[nf_idx]]
        if len(idx_str) > 0:
            categories[idx_str] = 4
            scores[idx_str] = 4 * (BASE**5) + straight_top[idx_str]

        # Three of a kind
        t3_mask = (~has_4) & (~fh_mask) & (~str_mask) & (num_3 == 1)
        idx_3 = nf_idx[t3_mask[nf_idx]]
        if len(idx_3) > 0:
            t = np.argmax(c3[idx_3], axis=1)
            c_rem = counts[idx_3].copy()
            c_rem[np.arange(len(idx_3)), t] = 0
            k0 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_3)), k0] -= 1
            k1 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            categories[idx_3] = 3
            scores[idx_3] = 3 * (BASE**5) + t * (BASE**2) + k0 * BASE + k1

        # Two pair
        tp_mask = (~has_4) & (~fh_mask) & (~str_mask) & (~t3_mask) & (num_2 == 2)
        idx_2p = nf_idx[tp_mask[nf_idx]]
        if len(idx_2p) > 0:
            c_pairs = counts[idx_2p] == 2
            p1 = 12 - np.argmax(c_pairs[:, ::-1], axis=1)
            c_pairs_rem = c_pairs.copy()
            c_pairs_rem[np.arange(len(idx_2p)), p1] = False
            p0 = 12 - np.argmax(c_pairs_rem[:, ::-1], axis=1)
            k = np.argmax(counts[idx_2p] == 1, axis=1)
            categories[idx_2p] = 2
            scores[idx_2p] = 2 * (BASE**5) + p1 * (BASE**2) + p0 * BASE + k

        # One pair
        p1_mask = (~has_4) & (~fh_mask) & (~str_mask) & (~t3_mask) & (~tp_mask) & (num_2 == 1)
        idx_1p = nf_idx[p1_mask[nf_idx]]
        if len(idx_1p) > 0:
            p = np.argmax(c2[idx_1p], axis=1)
            c_rem = counts[idx_1p].copy()
            c_rem[np.arange(len(idx_1p)), p] = 0
            k0 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_1p)), k0] -= 1
            k1 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_1p)), k1] -= 1
            k2 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            categories[idx_1p] = 1
            scores[idx_1p] = 1 * (BASE**5) + p * (BASE**3) + k0 * (BASE**2) + k1 * BASE + k2

        # High card
        hc_mask = (~has_4) & (~fh_mask) & (~str_mask) & (~t3_mask) & (~tp_mask) & (~p1_mask)
        idx_hc = nf_idx[hc_mask[nf_idx]]
        if len(idx_hc) > 0:
            sr = np.sort(ranks[idx_hc], axis=1)[:, ::-1]
            categories[idx_hc] = 0
            scores[idx_hc] = (
                sr[:, 0] * (BASE**4)
                + sr[:, 1] * (BASE**3)
                + sr[:, 2] * (BASE**2)
                + sr[:, 3] * BASE
                + sr[:, 4]
            )

    return scores, categories


def evaluate_7cards(hands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an array of 7-card hands in NumPy.

    Parameters
    ----------
    hands : np.ndarray
        Array of shape (N, 7) with card integers in 0..51.

    Returns
    -------
    scores : np.ndarray
        Array of shape (N,) with integer scores (higher is better, ties equal).
    categories : np.ndarray
        Array of shape (N,) with integer categories 0..9.
    """
    if hands.ndim == 1:
        hands = hands.reshape(1, -1)
    N = hands.shape[0]

    ranks = (hands // 4).astype(np.int32)
    suits = (hands % 4).astype(np.int32)

    scores = np.zeros(N, dtype=np.int32)
    categories = np.zeros(N, dtype=np.int32)

    s0 = np.sum(suits == 0, axis=1)
    s1 = np.sum(suits == 1, axis=1)
    s2 = np.sum(suits == 2, axis=1)
    s3 = np.sum(suits == 3, axis=1)

    has_flush = (s0 >= 5) | (s1 >= 5) | (s2 >= 5) | (s3 >= 5)
    flush_idx = np.where(has_flush)[0]

    # 1. Hands containing a flush suit (>= 5 cards of one suit)
    if len(flush_idx) > 0:
        f_ranks = ranks[flush_idx]
        f_suits = suits[flush_idx]
        f_s0 = s0[flush_idx]
        f_s1 = s1[flush_idx]
        f_s2 = s2[flush_idx]
        flush_suit = np.where(f_s0 >= 5, 0, np.where(f_s1 >= 5, 1, np.where(f_s2 >= 5, 2, 3)))
        suit_matches = f_suits == flush_suit[:, None]
        f_masks = np.bitwise_or.reduce(np.where(suit_matches, 1 << f_ranks, 0), axis=1)

        scores[flush_idx] = FLUSH_LOOKUP_SCORE[f_masks]
        categories[flush_idx] = FLUSH_LOOKUP_CAT[f_masks]

    # 2. Hands without a flush suit
    nf_idx = np.where(~has_flush)[0]
    if len(nf_idx) > 0:
        nf_ranks = ranks[nf_idx]
        M = len(nf_idx)
        counts = np.zeros((M, 13), dtype=np.int32)
        rows = np.arange(M)
        for i in range(7):
            counts[rows, nf_ranks[:, i]] += 1

        rank_mask = np.bitwise_or.reduce(1 << nf_ranks, axis=1)
        straight_top = STRAIGHT_LOOKUP[rank_mask]

        c4 = counts == 4
        c3 = counts == 3
        c2 = counts == 2

        has_4 = np.any(c4, axis=1)
        num_3 = np.sum(c3, axis=1)
        num_2 = np.sum(c2, axis=1)
        has_fh = (num_3 >= 2) | ((num_3 == 1) & (num_2 >= 1))

        # Category 7: Four of a kind
        idx_4 = np.where(has_4)[0]
        if len(idx_4) > 0:
            q = np.argmax(c4[idx_4], axis=1)
            c_rem = counts[idx_4].copy()
            c_rem[np.arange(len(idx_4)), q] = 0
            k = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            scores[nf_idx[idx_4]] = 7 * (BASE**5) + q * BASE + k
            categories[nf_idx[idx_4]] = 7

        # Category 6: Full House
        idx_fh = np.where((~has_4) & has_fh)[0]
        if len(idx_fh) > 0:
            fh_c3 = c3[idx_fh]
            t = 12 - np.argmax(fh_c3[:, ::-1], axis=1)
            c_pair = counts[idx_fh].copy()
            c_pair[np.arange(len(idx_fh)), t] = 0
            p = 12 - np.argmax((c_pair[:, ::-1] >= 2), axis=1)
            scores[nf_idx[idx_fh]] = 6 * (BASE**5) + t * BASE + p
            categories[nf_idx[idx_fh]] = 6

        # Category 4: Straight
        has_str = (~has_4) & (~has_fh) & (straight_top > 0)
        idx_str = np.where(has_str)[0]
        if len(idx_str) > 0:
            scores[nf_idx[idx_str]] = 4 * (BASE**5) + straight_top[idx_str]
            categories[nf_idx[idx_str]] = 4

        # Category 3: Three of a kind
        has_3 = (~has_4) & (~has_fh) & (~has_str) & (num_3 == 1)
        idx_3 = np.where(has_3)[0]
        if len(idx_3) > 0:
            t = np.argmax(c3[idx_3], axis=1)
            c_rem = counts[idx_3].copy()
            c_rem[np.arange(len(idx_3)), t] = 0
            k0 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_3)), k0] -= 1
            k1 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            scores[nf_idx[idx_3]] = 3 * (BASE**5) + t * (BASE**2) + k0 * BASE + k1
            categories[nf_idx[idx_3]] = 3

        # Category 2: Two Pair
        has_2p = (~has_4) & (~has_fh) & (~has_str) & (~has_3) & (num_2 >= 2)
        idx_2p = np.where(has_2p)[0]
        if len(idx_2p) > 0:
            c_pairs = counts[idx_2p] >= 2
            p1 = 12 - np.argmax(c_pairs[:, ::-1], axis=1)
            c_pairs_rem = c_pairs.copy()
            c_pairs_rem[np.arange(len(idx_2p)), p1] = False
            p0 = 12 - np.argmax(c_pairs_rem[:, ::-1], axis=1)
            c_rem = counts[idx_2p].copy()
            c_rem[np.arange(len(idx_2p)), p1] = 0
            c_rem[np.arange(len(idx_2p)), p0] = 0
            k = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            scores[nf_idx[idx_2p]] = 2 * (BASE**5) + p1 * (BASE**2) + p0 * BASE + k
            categories[nf_idx[idx_2p]] = 2

        # Category 1: One Pair
        has_1p = (~has_4) & (~has_fh) & (~has_str) & (~has_3) & (~has_2p) & (num_2 == 1)
        idx_1p = np.where(has_1p)[0]
        if len(idx_1p) > 0:
            p = np.argmax(c2[idx_1p], axis=1)
            c_rem = counts[idx_1p].copy()
            c_rem[np.arange(len(idx_1p)), p] = 0
            k0 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_1p)), k0] -= 1
            k1 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_1p)), k1] -= 1
            k2 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            scores[nf_idx[idx_1p]] = 1 * (BASE**5) + p * (BASE**3) + k0 * (BASE**2) + k1 * BASE + k2
            categories[nf_idx[idx_1p]] = 1

        # Category 0: High Card
        has_hc = (~has_4) & (~has_fh) & (~has_str) & (~has_3) & (~has_2p) & (~has_1p)
        idx_hc = np.where(has_hc)[0]
        if len(idx_hc) > 0:
            c_rem = counts[idx_hc].copy()
            k0 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_hc)), k0] -= 1
            k1 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_hc)), k1] -= 1
            k2 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_hc)), k2] -= 1
            k3 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            c_rem[np.arange(len(idx_hc)), k3] -= 1
            k4 = 12 - np.argmax(c_rem[:, ::-1] > 0, axis=1)
            scores[nf_idx[idx_hc]] = (
                k0 * (BASE**4) + k1 * (BASE**3) + k2 * (BASE**2) + k3 * BASE + k4
            )
            categories[nf_idx[idx_hc]] = 0

    return scores, categories


def evaluate_hands(hands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate hands array with either 5 or 7 cards."""
    if hands.shape[-1] == 7:
        return evaluate_7cards(hands)
    elif hands.shape[-1] == 5:
        return evaluate_5cards(hands)
    else:
        raise ValueError(f"Hand array must have 5 or 7 cards per hand, got {hands.shape[-1]}")


def measure_evaluator_speed(n_hands: int = 100_000, seed: int = 42) -> float:
    """Measure hand evaluation speed in hands per second.

    Returns
    -------
    speed : float
        Hands per second.
    """
    rng = np.random.default_rng(seed)
    deck = np.tile(np.arange(52, dtype=np.int8), (n_hands, 1))
    hands = rng.permuted(deck, axis=1)[:, :7]

    t0 = time.perf_counter()
    evaluate_7cards(hands)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    return n_hands / elapsed if elapsed > 0 else 0.0
