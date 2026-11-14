"""Hold'em Lab - Opponent ranges, limp-or-fold showdown game, and table selection (G4, G8).

Contains the range model for opponents, archetype ranges (Rock, TAG, LAG, Calling Station, All-Hands),
the limp-or-fold showdown game with blinds, closed-form blind-cost break-even models,
and table selection ANOVA routines.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import pandas as pd
from holdem_lab import DATA_DIR, FIGURES_DATA_DIR
from holdem_lab.cards import evaluate_7cards, RANKS_STR, SUITS_STR


# =========================================================================
# 1. 169 Hand Class Mapping & Card Helpers
# =========================================================================

# Fast 52x52 lookup table for (c1, c2) -> hand class string
_CARD_PAIR_TO_CLASS: Optional[List[List[str]]] = None
_CLASS_COMBOS: Optional[Dict[str, int]] = None


def get_card_to_class_map() -> List[List[str]]:
    """Return a 52x52 matrix mapping (card1, card2) to its 169 hand class string."""
    global _CARD_PAIR_TO_CLASS
    if _CARD_PAIR_TO_CLASS is not None:
        return _CARD_PAIR_TO_CLASS

    matrix = [["" for _ in range(52)] for _ in range(52)]
    rank_chars = "23456789TJQKA"

    for c1 in range(52):
        r1, s1 = c1 // 4, c1 % 4
        for c2 in range(52):
            if c1 == c2:
                continue
            r2, s2 = c2 // 4, c2 % 4
            hi_r, lo_r = max(r1, r2), min(r1, r2)
            if hi_r == lo_r:
                cls_str = f"{rank_chars[hi_r]}{rank_chars[lo_r]}"
            elif s1 == s2:
                cls_str = f"{rank_chars[hi_r]}{rank_chars[lo_r]}s"
            else:
                cls_str = f"{rank_chars[hi_r]}{rank_chars[lo_r]}o"
            matrix[c1][c2] = cls_str

    _CARD_PAIR_TO_CLASS = matrix
    return _CARD_PAIR_TO_CLASS


def get_class_combos() -> Dict[str, int]:
    """Return dictionary of 169 classes to their combo counts (summing to 1,326)."""
    global _CLASS_COMBOS
    if _CLASS_COMBOS is not None:
        return _CLASS_COMBOS

    combos: Dict[str, int] = {}
    rank_chars = "23456789TJQKA"
    for i in range(13):
        combos[f"{rank_chars[i]}{rank_chars[i]}"] = 6
        for j in range(i):
            combos[f"{rank_chars[i]}{rank_chars[j]}s"] = 4
            combos[f"{rank_chars[i]}{rank_chars[j]}o"] = 12

    _CLASS_COMBOS = combos
    return _CLASS_COMBOS


# =========================================================================
# 2. Opponent Range Archetypes & Ranking
# =========================================================================

ARCHETYPES = {
    "Rock": 8.0,
    "TAG": 20.0,
    "LAG": 33.0,
    "Calling_Station": 50.0,
    "All_Hands": 100.0,
}


def get_ranked_classes(ranking: str = "heads_up") -> List[str]:
    """Return the 169 hand classes sorted by equity (descending).

    Parameters
    ----------
    ranking : str
        'heads_up' (n=2 equity) or 'multiway' (n=6 equity).
    """
    csv_path = FIGURES_DATA_DIR / "equity_table.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # Fallback if equity_table not on disk yet
        from holdem_lab.equity import generate_equity_table
        df = generate_equity_table(n_sim_per_class=400)

    n_val = 2 if ranking == "heads_up" else 6
    sub = df[df["n_players"] == n_val].sort_values("equity", ascending=False)
    return list(sub["hand_class"].unique())


def get_range_classes(
    q_percent: float,
    ranking: str = "heads_up",
) -> Set[str]:
    """Return the set of hand classes that make up the top q% of combinations.

    Parameters
    ----------
    q_percent : float
        Percentage in (0, 100].
    ranking : str
        'heads_up' or 'multiway'.
    """
    if q_percent >= 100.0:
        return set(get_class_combos().keys())

    combos_dict = get_class_combos()
    ranked = get_ranked_classes(ranking)
    target_combos = 1326.0 * (q_percent / 100.0)

    cum_combos = 0
    selected_classes: Set[str] = set()

    for cls in ranked:
        w = combos_dict[cls]
        selected_classes.add(cls)
        cum_combos += w
        if cum_combos >= target_combos:
            break

    return selected_classes


# =========================================================================
# 3. Limp-or-Fold Showdown Game with Blinds
# =========================================================================

def simulate_limp_fold_game(
    n_players: int = 6,
    n_hands: int = 10000,
    strategies: Optional[List[Union[float, Set[str], str]]] = None,
    ranking: str = "heads_up",
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate a limp-or-fold poker game with rotating blinds.

    Rules (CONTEXT.md Section 4):
    - Small blind 0.5 bb, Big blind 1.0 bb posted each hand.
    - Positions rotate by 1 seat each hand.
    - Each seat decides to play (put in 1.0 bb) or fold based on their strategy.
    - If nobody calls and SB folds: Big Blind wins Small Blind (BB +0.5, SB -0.5, others 0).
    - If at least one player calls: showdown between all active players (those putting in 1.0 bb).
    - Total pot is divided equally among players tied for best 7-card hand.
    - Chip conservation is strictly enforced (sum of profits = 0 on every hand).

    Parameters
    ----------
    n_players : int
        Number of players at the table (2 to 9).
    n_hands : int
        Number of hands to simulate.
    strategies : list or None
        List of strategy specs per player (can be archetype name, q percentage, or set of classes).
        Defaults to a mix: seat 0 TAG (20%), seat 1 LAG (33%), others Calling Station (50%).
    ranking : str
        Ranking used to convert q% to hand classes ('heads_up' or 'multiway').
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict containing:
        'profits': (n_hands, n_players) array of net profits in big blinds.
        'active_counts': (n_hands,) array of players reaching showdown.
        'pot_sizes': (n_hands,) array of total pot sizes.
        'chips_conserved': bool
        'summary': dict of statistics per player.
    """
    if strategies is None:
        # Default archetype lineup scaled to any table size (2 to 9)
        preset = ["TAG", "LAG", "Calling_Station", "Rock", "Calling_Station", "All_Hands", "TAG", "LAG", "Rock"]
        strategies = [preset[i % len(preset)] for i in range(n_players)]

    # Convert strategy specs to sets of playable hand classes
    player_ranges: List[Set[str]] = []
    for strat in strategies:
        if isinstance(strat, set):
            player_ranges.append(strat)
        elif isinstance(strat, (int, float)):
            player_ranges.append(get_range_classes(float(strat), ranking=ranking))
        elif isinstance(strat, str):
            if strat in ARCHETYPES:
                player_ranges.append(get_range_classes(ARCHETYPES[strat], ranking=ranking))
            else:
                player_ranges.append(get_range_classes(50.0, ranking=ranking))
        else:
            player_ranges.append(get_range_classes(50.0, ranking=ranking))

    card_to_class = get_card_to_class_map()
    rng = np.random.default_rng(seed)

    # Profits matrix: row = hand index, col = player index
    profits = np.zeros((n_hands, n_players), dtype=np.float64)
    active_counts = np.zeros(n_hands, dtype=np.int32)
    pot_sizes = np.zeros(n_hands, dtype=np.float64)

    deck = np.arange(52, dtype=np.int32)
    cards_needed = 2 * n_players + 5

    # Run hand simulation
    for h in range(n_hands):
        # Rotate button: hand h has SB at seat (h % n_players), BB at (h + 1) % n_players
        sb_seat = h % n_players
        bb_seat = (h + 1) % n_players

        # Deal cards
        deal = rng.choice(deck, size=cards_needed, replace=False)
        # Hole cards for seat i: deal[2*i], deal[2*i + 1]
        board = deal[2 * n_players : 2 * n_players + 5]

        # Determine hole classes
        hand_classes = [card_to_class[deal[2 * i]][deal[2 * i + 1]] for i in range(n_players)]

        # Determine voluntary play decisions
        would_play = [hand_classes[i] in player_ranges[i] for i in range(n_players)]

        # Track contributions and active showdown players
        contributions = np.zeros(n_players, dtype=np.float64)
        active_mask = np.zeros(n_players, dtype=bool)

        # 1. Posting blinds
        contributions[sb_seat] = 0.5
        contributions[bb_seat] = 1.0

        # 2. Non-blind players (seats other than SB and BB) act
        for i in range(n_players):
            if i != sb_seat and i != bb_seat:
                if would_play[i]:
                    contributions[i] = 1.0
                    active_mask[i] = True

        # 3. SB acts: complete to 1.0 or fold
        if would_play[sb_seat]:
            contributions[sb_seat] = 1.0
            active_mask[sb_seat] = True
        else:
            # SB folds, forfeiting 0.5
            active_mask[sb_seat] = False

        # 4. BB acts: BB posted 1.0
        # If nobody called and SB folded: BB wins SB without showdown
        if not np.any(active_mask):
            pot = contributions.sum()
            profits[h, bb_seat] = pot - contributions[bb_seat]  # +0.5
            profits[h, sb_seat] = -contributions[sb_seat]       # -0.5
            active_counts[h] = 1
            pot_sizes[h] = pot
            continue

        # If someone called, BB plays if hand is in range; folds (forfeiting 1.0) otherwise
        if would_play[bb_seat]:
            active_mask[bb_seat] = True
        else:
            active_mask[bb_seat] = False

        # Showdown occurs among active players
        pot = contributions.sum()
        pot_sizes[h] = pot
        active_seats = np.where(active_mask)[0]
        n_active = len(active_seats)
        active_counts[h] = n_active

        # If only 1 player remains active (e.g. caller won when BB folded), lone active player wins pot
        if n_active == 1:
            lone_winner = active_seats[0]
            payouts = np.zeros(n_players, dtype=np.float64)
            payouts[lone_winner] = pot
            profits[h, :] = payouts - contributions
            continue

        # Evaluate 7-card hands for active players
        active_7cards = np.empty((n_active, 7), dtype=np.int32)
        for idx, seat in enumerate(active_seats):
            active_7cards[idx, :2] = deal[2 * seat : 2 * seat + 2]
            active_7cards[idx, 2:] = board

        scores, _ = evaluate_7cards(active_7cards)
        max_score = np.max(scores)
        winners = active_seats[scores == max_score]
        payout_per_winner = pot / len(winners)

        payouts = np.zeros(n_players, dtype=np.float64)
        for w in winners:
            payouts[w] += payout_per_winner

        profits[h, :] = payouts - contributions

    chips_conserved = bool(np.all(np.abs(profits.sum(axis=1)) < 1e-9))

    summary = {}
    for p in range(n_players):
        mean_p = float(np.mean(profits[:, p]))
        std_p = float(np.std(profits[:, p]))
        strat_desc = str(strategies[p])
        summary[f"player_{p}"] = {
            "strategy": strat_desc,
            "mean_bb_hand": mean_p,
            "std_bb_hand": std_p,
            "win_rate_bb_100": mean_p * 100.0,
        }

    return {
        "profits": profits,
        "active_counts": active_counts,
        "pot_sizes": pot_sizes,
        "chips_conserved": chips_conserved,
        "summary": summary,
    }


def closed_form_blind_cost(
    n_players: int,
    q_fraction: float,
    expected_gain_when_played: float = 0.0,
) -> float:
    """Theoretical expected profit per hand from blind costs and played hands.

    In an orbit of n hands, a player pays 0.5 (SB) and 1.0 (BB), total 1.5 bb.
    Baseline blind loss rate = 1.5 / n bb/hand.
    If playing fraction q of hands yields an expected profit of `gain` bb per played hand,
    the net win rate is:
    E[profit] = q * expected_gain_when_played - (1.5 / n).
    """
    baseline_blind_loss = 1.5 / float(n_players)
    return float(q_fraction * expected_gain_when_played - baseline_blind_loss)


# =========================================================================
# 4. Range Combos and Equity vs Range Models (G4)
# =========================================================================

def get_range_combos(
    q_percent: float,
    ranking: str = "heads_up",
) -> np.ndarray:
    """Return all valid 2-card combinations in the top q% range.

    Returns
    -------
    np.ndarray of shape (K, 2)
    """
    card_to_class = get_card_to_class_map()
    range_classes = get_range_classes(q_percent, ranking=ranking)

    combos_list = []
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            if card_to_class[c1][c2] in range_classes:
                combos_list.append((c1, c2))

    return np.array(combos_list, dtype=np.int32)


def estimate_equity_vs_range(
    hand_cards: Sequence[int],
    n_players: int = 2,
    q_percent: float = 100.0,
    ranking: str = "heads_up",
    n_sim: int = 600,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate equity of hand_cards against n-1 opponents holding hands from top q% range.

    Parameters
    ----------
    hand_cards : sequence of 2 ints
        Focal hole cards.
    n_players : int
        Table size (2 to 9).
    q_percent : float
        Opponent tightness percentage (8, 20, 33, 50, 100).
    ranking : str
        'heads_up' or 'multiway'.
    n_sim : int
        Monte Carlo simulation deals.
    seed : int
        Random seed.
    """
    focal = np.array(hand_cards, dtype=np.int32)
    range_combos = get_range_combos(q_percent, ranking=ranking)
    deck = np.arange(52, dtype=np.int32)

    rng = np.random.default_rng(seed)
    wins = 0.0

    for _ in range(n_sim):
        dead_cards = set(focal)

        # Draw n-1 opponents from available range combos
        opp_hands = []
        valid_opponents = True
        for _ in range(n_players - 1):
            # Filter combos not containing any dead cards
            avail = [
                comb
                for comb in range_combos
                if comb[0] not in dead_cards and comb[1] not in dead_cards
            ]
            if len(avail) == 0:
                valid_opponents = False
                break
            chosen = avail[rng.integers(0, len(avail))]
            dead_cards.add(chosen[0])
            dead_cards.add(chosen[1])
            opp_hands.append(chosen)

        if not valid_opponents:
            continue

        # Deal 5 community board cards from remaining deck
        avail_board = [c for c in deck if c not in dead_cards]
        board = rng.choice(avail_board, size=5, replace=False)

        # Showdown evaluation
        hands_7 = np.empty((n_players, 7), dtype=np.int32)
        hands_7[0, :2] = focal
        hands_7[0, 2:] = board
        for p in range(n_players - 1):
            hands_7[p + 1, :2] = opp_hands[p]
            hands_7[p + 1, 2:] = board

        scores, _ = evaluate_7cards(hands_7)
        max_s = np.max(scores)
        if scores[0] == max_s:
            tied = np.sum(scores == max_s)
            wins += 1.0 / float(tied)

    eq = wins / float(n_sim)
    return {
        "equity": eq,
        "n_players": n_players,
        "q_percent": q_percent,
        "ranking": ranking,
    }


# =========================================================================
# 5. Table Selection ANOVA Factorial Design (G8)
# =========================================================================

TABLE_MIXES = {
    "All_Wild": ["Calling_Station"] * 5,
    "Mixed": ["TAG", "LAG", "Calling_Station", "Rock", "LAG"],
    "All_Tight": ["Rock", "TAG", "Rock", "TAG", "TAG"],
    "All_Medium": ["LAG"] * 5,
}

FOCAL_STRATEGIES = ["Rock", "TAG", "LAG", "Calling_Station", "All_Hands"]


def run_table_selection_experiment(
    n_reps: int = 15,
    n_hands_per_rep: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run factorial experiment of focal strategy x opponent mix across independent replicates.

    Parameters
    ----------
    n_reps : int
        Number of independent game replicates per cell.
    n_hands_per_rep : int
        Hands played per replicate.
    seed : int
        Base random seed.
    """
    records = []
    current_seed = seed

    for mix_name, opp_strats in TABLE_MIXES.items():
        for focal_strat in FOCAL_STRATEGIES:
            table_lineup = [focal_strat] + opp_strats[:5]

            for rep in range(n_reps):
                res = simulate_limp_fold_game(
                    n_players=6,
                    n_hands=n_hands_per_rep,
                    strategies=table_lineup,
                    ranking="heads_up",
                    seed=current_seed,
                )
                current_seed += 1

                focal_profit = res["profits"][:, 0]
                mean_bb_hand = float(np.mean(focal_profit))
                win_rate_100 = mean_bb_hand * 100.0

                records.append({
                    "strategy": focal_strat,
                    "mix": mix_name,
                    "replicate": rep,
                    "mean_bb_hand": mean_bb_hand,
                    "win_rate_bb_100": win_rate_100,
                })

    return pd.DataFrame(records)

