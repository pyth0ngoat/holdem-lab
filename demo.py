#!/usr/bin/env python3
"""Hold'em Lab — Interactive Classroom Probability Demo (demo.py).

Educational probability demonstration for Texas Hold'em:
- Single self-contained file (server, game engine, bots, mobile and host UI).
- Educational framing: play chips only, no gambling advice, visible banner.
- Max 9 seats; overflow joiners become spectators; empty seats filled by bots.
- Community board (Flop, Turn, River) rendered in real time on mobile phones and host.
- Hole cards sent strictly to owning seat WebSocket.
- Unaided guess slider logged first, then simulation aid and formula recipe revealed.
- Dynamic table-size n recomputation on folds with real-time equity updates.
- Postflop outs counter, Rule of 2 and 4 estimate vs exact live board equity.
- Host projector view: QR code join, board, equity lines, chip and calibration leaderboards.
- Pre-flight checklist: python demo.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import qrcode
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Hold'em Lab package imports
from holdem_lab.cards import (
    CATEGORY_NAMES,
    RANKS_STR,
    SUITS_STR,
    card_to_str,
    str_to_card,
    evaluate_7cards,
    evaluate_5cards,
)
from holdem_lab.equity import (
    ALL_CLASSES,
    COMBO_WEIGHTS,
    cards_to_class,
    estimate_hand_equity,
)
from holdem_lab.strategy import (
    ARCHETYPES,
    FOCAL_STRATEGIES,
    get_range_classes,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DATA_DIR = ROOT_DIR / "figures" / "data"
LOGS_DIR = DATA_DIR / "demo_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Loading & Equity Cache
# ---------------------------------------------------------------------------

_EQUITY_CACHE: dict[tuple[str, int], dict[str, float]] = {}
_FORMULA_PARAMS: dict[str, Any] = {}


def load_demo_data() -> None:
    """Load pre-computed 169-class equity table and formula parameters."""
    global _EQUITY_CACHE, _FORMULA_PARAMS
    eq_path = FIGURES_DATA_DIR / "equity_table.csv"
    if eq_path.exists():
        df_eq = pd.read_csv(eq_path)
        for _, row in df_eq.iterrows():
            key = (str(row["hand_class"]), int(row["n_players"]))
            _EQUITY_CACHE[key] = {
                "equity": float(row["equity"]),
                "p_win": float(row["p_win"]),
                "p_tie": float(row["p_tie"]),
                "equity_se": float(row["equity_se"]),
            }

    fp_path = DATA_DIR / "formula_params.json"
    if fp_path.exists():
        with open(fp_path, "r", encoding="utf-8") as f:
            _FORMULA_PARAMS = json.load(f)


def get_empirical_equity(hand_class: str, n_players: int) -> dict[str, float]:
    """Retrieve pre-computed empirical equity row for (hand_class, n)."""
    n_clamped = max(2, min(9, n_players))
    key = (hand_class, n_clamped)
    if key in _EQUITY_CACHE:
        return _EQUITY_CACHE[key]
    base_eq = 1.0 / n_clamped
    return {
        "equity": base_eq,
        "p_win": base_eq * 0.95,
        "p_tie": base_eq * 0.05,
        "equity_se": 0.005,
    }


def compute_formula_estimate(hand_class: str, n_players: int) -> dict[str, Any]:
    """Calculate Model M1 closed-form formula prediction and by-hand recipe."""
    n_clamped = max(2, min(9, n_players))
    hu_row = get_empirical_equity(hand_class, 2)
    p2 = hu_row["p_win"]
    b_law = _FORMULA_PARAMS.get("smooth_law", {}).get("b", 0.2301)
    beta_n = 1.0 + b_law * np.sqrt(n_clamped - 2)
    two_p2 = 2.0 * p2
    pred_equity = (1.0 / n_clamped) * (two_p2 ** beta_n)

    recipe = (
        f"1. Heads-up win rate p2 = {p2*100:.1f}% -> 2*p2 = {two_p2:.3f}\n"
        f"2. Exponent beta_{n_clamped} = 1.0 + {b_law:.3f}*sqrt({n_clamped}-2) = {beta_n:.3f}\n"
        f"3. Formula = (1/{n_clamped}) * ({two_p2:.3f})^{beta_n:.3f} = {pred_equity*100:.1f}%"
    )

    return {
        "pred_equity": float(pred_equity),
        "beta_n": float(beta_n),
        "p2": float(p2),
        "recipe": recipe,
    }


def get_chart_coordinates(hand_class: str) -> tuple[int, int]:
    """Map canonical 169-class name to 13x13 preflop chart row and column."""
    r_map = {r: 12 - i for i, r in enumerate("AKQJT98765432")}
    c1, c2 = hand_class[0], hand_class[1]
    i1, i2 = r_map[c1], r_map[c2]
    if len(hand_class) == 2:  # Pair: diagonal
        return (i1, i1)
    elif hand_class[2] == "s":  # Suited: upper triangle
        return (min(i1, i2), max(i1, i2))
    else:  # Offsuit: lower triangle
        return (max(i1, i2), min(i1, i2))


def describe_hand(hand_class: str) -> str:
    """Return friendly one-line pedagogical explanation of starting hand."""
    if not hand_class:
        return ""
    c1, c2 = hand_class[0], hand_class[1]
    if len(hand_class) == 2:
        return f"Pocket Pair ({c1}{c1}) — strong preflop showdown value, sets on flop 11.8%."
    elif hand_class[2] == "s":
        return f"Suited Hand ({hand_class}) — high multiway potential with flush and straight draws."
    else:
        return f"Offsuit Hand ({hand_class}) — high card value, but weaker multiway drawing power."


def analyze_postflop_hand(
    hero_cards: list[int], board_cards: list[int], n_opponents: int
) -> dict[str, Any]:
    """Compute outs, Rule of 2 and 4 estimate, and live Monte Carlo board equity."""
    if len(board_cards) < 3 or len(hero_cards) < 2:
        return {}

    used = set(hero_cards + board_cards)
    deck = [c for c in range(52) if c not in used]
    k = len(board_cards)

    # 1. Flush Draw Check
    suits = [c % 4 for c in hero_cards + board_cards]
    flush_outs = 0
    draw_notes: list[str] = []
    suit_names = ["♣", "♦", "♥", "♠"]
    for s in range(4):
        if suits.count(s) == 4:
            flush_outs = 9
            draw_notes.append(f"Flush Draw ({suit_names[s]}, 9 outs)")
            break
        elif suits.count(s) >= 5:
            draw_notes.append(f"Made Flush ({suit_names[s]}!)")

    # 2. Straight Draw Check
    ranks_present = set(c // 4 for c in hero_cards + board_cards)
    straight_outs = 0
    # Check possible 5-high straight (A,2,3,4,5) to ace-high straight (T,J,Q,K,A)
    for low in range(-1, 9):
        needed = set(12 if x == -1 else x for x in range(low, low + 5))
        missing = needed - ranks_present
        if len(missing) == 1:
            missing_rank = list(missing)[0]
            # count unseen cards of that rank
            seen_of_rank = sum(1 for c in (hero_cards + board_cards) if c // 4 == missing_rank)
            unseen_rank_cards = max(0, 4 - seen_of_rank)
            straight_outs += unseen_rank_cards
    if straight_outs >= 8:
        draw_notes.append(f"Open-Ended Straight Draw ({straight_outs} outs)")
    elif straight_outs > 0:
        draw_notes.append(f"Gutshot Straight Draw ({straight_outs} outs)")

    # 3. Overcards Check
    board_ranks = [c // 4 for c in board_cards]
    hero_ranks = [c // 4 for c in hero_cards]
    max_b = max(board_ranks)
    overcards = [r for r in hero_ranks if r > max_b]
    over_outs = 0
    if overcards and not flush_outs and not straight_outs:
        over_outs = len(overcards) * 3
        draw_notes.append(f"{len(overcards)} Overcard(s) to Board ({over_outs} outs)")

    total_outs = flush_outs + min(8, straight_outs) + over_outs
    if not draw_notes:
        draw_notes.append("Made hand / showdown value")

    # Rule of 2 and 4
    if k == 3:
        rule_pct = round(min(100.0, total_outs * 4.0), 1)
        rule_desc = f"Rule of 4: {total_outs} outs × 4% = {rule_pct}%"
    elif k == 4:
        rule_pct = round(min(100.0, total_outs * 2.0), 1)
        rule_desc = f"Rule of 2: {total_outs} outs × 2% = {rule_pct}%"
    else:
        rule_pct = None
        rule_desc = "River reached (Showdown evaluation)"

    # Fast Monte Carlo Live Equity (250 iterations)
    rng = np.random.default_rng(2026 + k)
    wins = 0.0
    n_sim = 250
    cards_needed = 5 - k
    active_opps = max(1, min(8, n_opponents))

    for _ in range(n_sim):
        sub = rng.permutation(deck)
        full_b = board_cards + list(sub[:cards_needed])
        opps = [list(sub[cards_needed + 2 * i : cards_needed + 2 * i + 2]) + full_b for i in range(active_opps)]
        all_7 = np.array([hero_cards + full_b] + opps, dtype=np.int8)
        sc, _ = evaluate_7cards(all_7)
        max_sc = np.max(sc)
        if sc[0] == max_sc:
            wins += 1.0 / np.sum(sc == max_sc)

    live_equity = wins / n_sim

    return {
        "draw_notes": draw_notes,
        "total_outs": total_outs,
        "rule_desc": rule_desc,
        "rule_pct": rule_pct,
        "live_equity": round(float(live_equity), 3),
    }


# ---------------------------------------------------------------------------
# Network & QR Utilities
# ---------------------------------------------------------------------------


def get_lan_ip() -> str:
    """Detect LAN IPv4 address with UDP socket probe."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def generate_qr_data_url(url: str) -> str:
    """Generate in-memory base64 PNG data URI for a QR code."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def sanitize_nickname(name: str) -> str:
    """Sanitize nickname to alphanumeric + underscore/space, max 16 chars."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\- ]", "", name).strip()
    return cleaned[:16] if cleaned else "Player"


# ---------------------------------------------------------------------------
# Game Engine & Room State
# ---------------------------------------------------------------------------


class Seat:
    def __init__(self, seat_idx: int):
        self.seat_idx: int = seat_idx
        self.occupied: bool = False
        self.player_id: str = ""
        self.nickname: str = f"Seat {seat_idx+1}"
        self.is_bot: bool = False
        self.bot_archetype: str = "TAG"
        self.chips: int = 100
        self.cards: list[int] = []
        self.hand_class: str = ""
        self.active: bool = False
        self.folded: bool = False
        self.unaided_guess: Optional[float] = None
        self.aided_guess: Optional[float] = None
        self.aid_revealed: bool = False
        self.brier_score_sum: float = 0.0
        self.rounds_guessed: int = 0
        self.showdown_hand_cat: str = ""
        self.showdown_score: int = 0
        self.showdown_is_winner: bool = False
        self.showdown_payout: int = 0

    def to_dict(self, is_self: bool = False, reveal_showdown: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seat_idx": self.seat_idx,
            "occupied": self.occupied,
            "nickname": self.nickname,
            "is_bot": self.is_bot,
            "bot_archetype": self.bot_archetype,
            "chips": self.chips,
            "active": self.active,
            "folded": self.folded,
            "unaided_guess": self.unaided_guess,
            "aided_guess": self.aided_guess,
            "aid_revealed": self.aid_revealed,
            "calib_score": round(self.brier_score_sum / max(1, self.rounds_guessed), 4),
            "showdown_cat": self.showdown_hand_cat,
            "is_winner": self.showdown_is_winner,
            "payout": self.showdown_payout,
        }
        if (is_self or reveal_showdown) and self.cards:
            d["cards"] = [card_to_str(c) for c in self.cards]
            d["hand_class"] = self.hand_class
        return d


class GameRoom:
    def __init__(self, room_code: str, seed: int = 2026, bots_count: int = 6):
        self.room_code: str = room_code.upper()
        self.seed: int = seed
        self.rng = np.random.default_rng(seed)
        self.seats: list[Seat] = [Seat(i) for i in range(9)]
        self.bots_enabled: bool = True
        self.phase: str = "waiting"  # waiting, preflop, flop, turn, river, showdown
        self.community_cards: list[int] = []
        self.deck: list[int] = []
        self.pot: int = 0
        self.round_id: int = 0
        self.audit_log: list[dict[str, Any]] = []
        self.record_log: list[dict[str, Any]] = []
        self.log_file: Optional[Path] = None
        self.last_winners: list[dict[str, Any]] = []

        # Central WebSocket Connections
        self.host_sockets: set[WebSocket] = set()
        self.player_sockets: dict[int, WebSocket] = {}
        self.spectator_sockets: set[WebSocket] = set()

        # Fill default bots
        archetypes = ["Rock", "TAG", "LAG", "Calling_Station", "Wild", "TAG"]
        for i in range(min(bots_count, 9)):
            s = self.seats[i]
            s.occupied = True
            s.is_bot = True
            arch = archetypes[i % len(archetypes)]
            s.bot_archetype = arch
            s.nickname = f"Bot-{arch}-{i+1}"
            s.player_id = f"bot_{i}"

    def get_active_count(self) -> int:
        return sum(1 for s in self.seats if s.occupied and s.active and not s.folded)

    def start_round(self) -> None:
        """Deal a new round of Texas Hold'em."""
        self.round_id += 1
        self.community_cards = []
        self.pot = 0
        self.phase = "preflop"
        self.last_winners = []

        # Shuffle 52-card deck
        deck = np.arange(52, dtype=np.int32)
        self.rng.shuffle(deck)
        self.deck = deck.tolist()

        # Deal 2 cards and 1-chip ante from each active player
        for s in self.seats:
            if s.occupied and s.chips > 0:
                s.active = True
                s.folded = False
                s.unaided_guess = None
                s.aided_guess = None
                s.aid_revealed = False
                s.showdown_hand_cat = ""
                s.showdown_score = 0
                s.showdown_is_winner = False
                s.showdown_payout = 0
                s.chips -= 1
                self.pot += 1
                c1 = self.deck.pop(0)
                c2 = self.deck.pop(0)
                s.cards = [c1, c2]
                s.hand_class = cards_to_class(c1, c2)
            else:
                s.active = False
                s.cards = []
                s.hand_class = ""
                s.showdown_hand_cat = ""
                s.showdown_score = 0
                s.showdown_is_winner = False
                s.showdown_payout = 0

        # Pre-assign bot guesses
        active_n = self.get_active_count()
        for s in self.seats:
            if s.active and s.is_bot:
                eq_row = get_empirical_equity(s.hand_class, active_n)
                true_eq = eq_row["equity"]
                noise = float(self.rng.normal(0.0, 0.08))
                s.unaided_guess = round(max(0.02, min(0.98, true_eq + noise)), 3)
                s.aided_guess = round(max(0.02, min(0.98, true_eq + noise * 0.25)), 3)
                s.aid_revealed = True

        self._log_event("deal_preflop", {"active_n": self.get_active_count(), "pot": self.pot})

    def apply_bot_decisions(self) -> None:
        """Bots decide whether to play or fold based on their archetype range."""
        active_n = self.get_active_count()
        for s in self.seats:
            if s.active and not s.folded and s.is_bot:
                played_classes = get_range_classes(ARCHETYPES.get(s.bot_archetype, 20.0))
                if s.hand_class not in played_classes and active_n > 2:
                    s.folded = True
                    active_n = self.get_active_count()

    def deal_flop(self) -> None:
        if self.phase == "preflop" and len(self.deck) >= 3:
            self.apply_bot_decisions()
            self.community_cards = [self.deck.pop(0), self.deck.pop(0), self.deck.pop(0)]
            self.phase = "flop"
            self._log_event("deal_flop", {"board": [card_to_str(c) for c in self.community_cards]})

    def deal_turn(self) -> None:
        if self.phase == "flop" and len(self.deck) >= 1:
            self.community_cards.append(self.deck.pop(0))
            self.phase = "turn"
            self._log_event("deal_turn", {"board": [card_to_str(c) for c in self.community_cards]})

    def deal_river(self) -> None:
        if self.phase == "turn" and len(self.deck) >= 1:
            self.community_cards.append(self.deck.pop(0))
            self.phase = "river"
            self._log_event("deal_river", {"board": [card_to_str(c) for c in self.community_cards]})

    def showdown(self) -> dict[str, Any]:
        """Evaluate hands, award pot with split-pot support, update calibration."""
        if self.phase not in ("preflop", "flop", "turn", "river"):
            return {}

        if self.phase == "preflop":
            self.apply_bot_decisions()

        # Complete board to 5 cards if needed
        while len(self.community_cards) < 5 and len(self.deck) > 0:
            self.community_cards.append(self.deck.pop(0))

        self.phase = "showdown"
        for s in self.seats:
            s.showdown_hand_cat = ""
            s.showdown_score = 0
            s.showdown_is_winner = False
            s.showdown_payout = 0

        active_seats = [s for s in self.seats if s.active and not s.folded]

        if not active_seats:
            return {"winners": [], "payout": 0}

        board = self.community_cards
        eval_hands = [np.array(s.cards + board, dtype=np.int8) for s in active_seats]
        scores, cats = evaluate_7cards(np.array(eval_hands, dtype=np.int8))
        for i, s in enumerate(active_seats):
            s.showdown_score = int(scores[i])
            s.showdown_hand_cat = CATEGORY_NAMES[int(cats[i])]

        max_score = int(np.max(scores))
        winning_indices = [i for i, sc in enumerate(scores) if sc == max_score]
        n_winners = len(winning_indices)
        share = self.pot // n_winners
        remainder = self.pot % n_winners

        winners = []
        for idx in winning_indices:
            ws = active_seats[idx]
            payout = share + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            ws.chips += payout
            ws.showdown_is_winner = True
            ws.showdown_payout = payout
            winners.append({
                "nickname": ws.nickname,
                "seat_idx": ws.seat_idx,
                "payout": payout,
                "category": ws.showdown_hand_cat,
                "cards": [card_to_str(c) for c in ws.cards],
            })

        self.last_winners = winners

        # Calibration scoring & G9 Logging
        active_n_start = sum(1 for s in self.seats if s.active)
        for s in self.seats:
            if s.active:
                eq_row = get_empirical_equity(s.hand_class, active_n_start)
                true_eq = eq_row["equity"]
                if s.unaided_guess is not None:
                    brier = (s.unaided_guess - true_eq) ** 2
                    s.brier_score_sum += brier
                    s.rounds_guessed += 1

                log_entry = {
                    "round_id": self.round_id,
                    "seat_idx": s.seat_idx,
                    "nickname": s.nickname,
                    "is_bot": s.is_bot,
                    "hand_class": s.hand_class,
                    "n_players": active_n_start,
                    "true_equity": round(true_eq, 4),
                    "unaided_guess": s.unaided_guess,
                    "aided_guess": s.aided_guess,
                    "decision": "fold" if s.folded else "play",
                    "payout": next((w["payout"] for w in winners if w["seat_idx"] == s.seat_idx), 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.record_log.append(log_entry)
                if self.log_file:
                    with open(self.log_file, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_entry) + "\n")

        self.pot = 0
        self._log_event("showdown", {"winners": winners})
        return {"winners": winners}

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.audit_log.append({
            "round_id": self.round_id,
            "event": event_type,
            "timestamp": time.time(),
            "data": data,
        })

    def build_player_payload(
        self, seat: Seat, active_n: int, board_strs: list[str]
    ) -> dict[str, Any]:
        """Construct the personalized private payload for a seated player."""
        emp = (
            get_empirical_equity(seat.hand_class, active_n)
            if (seat.active and seat.hand_class)
            else {"equity": 0.0, "p_win": 0.0, "p_tie": 0.0, "equity_se": 0.0}
        )
        form = (
            compute_formula_estimate(seat.hand_class, active_n)
            if (seat.active and seat.hand_class)
            else {"pred_equity": 0.0, "beta_n": 1.0, "p2": 0.5, "recipe": ""}
        )
        chart_coord = (
            get_chart_coordinates(seat.hand_class)
            if (seat.active and seat.hand_class)
            else [0, 0]
        )

        postflop = None
        if seat.active and seat.cards and len(self.community_cards) >= 3:
            postflop = analyze_postflop_hand(
                seat.cards, self.community_cards, max(1, active_n - 1)
            )

        my_showdown = None
        if self.phase == "showdown":
            my_showdown = {
                "is_winner": seat.showdown_is_winner,
                "payout": seat.showdown_payout,
                "hand_category": seat.showdown_hand_cat,
                "folded": seat.folded,
            }

        return {
            "type": "update",
            "role": "player",
            "seat_idx": seat.seat_idx,
            "nickname": seat.nickname,
            "chips": seat.chips,
            "round_id": self.round_id,
            "phase": self.phase,
            "pot": self.pot,
            "active_n": active_n,
            "board": board_strs,
            "active": seat.active,
            "folded": seat.folded,
            "cards": [card_to_str(c) for c in seat.cards] if seat.active else [],
            "hand_class": seat.hand_class if seat.active else "",
            "hand_desc": describe_hand(seat.hand_class) if seat.active else "",
            "chart_coord": chart_coord,
            "emp_equity": emp["equity"],
            "emp_se": emp["equity_se"],
            "emp_p_win": emp["p_win"],
            "emp_p_tie": emp["p_tie"],
            "form_equity": form["pred_equity"],
            "form_beta_n": form["beta_n"],
            "form_recipe": form["recipe"],
            "unaided_guess": seat.unaided_guess,
            "aided_guess": seat.aided_guess,
            "aid_revealed": seat.aid_revealed,
            "postflop": postflop,
            "winners": self.last_winners,
            "my_showdown": my_showdown,
        }

    async def broadcast_state(self) -> None:
        """Broadcast live game updates to host, player phones, and spectators."""
        active_n = self.get_active_count()
        board_strs = [card_to_str(c) for c in self.community_cards]
        reveal_showdown = self.phase == "showdown"

        # 1. State for Host Projector View
        host_payload = {
            "type": "state",
            "round_id": self.round_id,
            "phase": self.phase,
            "pot": self.pot,
            "active_n": active_n,
            "board": board_strs,
            "seats": [s.to_dict(is_self=False, reveal_showdown=reveal_showdown) for s in self.seats],
            "winners": self.last_winners,
        }
        for ws in list(self.host_sockets):
            try:
                await ws.send_json(host_payload)
            except Exception:
                self.host_sockets.discard(ws)

        # 2. State for Spectator Devices
        spec_payload = {
            "type": "update",
            "role": "spectator",
            "round_id": self.round_id,
            "phase": self.phase,
            "pot": self.pot,
            "active_n": active_n,
            "board": board_strs,
            "seats": [s.to_dict(is_self=False, reveal_showdown=reveal_showdown) for s in self.seats],
            "winners": self.last_winners,
        }
        for ws in list(self.spectator_sockets):
            try:
                await ws.send_json(spec_payload)
            except Exception:
                self.spectator_sockets.discard(ws)

        # 3. Personalized Real-Time Payload for Each Phone Seat
        for seat_idx, ws in list(self.player_sockets.items()):
            seat = self.seats[seat_idx]
            player_payload = self.build_player_payload(seat, active_n, board_strs)
            try:
                await ws.send_json(player_payload)
            except Exception:
                self.player_sockets.pop(seat_idx, None)


# Global in-memory rooms registry
ROOMS: dict[str, GameRoom] = {}


def get_or_create_room(room_code: str, seed: int = 2026, bots_count: int = 6) -> GameRoom:
    code = room_code.upper().strip()
    if code not in ROOMS:
        ROOMS[code] = GameRoom(code, seed=seed, bots_count=bots_count)
    return ROOMS[code]


# ---------------------------------------------------------------------------
# HTML UI Templates (Single File Embedding)
# ---------------------------------------------------------------------------

HOST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hold'em Lab — Host Projector View</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0b1329; --panel: #1e293b; --text: #f8fafc; --text-muted: #94a3b8;
      --accent: #38bdf8; --border: #334155; --spade: #0f172a; --heart: #e11d48;
      --diamond: #0284c7; --club: #16a34a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 20px; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
    .banner { background: #0369a1; color: white; text-align: center; padding: 8px; font-weight: bold; font-size: 0.95rem; border-radius: 6px; margin-bottom: 16px; letter-spacing: 0.5px; }
    .grid { display: grid; grid-template-columns: 280px 1fr 340px; gap: 20px; }
    .card-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }
    h2 { font-size: 1.2rem; color: var(--accent); margin-bottom: 12px; }
    .qr-container { text-align: center; margin: 12px 0; }
    .qr-container img { width: 190px; height: 190px; border-radius: 8px; border: 4px solid white; }
    .board-cards { display: flex; gap: 12px; justify-content: center; margin: 20px 0; }
    .poker-card { width: 64px; height: 92px; background: white; border-radius: 8px; display: flex; flex-direction: column; justify-content: space-between; padding: 6px 8px; font-weight: bold; font-size: 1.3rem; box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
    .poker-card.suit-s { color: var(--spade); }
    .poker-card.suit-h { color: var(--heart); }
    .poker-card.suit-d { color: var(--diamond); }
    .poker-card.suit-c { color: var(--club); }
    .card-label { font-size: 0.75rem; color: var(--text-muted); text-align: center; margin-top: 4px; font-weight: normal; }
    .seats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 15px; }
    .seat-card { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 0.9rem; text-align: left; }
    .seat-card.active { border-color: var(--accent); box-shadow: 0 0 8px rgba(56, 189, 248, 0.3); }
    .seat-card.winner { border: 2px solid #22c55e !important; background: rgba(34, 197, 94, 0.18) !important; box-shadow: 0 0 15px rgba(34, 197, 94, 0.5) !important; }
    .seat-card.folded { opacity: 0.4; }
    .mini-card { width: 34px; height: 48px; background: white; border-radius: 4px; display: inline-flex; flex-direction: column; justify-content: space-between; padding: 2px 4px; font-weight: bold; font-size: 0.8rem; box-shadow: 0 2px 4px rgba(0,0,0,0.3); vertical-align: middle; }
    .mini-card.suit-s { color: var(--spade); }
    .mini-card.suit-h { color: var(--heart); }
    .mini-card.suit-d { color: var(--diamond); }
    .mini-card.suit-c { color: var(--club); }
    .btn { background: #0284c7; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.95rem; margin: 4px; transition: background 0.2s; }
    .btn:hover { background: #0369a1; }
    .leaderboard table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 10px; }
    .leaderboard th, .leaderboard td { padding: 8px 6px; text-align: left; border-bottom: 1px solid var(--border); }
    .badge { padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; background: #334155; }
    .crowd-stat { font-size: 1.2rem; font-weight: bold; color: #34d399; margin-top: 8px; }
    .winner-banner { background: #15803d; color: white; padding: 10px; border-radius: 6px; font-weight: bold; text-align: center; margin-top: 15px; font-size: 1.1rem; }
  </style>
</head>
<body>
  <div class="banner">EDUCATIONAL PROBABILITY DEMO — PLAY CHIPS ONLY — NO REAL MONEY</div>
  <header>
    <div>
      <h1>Hold'em Lab Demo <span class="badge" id="roomBadge">ROOM: ...</span></h1>
      <p style="color:var(--text-muted); font-size:0.95rem;">MSc Applied Probability Distributions (THM6242)</p>
    </div>
    <div id="controls">
      <button class="btn" onclick="sendAction('start')">Deal Preflop</button>
      <button class="btn" onclick="sendAction('flop')">Flop (3 Cards)</button>
      <button class="btn" onclick="sendAction('turn')">Turn (4th Card)</button>
      <button class="btn" onclick="sendAction('river')">River (5th Card)</button>
      <button class="btn" onclick="sendAction('showdown')">Showdown</button>
    </div>
  </header>

  <div class="grid">
    <!-- QR & Join Panel -->
    <div class="card-panel">
      <h2>Mobile QR Join</h2>
      <div class="qr-container">
        <img id="qrImg" src="" alt="QR Code">
      </div>
      <p style="text-align:center; font-size:0.9rem; color:var(--text-muted); margin-bottom:12px;">
        Scan or connect via phone browser:<br><strong id="joinUrl" style="color:var(--accent); word-break:break-all;">http://...</strong>
      </p>
      <div style="font-size:0.85rem; color:var(--text-muted); border-top:1px solid var(--border); padding-top:10px;">
        <strong>Campus Wi-Fi Tip:</strong> Many university networks isolate devices. Use a laptop or mobile hotspot so all phones connect to the same subnet.
      </div>
    </div>

    <!-- Main Table & Community Board -->
    <div class="card-panel" style="text-align:center;">
      <h2 style="text-align:left;">Community Board (Round #<span id="roundId">1</span> — Phase: <span id="phase" style="text-transform:uppercase; color:#38bdf8;">WAITING</span>)</h2>
      <div style="font-size:1.3rem; font-weight:bold; margin-bottom:12px;">
        Pot: <span id="pot" style="color:#f59e0b;">0</span> chips | Active In-Hand: <span id="activeN" style="color:#34d399;">0</span> seats
      </div>

      <div class="board-cards" id="boardCards"></div>
      <div id="winnerBanner" class="winner-banner" style="display:none;"></div>

      <h2 style="text-align:left; margin-top:20px;">Seats at Table (Max 9)</h2>
      <div class="seats-grid" id="seatsGrid"></div>
    </div>

    <!-- Leaderboards & Crowd Wisdom -->
    <div class="card-panel">
      <h2>Calibration & Leaderboard</h2>
      <div class="leaderboard">
        <table>
          <thead>
            <tr><th>Seat</th><th>Chips</th><th>Calibration</th></tr>
          </thead>
          <tbody id="leaderboardBody"></tbody>
        </table>
      </div>

      <h2 style="margin-top:24px;">Wisdom of Crowds</h2>
      <div id="crowdStats">
        <div>Crowd Avg Guess: <span id="crowdAvg" class="crowd-stat">--</span></div>
        <div style="color:var(--text-muted); font-size:0.85rem; margin-top:6px;">Average unaided player guess across seats compared against combinatorial truth.</div>
      </div>
    </div>
  </div>

  <script>
    const room = new URLSearchParams(window.location.search).get("room") || "DEMO";
    document.getElementById("roomBadge").textContent = "ROOM: " + room;
    let ws;

    function initWS() {
      const loc = window.location;
      const proto = loc.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${loc.host}/ws?room=${room}&role=host`);
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "init") {
          document.getElementById("qrImg").src = data.qr_url;
          document.getElementById("joinUrl").textContent = data.join_url;
        } else if (data.type === "state") {
          updateState(data);
        }
      };
      ws.onclose = () => setTimeout(initWS, 2000);
    }

    function renderCard(cardStr) {
      if (!cardStr) {
        return `<div class="poker-card" style="opacity:0.25; border:2px dashed #94a3b8; background:transparent;"></div>`;
      }
      const rank = cardStr[0];
      const suit = cardStr[1];
      const suitSym = suit === "s" ? "♠" : suit === "h" ? "♥" : suit === "d" ? "♦" : "♣";
      return `<div class="poker-card suit-${suit}"><div>${rank}</div><div style="align-self:flex-end;">${suitSym}</div></div>`;
    }

    function renderMiniCard(cardStr) {
      if (!cardStr) return '';
      const rank = cardStr[0];
      const suit = cardStr[1];
      const suitSym = suit === "s" ? "♠" : suit === "h" ? "♥" : suit === "d" ? "♦" : "♣";
      return `<span class="mini-card suit-${suit}"><span>${rank}</span><span style="align-self:flex-end;">${suitSym}</span></span>`;
    }

    function updateState(state) {
      document.getElementById("roundId").textContent = state.round_id;
      document.getElementById("phase").textContent = state.phase;
      document.getElementById("pot").textContent = state.pot;
      document.getElementById("activeN").textContent = state.active_n;

      // Board with labels
      const board = state.board || [];
      const labels = ["Flop", "Flop", "Flop", "Turn", "River"];
      let bHtml = "";
      for (let i = 0; i < 5; i++) {
        bHtml += `<div>${renderCard(board[i])}<div class="card-label">${labels[i]}</div></div>`;
      }
      document.getElementById("boardCards").innerHTML = bHtml;

      // Winners Banner
      const winBanner = document.getElementById("winnerBanner");
      if (state.phase === "showdown" && state.winners && state.winners.length > 0) {
        const winDetails = state.winners.map(w => `<strong>${w.nickname}</strong> (+${w.payout} chips) with <em>${w.category || 'High Hand'}</em>`).join(", ");
        winBanner.innerHTML = `🏆 <strong>Showdown Result:</strong> ${winDetails}`;
        winBanner.style.display = "block";
      } else {
        winBanner.style.display = "none";
      }

      // Seats Grid
      const sGrid = document.getElementById("seatsGrid");
      sGrid.innerHTML = state.seats.map(s => {
        const isWinner = state.phase === "showdown" && s.is_winner;
        return `
        <div class="seat-card ${s.active ? 'active' : ''} ${s.folded ? 'folded' : ''} ${isWinner ? 'winner' : ''}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color:var(--accent);">${s.nickname}</span>
            ${isWinner ? `<span class="badge" style="background:#22c55e; color:#052e16; font-weight:bold;">🏆 WINNER (+${s.payout})</span>` : ''}
          </div>
          <div>Chips: <strong>${s.chips}</strong> | ${s.is_bot ? 'Bot' : 'Phone'}</div>
          <div>Status: <span style="font-weight:bold; color:${s.folded ? '#ef4444' : (s.active ? '#22c55e' : '#94a3b8')};">${s.folded ? 'Folded' : (s.active ? 'In Hand' : 'Empty')}</span></div>
          ${s.aided_guess !== null ? `<div style="font-size:0.8rem; color:#34d399; margin-top:2px;">Guess: ${(s.aided_guess*100).toFixed(0)}%</div>` : ''}
          ${state.phase === "showdown" && s.active && !s.folded && s.cards && s.cards.length > 0 ? `
            <div style="margin-top:8px; display:flex; align-items:center; gap:4px;">
              ${s.cards.map(c => renderMiniCard(c)).join("")}
              <span style="font-size:0.8rem; font-weight:bold; margin-left:6px; color:${isWinner ? '#86efac' : '#cbd5e1'};">${s.showdown_cat || ''}</span>
            </div>
          ` : ''}
        </div>
        `;
      }).join("");

      // Leaderboard
      const lbBody = document.getElementById("leaderboardBody");
      const sortedSeats = [...state.seats].filter(s => s.occupied).sort((a,b) => b.chips - a.chips);
      lbBody.innerHTML = sortedSeats.map(s => `
        <tr>
          <td>${s.nickname}</td>
          <td><strong>${s.chips}</strong></td>
          <td>${s.calib_score > 0 ? (s.calib_score*100).toFixed(1) + '%' : '--'}</td>
        </tr>
      `).join("");

      // Crowd Average
      const guesses = state.seats.filter(s => s.unaided_guess !== null).map(s => s.unaided_guess);
      if (guesses.length > 0) {
        const avg = guesses.reduce((a,b)=>a+b,0) / guesses.length;
        document.getElementById("crowdAvg").textContent = `${(avg*100).toFixed(1)}% (n=${guesses.length})`;
      } else {
        document.getElementById("crowdAvg").textContent = "--";
      }
    }

    function sendAction(action) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "host_action", action: action}));
      }
    }

    window.onload = initWS;
  </script>
</body>
</html>
"""

PHONE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hold'em Lab — Mobile Seat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <style>
    :root {
      --bg: #0b1329; --panel: #1e293b; --text: #f8fafc; --text-muted: #94a3b8;
      --accent: #38bdf8; --border: #334155; --spade: #0f172a; --heart: #e11d48;
      --diamond: #0284c7; --club: #16a34a; --success: #22c55e; --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 12px; font-size: 14px; }
    .banner { background: #0369a1; color: white; text-align: center; padding: 6px; font-weight: bold; font-size: 0.75rem; border-radius: 4px; margin-bottom: 10px; }
    .status-bar { display: flex; justify-content: space-between; align-items: center; background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; font-size: 0.85rem; }
    .card-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .cards-row { display: flex; gap: 8px; justify-content: center; margin: 8px 0; }
    .poker-card { width: 54px; height: 78px; background: white; border-radius: 6px; display: flex; flex-direction: column; justify-content: space-between; padding: 4px 6px; font-weight: bold; font-size: 1.15rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .poker-card.suit-s { color: var(--spade); }
    .poker-card.suit-h { color: var(--heart); }
    .poker-card.suit-d { color: var(--diamond); }
    .poker-card.suit-c { color: var(--club); }
    .card-slot { text-align: center; }
    .slot-label { font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; }
    .btn { background: #0284c7; color: white; border: none; padding: 12px; border-radius: 6px; font-weight: bold; font-size: 0.95rem; width: 100%; cursor: pointer; transition: opacity 0.2s; }
    .btn:active { opacity: 0.8; }
    .btn-fold { background: #e11d48; }
    .btn-play { background: #16a34a; }
    .btn-disabled { opacity: 0.5; pointer-events: none; }
    .slider { width: 100%; margin: 12px 0; accent-color: var(--accent); }
    .aid-box { background: #071026; border: 1px solid var(--accent); border-radius: 6px; padding: 10px; font-size: 0.85rem; margin-top: 10px; }
    .recipe { background: #1e293b; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; margin-top: 6px; white-space: pre-wrap; color: #38bdf8; border: 1px solid #334155; }
    .badge { padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background: #334155; }
    .fold-overlay { background: rgba(225, 29, 72, 0.15); border: 1px solid #e11d48; color: #fca5a5; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 8px; }
    .quick-host-bar { background: #111e38; border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin-top: 12px; text-align: center; font-size: 0.8rem; }
  </style>
</head>
<body>
  <div class="banner">EDUCATIONAL DEMO — PLAY CHIPS ONLY — NO REAL MONEY</div>

  <div class="status-bar">
    <div><strong id="seatLabel" style="color:var(--accent);">Joining...</strong></div>
    <div>Phase: <strong id="phaseLabel" style="color:var(--warning); text-transform:uppercase;">WAITING</strong></div>
    <div>Chips: <strong id="chipCount" style="color:#f59e0b;">100</strong></div>
  </div>

  <div id="foldedNotice" class="fold-overlay" style="display:none;">
    ⚠️ YOU FOLDED — Spectating remainder of this hand
  </div>

  <!-- Join Screen -->
  <div id="joinSection" class="card-panel">
    <h3 style="color:var(--accent); margin-bottom:8px;">Take Your Seat</h3>
    <input type="text" id="nicknameInput" placeholder="Enter Nickname" style="width:100%; padding:10px; margin:10px 0; border-radius:6px; border:1px solid var(--border); background:#0f172a; color:white; font-size:1rem;">
    <button class="btn" onclick="joinRoom()">Join Game</button>
  </div>

  <!-- Game Table Screen -->
  <div id="gameSection" style="display:none;">
    <!-- Showdown Winner Announcement Banner -->
    <div id="showdownBanner" style="display:none; margin-bottom:12px; border-radius:8px; padding:12px; text-align:center;"></div>

    <!-- 1. Community Board -->
    <div class="card-panel">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-weight:bold; color:var(--accent); font-size:0.9rem;">Community Board</span>
        <span style="font-size:0.8rem; color:var(--text-muted);">Pot: <strong id="potLabel" style="color:#f59e0b;">0</strong> chips</span>
      </div>
      <div class="cards-row" id="boardCardsRow"></div>
    </div>

    <!-- 2. Private Hole Cards -->
    <div class="card-panel" style="text-align:center;">
      <div style="font-size:0.8rem; color:var(--text-muted);">Your Private Hole Cards</div>
      <div class="cards-row" id="holeCardsRow"></div>
      <div style="font-weight:bold; font-size:1.1rem; color:var(--accent);" id="handClassLabel">--</div>
      <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;" id="handDescLabel">Waiting for deal...</div>
    </div>

    <!-- 3. Postflop Outs & Live Equity (Revealed on Flop/Turn/River) -->
    <div id="postflopPanel" class="card-panel" style="display:none; background:#071026; border-color:#22c55e;">
      <div style="font-weight:bold; color:#22c55e; font-size:0.9rem; margin-bottom:6px;">Postflop Board Analysis</div>
      <div id="drawNotes" style="font-size:0.85rem; font-weight:bold; color:#f8fafc;">--</div>
      <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.85rem;">
        <div>Outs Counter: <strong id="outsCount" style="color:#38bdf8;">0</strong></div>
        <div>Live Board Equity: <strong id="liveEquity" style="color:#34d399;">--</strong></div>
      </div>
      <div id="ruleDesc" style="font-size:0.8rem; color:#94a3b8; margin-top:4px; font-style:italic;">--</div>
    </div>

    <!-- 4. Intuition vs Science: Probability Guess Slider & Aid -->
    <div class="card-panel">
      <div style="font-weight:bold; font-size:0.9rem; color:var(--accent);">Intuition vs Science</div>
      <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Guess your win probability against <span id="nOpponents">N</span> opponents:</div>
      <input type="range" min="1" max="99" value="35" class="slider" id="guessSlider" oninput="updateGuessLabel(this.value)">
      <div style="text-align:center; font-size:1.3rem; font-weight:bold; color:var(--accent); margin:4px 0;"><span id="guessVal">35</span>%</div>
      
      <button class="btn" id="lockGuessBtn" onclick="lockGuess()">🔒 Lock Guess & Reveal Simulation Aid</button>
      <div id="guessLockedBadge" style="display:none; text-align:center; font-size:0.85rem; font-weight:bold; color:#22c55e; margin:6px 0;">
        ✅ Guess Locked: <span id="lockedVal">35%</span>
      </div>

      <!-- Simulation Aid Box -->
      <div id="aidBox" class="aid-box" style="display:none;">
        <div style="font-weight:bold; color:#34d399; margin-bottom:4px;">📘 Simulation & Formula Aid</div>
        <div>Empirical Table Equity: <strong id="empEquity">--</strong> ± <span id="empSE">--</span></div>
        <div>Model M1 Formula Estimate: <strong id="formEquity">--</strong></div>
        <div class="recipe" id="formRecipe">--</div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">13×13 Preflop Chart Coordinate: <span id="chartCoord">--</span></div>
      </div>
    </div>

    <!-- 5. Decision Buttons -->
    <div class="card-panel">
      <div style="font-weight:bold; font-size:0.9rem; margin-bottom:8px;">Preflop Decision</div>
      <div style="display:flex; gap:10px;">
        <button class="btn btn-fold" id="btnFold" onclick="makeDecision('fold')">✋ Fold Hand</button>
        <button class="btn btn-play" id="btnPlay" onclick="makeDecision('play')">⚔️ Play Hand</button>
      </div>
    </div>

    <!-- 6. Mobile Quick Host Controls (Convenience for Single-Device Testing) -->
    <div class="quick-host-bar">
      <div style="font-weight:bold; color:var(--text-muted); margin-bottom:4px;">Host Quick Advance (Demo Testing)</div>
      <div style="display:flex; gap:4px; justify-content:center; flex-wrap:wrap;">
        <button class="badge" style="border:none; cursor:pointer;" onclick="sendHostAction('start')">Deal</button>
        <button class="badge" style="border:none; cursor:pointer;" onclick="sendHostAction('flop')">Flop</button>
        <button class="badge" style="border:none; cursor:pointer;" onclick="sendHostAction('turn')">Turn</button>
        <button class="badge" style="border:none; cursor:pointer;" onclick="sendHostAction('river')">River</button>
        <button class="badge" style="border:none; cursor:pointer;" onclick="sendHostAction('showdown')">Showdown</button>
      </div>
    </div>
  </div>

  <script>
    const room = new URLSearchParams(window.location.search).get("room") || "DEMO";
    let ws;
    let mySeatIdx = null;
    let isFolded = false;

    function updateGuessLabel(val) {
      document.getElementById("guessVal").textContent = val;
    }

    function renderCard(cardStr) {
      if (!cardStr) {
        return `<div class="poker-card" style="opacity:0.25; border:2px dashed #94a3b8; background:transparent;"></div>`;
      }
      const rank = cardStr[0];
      const suit = cardStr[1];
      const suitSym = suit === "s" ? "♠" : suit === "h" ? "♥" : suit === "d" ? "♦" : "♣";
      return `<div class="poker-card suit-${suit}"><div>${rank}</div><div style="align-self:flex-end;">${suitSym}</div></div>`;
    }

    function initWS() {
      const loc = window.location;
      const proto = loc.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${loc.host}/ws?room=${room}&role=player&nickname=${encodeURIComponent(document.getElementById("nicknameInput").value || "Player")}`);

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "joined") {
          mySeatIdx = data.seat_idx;
          document.getElementById("seatLabel").textContent = `${data.nickname} (Seat ${data.seat_idx+1})`;
          document.getElementById("joinSection").style.display = "none";
          document.getElementById("gameSection").style.display = "block";
        } else if (data.type === "update") {
          handleUpdate(data);
        }
      };

      ws.onclose = () => {
        document.getElementById("seatLabel").textContent = "Reconnecting...";
        setTimeout(initWS, 2000);
      };
    }

    function joinRoom() {
      initWS();
    }

    function handleUpdate(data) {
      document.getElementById("phaseLabel").textContent = data.phase;
      document.getElementById("potLabel").textContent = data.pot;
      document.getElementById("chipCount").textContent = data.chips;
      document.getElementById("nOpponents").textContent = Math.max(1, data.active_n - 1);

      // 0. Showdown Winner Announcement
      const sBanner = document.getElementById("showdownBanner");
      if (data.phase === "showdown" && data.winners && data.winners.length > 0) {
        sBanner.style.display = "block";
        const mySd = data.my_showdown;
        const isWin = mySd && mySd.is_winner;
        const myCat = (mySd && mySd.hand_category) || "";
        const winStr = data.winners.map(w => `${w.nickname} (${w.category || 'High Hand'}, +${w.payout} chips)`).join(", ");

        if (isWin) {
          sBanner.style.background = "linear-gradient(135deg, rgba(34,197,94,0.35), rgba(16,185,129,0.2))";
          sBanner.style.border = "2px solid #22c55e";
          sBanner.style.color = "#86efac";
          sBanner.innerHTML = `
            <div style="font-size:1.35rem; font-weight:bold;">🏆 YOU WON THE POT!</div>
            <div style="font-size:1rem; margin-top:4px;">Collected <strong>+${mySd.payout} chips</strong> with <strong>${myCat}</strong></div>
          `;
        } else if (isFolded) {
          sBanner.style.background = "rgba(100, 116, 139, 0.25)";
          sBanner.style.border = "1px solid #64748b";
          sBanner.style.color = "#cbd5e1";
          sBanner.innerHTML = `
            <div style="font-size:1.1rem; font-weight:bold;">⚠️ Hand Complete (You Folded)</div>
            <div style="font-size:0.9rem; margin-top:4px;">Winner: <strong>${winStr}</strong></div>
          `;
        } else {
          sBanner.style.background = "linear-gradient(135deg, rgba(239,68,68,0.25), rgba(185,28,28,0.15))";
          sBanner.style.border = "1px solid #ef4444";
          sBanner.style.color = "#fca5a5";
          sBanner.innerHTML = `
            <div style="font-size:1.25rem; font-weight:bold;">❌ You Lost This Hand</div>
            <div style="font-size:0.9rem; margin-top:4px;">Winner: <strong>${winStr}</strong></div>
            ${myCat ? `<div style="font-size:0.85rem; color:#cbd5e1; margin-top:4px;">Your Final Hand: <strong>${myCat}</strong></div>` : ''}
          `;
        }
      } else {
        sBanner.style.display = "none";
      }

      // 1. Community Board
      const bRow = document.getElementById("boardCardsRow");
      const board = data.board || [];
      const labels = ["Flop", "Flop", "Flop", "Turn", "River"];
      let bHtml = "";
      for (let i = 0; i < 5; i++) {
        bHtml += `<div class="card-slot">${renderCard(board[i])}<div class="slot-label">${labels[i]}</div></div>`;
      }
      bRow.innerHTML = bHtml;

      // 2. Hole Cards
      const hRow = document.getElementById("holeCardsRow");
      const cards = data.cards || [];
      if (cards.length > 0) {
        hRow.innerHTML = cards.map(c => `<div class="card-slot">${renderCard(c)}</div>`).join("");
        if (data.phase === "showdown" && data.my_showdown && data.my_showdown.hand_category) {
          document.getElementById("handClassLabel").textContent = data.my_showdown.hand_category;
          document.getElementById("handDescLabel").textContent = "Best 5-card combination at showdown.";
        } else {
          document.getElementById("handClassLabel").textContent = data.hand_class;
          document.getElementById("handDescLabel").textContent = data.hand_desc;
        }
      } else {
        hRow.innerHTML = `<div class="card-slot">${renderCard(null)}</div><div class="card-slot">${renderCard(null)}</div>`;
        document.getElementById("handClassLabel").textContent = "--";
        document.getElementById("handDescLabel").textContent = "Waiting for next deal...";
      }

      // 3. Fold state
      isFolded = data.folded;
      document.getElementById("foldedNotice").style.display = isFolded ? "block" : "none";
      if (isFolded) {
        document.getElementById("btnFold").classList.add("btn-disabled");
        document.getElementById("btnPlay").classList.add("btn-disabled");
        hRow.style.opacity = "0.35";
      } else {
        document.getElementById("btnFold").classList.remove("btn-disabled");
        document.getElementById("btnPlay").classList.remove("btn-disabled");
        hRow.style.opacity = "1.0";
      }

      // 4. Preflop Aid Data
      document.getElementById("empEquity").textContent = (data.emp_equity * 100).toFixed(1) + "%";
      document.getElementById("empSE").textContent = (data.emp_se * 100).toFixed(2) + "%";
      document.getElementById("formEquity").textContent = (data.form_equity * 100).toFixed(1) + "%";
      document.getElementById("formRecipe").textContent = data.form_recipe || "--";
      if (data.chart_coord) {
        document.getElementById("chartCoord").textContent = `Row ${data.chart_coord[0]}, Col ${data.chart_coord[1]}`;
      }

      // 5. Postflop Analysis
      const pf = data.postflop;
      const pfPanel = document.getElementById("postflopPanel");
      if (pf && Object.keys(pf).length > 0 && !isFolded) {
        pfPanel.style.display = "block";
        document.getElementById("drawNotes").textContent = pf.draw_notes.join(" + ");
        document.getElementById("outsCount").textContent = pf.total_outs;
        document.getElementById("liveEquity").textContent = (pf.live_equity * 100).toFixed(1) + "%";
        document.getElementById("ruleDesc").textContent = pf.rule_desc;
      } else {
        pfPanel.style.display = "none";
      }

      // 6. Reset or preserve lock button state
      if (data.phase === "preflop" && !data.aid_revealed && !data.unaided_guess) {
        document.getElementById("aidBox").style.display = "none";
        document.getElementById("lockGuessBtn").style.display = "block";
        document.getElementById("guessLockedBadge").style.display = "none";
      }
    }

    function lockGuess() {
      const g = parseFloat(document.getElementById("guessSlider").value) / 100.0;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "guess_unaided", guess: g}));
      }
      document.getElementById("aidBox").style.display = "block";
      document.getElementById("lockGuessBtn").style.display = "none";
      document.getElementById("lockedVal").textContent = (g * 100).toFixed(0) + "%";
      document.getElementById("guessLockedBadge").style.display = "block";
    }

    function makeDecision(action) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "decision", action: action}));
        if (action === "fold") {
          document.getElementById("foldedNotice").style.display = "block";
          document.getElementById("btnFold").classList.add("btn-disabled");
          document.getElementById("btnPlay").classList.add("btn-disabled");
        }
      }
    }

    function sendHostAction(action) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({type: "host_action", action: action}));
      }
    }
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI Application & Routes
# ---------------------------------------------------------------------------

app = FastAPI(title="Hold'em Lab Classroom Demo")


@app.get("/", response_class=HTMLResponse)
async def host_view(request: Request, room: str = "DEMO"):
    return HTMLResponse(HOST_HTML)


@app.get("/join", response_class=HTMLResponse)
@app.get("/play", response_class=HTMLResponse)
async def player_view(request: Request, room: str = "DEMO"):
    return HTMLResponse(PHONE_HTML)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "time": time.time()}


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    room: str = Query("DEMO"),
    role: str = Query("player"),
    nickname: str = Query("Player"),
):
    await websocket.accept()
    game = get_or_create_room(room)
    lan_ip = get_lan_ip()
    port = websocket.url.port or 8000
    join_url = f"http://{lan_ip}:{port}/join?room={room}"

    if role == "host":
        game.host_sockets.add(websocket)
        qr_url = generate_qr_data_url(join_url)
        await websocket.send_json({
            "type": "init",
            "join_url": join_url,
            "qr_url": qr_url,
            "room": room,
        })
        await game.broadcast_state()

        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "start":
                    game.start_round()
                elif action == "flop":
                    game.deal_flop()
                elif action == "turn":
                    game.deal_turn()
                elif action == "river":
                    game.deal_river()
                elif action == "showdown":
                    game.showdown()
                await game.broadcast_state()
        except WebSocketDisconnect:
            game.host_sockets.discard(websocket)

    else:
        # Player or Spectator
        sanitized = sanitize_nickname(nickname)
        assigned_seat: Optional[Seat] = None

        # Find first available seat
        for s in game.seats:
            if not s.occupied or s.is_bot:
                assigned_seat = s
                break

        if assigned_seat is None:
            # Spectator mode when all 9 seats occupied by humans
            game.spectator_sockets.add(websocket)
            await websocket.send_json({
                "type": "spectator",
                "message": "Room seats full (9/9). Connected as Spectator.",
            })
            await game.broadcast_state()
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                game.spectator_sockets.discard(websocket)
            return

        assigned_seat.occupied = True
        assigned_seat.is_bot = False
        assigned_seat.nickname = sanitized
        assigned_seat.player_id = f"user_{assigned_seat.seat_idx}"
        game.player_sockets[assigned_seat.seat_idx] = websocket

        await websocket.send_json({
            "type": "joined",
            "seat_idx": assigned_seat.seat_idx,
            "nickname": sanitized,
        })

        # Immediately broadcast updated room state to all
        await game.broadcast_state()

        try:
            while True:
                data = await websocket.receive_json()
                m_type = data.get("type")
                if m_type == "guess_unaided":
                    assigned_seat.unaided_guess = float(data.get("guess", 0.5))
                    assigned_seat.aid_revealed = True
                elif m_type == "guess_aided":
                    assigned_seat.aided_guess = float(data.get("guess", 0.5))
                elif m_type == "decision":
                    if data.get("action") == "fold":
                        assigned_seat.folded = True
                elif m_type == "host_action":
                    # Allow host actions triggered from mobile quick drawer
                    action = data.get("action")
                    if action == "start":
                        game.start_round()
                    elif action == "flop":
                        game.deal_flop()
                    elif action == "turn":
                        game.deal_turn()
                    elif action == "river":
                        game.deal_river()
                    elif action == "showdown":
                        game.showdown()
                await game.broadcast_state()
        except WebSocketDisconnect:
            game.player_sockets.pop(assigned_seat.seat_idx, None)
            assigned_seat.occupied = False
            assigned_seat.is_bot = True
            assigned_seat.nickname = f"Bot-TAG-{assigned_seat.seat_idx+1}"
            await game.broadcast_state()


# ---------------------------------------------------------------------------
# Pre-Flight Selftest & CLI
# ---------------------------------------------------------------------------


def run_selftest() -> int:
    """Run pre-flight PASS/FAIL checklist for live demo deployment."""
    print("=== Hold'em Lab Interactive Demo Pre-Flight Selftest ===")
    checks_passed = 0
    total_checks = 5

    # Check 1: Required Data Files
    eq_ok = (FIGURES_DATA_DIR / "equity_table.csv").exists()
    fp_ok = (DATA_DIR / "formula_params.json").exists()
    if eq_ok and fp_ok:
        print("[PASS] 1. Data files present (figures/data/equity_table.csv, data/formula_params.json)")
        checks_passed += 1
    else:
        print(f"[FAIL] 1. Missing data files: equity_table.csv={eq_ok}, formula_params.json={fp_ok}")

    # Check 2: LAN IP & QR Code Generation
    lan_ip = get_lan_ip()
    qr_data = generate_qr_data_url(f"http://{lan_ip}:8000/join?room=TEST")
    if lan_ip and qr_data.startswith("data:image/png;base64,"):
        print(f"[PASS] 2. LAN IP detected: {lan_ip}, in-memory QR code generated successfully")
        checks_passed += 1
    else:
        print(f"[FAIL] 2. IP detection or QR generation failed (IP: {lan_ip})")

    # Check 3: In-Memory 9-Seat Bots Game Simulation
    room = GameRoom("SELFTEST", seed=42, bots_count=9)
    try:
        total_starting_chips = sum(s.chips for s in room.seats)
        for _ in range(5):
            room.start_round()
            room.deal_flop()
            room.deal_turn()
            room.deal_river()
            room.showdown()
        total_ending_chips = sum(s.chips for s in room.seats)
        chips_conserved = (total_starting_chips == total_ending_chips)
        if chips_conserved and room.round_id == 5:
            print("[PASS] 3. Simulated 9-seat bots-only session completed 5 rounds")
            checks_passed += 1
        else:
            print(f"[FAIL] 3. Bot simulation failed (Chips conserved: {chips_conserved})")
    except Exception as e:
        print(f"[FAIL] 3. Bot simulation exception: {e}")

    # Check 4: Chip Conservation Verification
    if chips_conserved:
        print(f"[PASS] 4. Exact chip conservation verified: {total_starting_chips} == {total_ending_chips} chips")
        checks_passed += 1
    else:
        print(f"[FAIL] 4. Chip conservation mismatch: {total_starting_chips} vs {total_ending_chips}")

    # Check 5: FastAPI Application Routing Verification
    routes = [r.path for r in app.routes]
    expected_routes = {"/", "/join", "/play", "/ws", "/api/health"}
    if expected_routes.issubset(set(routes)):
        print(f"[PASS] 5. FastAPI server routes verified ({', '.join(expected_routes)})")
        checks_passed += 1
    else:
        print(f"[FAIL] 5. Missing routes: {expected_routes - set(routes)}")

    print(f"\nResult: {checks_passed}/{total_checks} checks passed.")
    if checks_passed == total_checks:
        print("SELFTEST STATUS: PASS")
        return 0
    else:
        print("SELFTEST STATUS: FAIL")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold'em Lab Classroom Probability Demo")
    parser.add_argument("--bots", type=int, default=6, help="Number of bot players (default: 6)")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds to run (default: 10)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--room", type=str, default="DEMO", help="Room code (default: DEMO)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--selftest", action="store_true", help="Run pre-flight checklist and exit")
    parser.add_argument("--record", action="store_true", help="Record game events to data/demo_logs/")
    parser.add_argument("--from-recording", type=str, default=None, help="Replay events from a log file")
    args = parser.parse_args()

    load_demo_data()

    if args.selftest:
        sys.exit(run_selftest())

    room = get_or_create_room(args.room, seed=args.seed, bots_count=args.bots)
    if args.record:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        room.log_file = LOGS_DIR / f"session_{args.room}_{ts}.jsonl"
        print(f"Logging demo session to: {room.log_file}")

    lan_ip = get_lan_ip()
    print("=" * 60)
    print(" Hold'em Lab — Classroom Probability Demo (Play Chips Only)")
    print("=" * 60)
    print(f" Host Dashboard:   http://localhost:{args.port}/?room={args.room}")
    print(f" Mobile Phone Join: http://{lan_ip}:{args.port}/join?room={args.room}")
    print(f" Room Code:        {args.room}")
    print(f" Initial Bots:     {args.bots} active seats")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
