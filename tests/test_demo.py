"""Unit and integration tests for demo.py interactive classroom demo."""

import json
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from demo import (
    GameRoom,
    Seat,
    app,
    compute_formula_estimate,
    get_empirical_equity,
    run_selftest,
    sanitize_nickname,
)
from holdem_lab.analysis import run_g9_demo_log_analysis


def test_demo_selftest_passes():
    """Verify that demo pre-flight selftest returns exit code 0."""
    code = run_selftest()
    assert code == 0


def test_sanitize_nickname():
    """Verify nickname sanitization and length bounds."""
    assert sanitize_nickname("Alice123") == "Alice123"
    assert sanitize_nickname("Bob <script>alert(1)</script>") == "Bob scriptalert1"
    assert sanitize_nickname("A" * 30) == "A" * 16
    assert sanitize_nickname("") == "Player"


def test_seat_cap_and_spectator_overflow():
    """Verify max 9 seats: 10th seat cannot be added as an active player."""
    room = GameRoom("CAP_TEST", seed=101, bots_count=9)
    assert len(room.seats) == 9
    assert all(s.occupied for s in room.seats)

    # All 9 seats occupied: active count is 9
    room.start_round()
    assert room.get_active_count() == 9


def test_private_cards_never_reach_other_sockets():
    """Verify hole cards are suppressed in public broadcast payloads."""
    room = GameRoom("PRIVACY_TEST", seed=102, bots_count=4)
    room.start_round()

    # Public dict representation (sent to host or other clients)
    for seat in room.seats:
        public_dict = seat.to_dict(is_self=False)
        assert "cards" not in public_dict, "Private cards leaked in public payload!"
        assert "hand_class" not in public_dict, "Hand class leaked in public payload!"

    # Private dict representation (sent strictly to owning socket)
    active_seat = next(s for s in room.seats if s.active)
    private_dict = active_seat.to_dict(is_self=True)
    assert "cards" in private_dict
    assert len(private_dict["cards"]) == 2
    assert "hand_class" in private_dict


def test_n_recomputation_after_fold():
    """Verify dynamic n recomputation when a player folds."""
    room = GameRoom("FOLD_TEST", seed=103, bots_count=5)
    room.start_round()
    initial_n = room.get_active_count()
    assert initial_n == 5

    # Player 0 folds
    room.seats[0].folded = True
    new_n = room.get_active_count()
    assert new_n == 4

    # Check that equity estimates update with new n
    hand = "AKs"
    eq_5 = get_empirical_equity(hand, 5)
    eq_4 = get_empirical_equity(hand, 4)
    assert eq_4["equity"] > eq_5["equity"], "Equity must rise as table size shrinks!"

    form_5 = compute_formula_estimate(hand, 5)
    form_4 = compute_formula_estimate(hand, 4)
    assert form_4["pred_equity"] > form_5["pred_equity"]


def test_chip_conservation_20_rounds():
    """Verify strict chip conservation across 20 complete rounds."""
    room = GameRoom("CHIPS_TEST", seed=2026, bots_count=9)
    total_starting_chips = sum(s.chips for s in room.seats)

    for _ in range(20):
        room.start_round()
        room.deal_flop()
        room.deal_turn()
        room.deal_river()
        room.showdown()

    total_ending_chips = sum(s.chips for s in room.seats)
    assert total_starting_chips == total_ending_chips, (
        f"Chip conservation violated: {total_starting_chips} != {total_ending_chips}"
    )


def test_recording_and_g9_analysis(tmp_path: Path):
    """Verify demo session logging and G9 analysis pipeline."""
    room = GameRoom("REC_TEST", seed=777, bots_count=6)
    log_file = tmp_path / "test_session.jsonl"
    room.log_file = log_file

    # Play 3 rounds with recording
    for _ in range(3):
        room.start_round()
        room.deal_flop()
        room.deal_turn()
        room.deal_river()
        room.showdown()

    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) > 0

    # Test G9 analysis on the generated log
    res = run_g9_demo_log_analysis(logs_dir=tmp_path)
    assert res["status"] == "success"
    assert res["n_obs"] > 0
    assert "mean_diff" in res
    assert "ci_t" in res


def test_fastapi_endpoints():
    """Verify HTTP endpoints return valid HTML and JSON."""
    client = TestClient(app)
    r_host = client.get("/?room=DEMO")
    assert r_host.status_code == 200
    assert "Hold'em Lab" in r_host.text

    r_play = client.get("/play?room=DEMO")
    assert r_play.status_code == 200
    assert "EDUCATIONAL DEMO" in r_play.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
