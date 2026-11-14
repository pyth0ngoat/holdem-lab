"""Hold'em Lab - Applied Probability Distributions (THM6242)
Package initialization and configuration.
"""

import os
from pathlib import Path

__version__ = "0.1.0"

# Workspace directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
UCI_DATA_DIR = EXTERNAL_DATA_DIR / "uci"
PHH_DATA_DIR = EXTERNAL_DATA_DIR / "phh"
CACHE_DIR = DATA_DIR / "cache"
DEMO_LOGS_DIR = DATA_DIR / "demo_logs"
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DATA_DIR = FIGURES_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure runtime directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEMO_LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODE = os.environ.get("HOLDEM_MODE", "quick").lower()
if MODE not in ("quick", "full"):
    MODE = "quick"

# Random Seeds (distinct for fitting, validation, demo)
SEED_FIT = 42
SEED_VALIDATION = 1337
SEED_DEMO = 2026

# Table sizes under study
N_PLAYERS_RANGE = list(range(2, 10))  # n = 2..9

# Monte Carlo sample sizes per mode
MC_CONFIG = {
    "quick": {
        "equity_precision": 0.003,      # target precision ~ 0.3 percentage points
        "equity_mc_hands": 10_000,
        "g1_tables_per_n": 50_000,
        "clt_samples": 5_000,
        "benchmark_hands": 100_000,
    },
    "full": {
        "equity_precision": 0.001,      # target precision ~ 0.1 percentage points
        "equity_mc_hands": 100_000,
        "g1_tables_per_n": 500_000,
        "clt_samples": 50_000,
        "benchmark_hands": 2_000_000,
    },
}

# Formula tolerance: 1.0 percentage points = 0.01
FORMULA_TOLERANCE_PP = 1.0
FORMULA_TOLERANCE = 0.01

CONFIG = {
    "mode": MODE,
    "seeds": {
        "fit": SEED_FIT,
        "validation": SEED_VALIDATION,
        "demo": SEED_DEMO,
    },
    "n_range": N_PLAYERS_RANGE,
    "formula_tolerance_pp": FORMULA_TOLERANCE_PP,
    "formula_tolerance": FORMULA_TOLERANCE,
    "mc": MC_CONFIG[MODE],
    "paths": {
        "root": ROOT_DIR,
        "data": DATA_DIR,
        "cache": CACHE_DIR,
        "figures": FIGURES_DIR,
        "figures_data": FIGURES_DATA_DIR,
        "reports": REPORTS_DIR,
        "uci": UCI_DATA_DIR,
        "phh": PHH_DATA_DIR,
    },
}
