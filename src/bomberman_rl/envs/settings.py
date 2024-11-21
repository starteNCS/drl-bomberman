import logging
from pathlib import Path
import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 17
ROWS = 17
SCENARIOS = {
    # modes useful for agent development
    "empty": {"CRATE_DENSITY": 0, "COIN_COUNT": 0},
    "coin-heaven": {"CRATE_DENSITY": 0, "COIN_COUNT": 50},
    "loot-crate": {"CRATE_DENSITY": 0.75, "COIN_COUNT": 50},
    # this is the tournament game mode
    "classic": {"CRATE_DENSITY": 0.75, "COIN_COUNT": 9},
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400


# GUI properties
SCALE = 2
GRID_SIZE = 30 * SCALE
WIDTH = 1000 * SCALE
HEIGHT = 600 * SCALE
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent.parent / "assets"  # TODO

AGENT_COLORS = ["green", "blue", "pink", "yellow"]

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_RETURN: "WAIT",
    pygame.K_SPACE: "BOMB",
}

# Logging levels
LOG_GAME = logging.DEBUG
LOG_AGENT_WRAPPER = logging.INFO
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Recording
VIDEO_DIR = Path(__file__).parent / "logs" / "videos"
