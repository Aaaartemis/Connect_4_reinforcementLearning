"""
Dense reward shaping for the learning agent (plays as X).

Uses the same win lines as the game board. Rewards building own 2- and 3-in-a-row
threats (empty cells only on that line), blocking opponent threats, and penalises
missing an immediate win or a unique forced block.
"""

from __future__ import annotations

WIN_LINES = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (3, 6, 9, 12),
]

MARK_X = " X "
MARK_O = " O "
EMPTY = "   "

# Keep small relative to terminal rewards (±1.0)
SHAPE_OWN2 = 0.03
SHAPE_OWN3 = 0.08
SHAPE_BLOCK2 = 0.03
SHAPE_BLOCK3 = 0.08
PENALTY_MISS_WIN = 0.12
PENALTY_MISS_BLOCK = 0.12


def _line_counts(board, line):
    cells = [board[i] for i in line]
    nx = sum(1 for c in cells if c == MARK_X)
    no = sum(1 for c in cells if c == MARK_O)
    ne = sum(1 for c in cells if c == EMPTY)
    return nx, no, ne


def _lines_through(move_idx):
    return [ln for ln in WIN_LINES if move_idx in ln]


def _x_wins_if_play(board, move_idx):
    if board[move_idx] != EMPTY:
        return False
    b = list(board)
    b[move_idx] = MARK_X
    for line in WIN_LINES:
        if all(b[i] == MARK_X for i in line):
            return True
    return False


def _winning_moves_for_x(board):
    return [i for i in range(16) if board[i] == EMPTY and _x_wins_if_play(board, i)]


def _forced_block_squares_for_x(board):
    """
    Squares where O currently has 3 in a line with exactly one empty (X must play
    there to stop immediate O win on O's next turn).
    """
    squares = []
    for line in WIN_LINES:
        nx, no, ne = _line_counts(board, line)
        if no == 3 and nx == 0 and ne == 1:
            for idx in line:
                if board[idx] == EMPTY:
                    squares.append(idx)
                    break
    return squares


def compute_agent_move_shaping(board_before, move, board_after):
    """
    board_before: list of 16 cells before X moves.
    move: index X played.
    board_after: list of 16 cells after X's stone is placed.
    """
    reward = 0.0
    lines = _lines_through(move)

    # --- Positive: own threats on lines through this move (no O on that line) ---
    for line in lines:
        nx, no, ne = _line_counts(board_after, line)
        if no != 0:
            continue
        if nx == 2 and ne == 2:
            reward += SHAPE_OWN2
        elif nx == 3 and ne == 1:
            reward += SHAPE_OWN3

    # --- Positive: blocked opponent threats (evaluated on board_before) ---
    for line in lines:
        nx, no, ne = _line_counts(board_before, line)
        if board_before[move] != EMPTY:
            continue
        # O has 3, one empty, X plays that empty
        if no == 3 and nx == 0 and ne == 1:
            reward += SHAPE_BLOCK3
        # O has 2, two empty, X occupies one of those empties on this line
        elif no == 2 and nx == 0 and ne == 2:
            reward += SHAPE_BLOCK2

    # --- Penalties: miss immediate win or unique forced block ---
    win_moves = _winning_moves_for_x(board_before)
    if win_moves:
        if move not in win_moves:
            reward -= PENALTY_MISS_WIN
    else:
        block_squares = _forced_block_squares_for_x(board_before)
        unique = set(block_squares)
        if len(unique) == 1:
            only = next(iter(unique))
            if move != only:
                reward -= PENALTY_MISS_BLOCK
        # If multiple disjoint forced blocks, X cannot stop all in one move — no penalty.

    return reward
