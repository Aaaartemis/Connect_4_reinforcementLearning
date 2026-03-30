import random
import pickle

from game import Game
from player import Player


TOTAL_GAMES = 40_000_000

# Opponent mix: Random 50%, Self-snapshot 20%, Heuristic 20%, Minimax 10%
OPPONENT_MODES = [
    ("random", 0.5),
    ("self_snapshot", 0.2),
    ("heuristic", 0.2),
    ("minimax", 0.1),
]

SNAPSHOT_INTERVAL = 200_000
MAX_SNAPSHOTS = 5

# Reward shaping constants.
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
CREATE_OPEN_2_REWARD = 0.4
CREATE_OPEN_3_REWARD = 0.8

Q_TABLE_OUTPUT = "q_table_training2_create_only.pkl"

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


def choose_mode():
    r = random.random()
    cumulative = 0.0
    for mode, prob in OPPONENT_MODES:
        cumulative += prob
        if r <= cumulative:
            return mode
    return OPPONENT_MODES[-1][0]


def valid_moves_from_state(state):
    return [i for i, cell in enumerate(state) if cell == "   "]


def apply_action_to_state(state, action_idx, mark):
    next_state = list(state)
    next_state[action_idx] = mark
    return next_state


def count_open_sequences(board, mark, length):
    opp_mark = " O " if mark == " X " else " X "
    count = 0

    for line in WIN_LINES:
        line_cells = [board[idx] for idx in line]
        if opp_mark in line_cells:
            continue
        if line_cells.count(mark) == length:
            count += 1

    return count


def episode_shaping_reward(states, mark):
    """
    Build shaping reward from each agent move stored as (state, act_id).

    - +create rewards when own open 2/3 lines increase after the move.
    """
    total = 0.0

    for state, act_id in states:
        valid_moves = valid_moves_from_state(state)
        if act_id < 0 or act_id >= len(valid_moves):
            continue

        action_idx = valid_moves[act_id]
        before = list(state)
        after = apply_action_to_state(before, action_idx, mark)

        own_open_2_before = count_open_sequences(before, mark, 2)
        own_open_2_after = count_open_sequences(after, mark, 2)
        own_open_3_before = count_open_sequences(before, mark, 3)
        own_open_3_after = count_open_sequences(after, mark, 3)

        create_open_2 = max(0, own_open_2_after - own_open_2_before)
        create_open_3 = max(0, own_open_3_after - own_open_3_before)

        total += create_open_2 * CREATE_OPEN_2_REWARD
        total += create_open_3 * CREATE_OPEN_3_REWARD

    return total


def main():
    p_agent = Player("Agent", "agent", 0.9, 0.9999999)
    snapshots = []

    for game_idx in range(1, TOTAL_GAMES + 1):
        mode = choose_mode()

        if mode == "random":
            opponent = Player("Random", "random")
        elif mode == "heuristic":
            opponent = Player("Heuristic", "heuristic")
        elif mode == "minimax":
            opponent = Player("Minimax", "minimax")
        elif mode == "self_snapshot" and snapshots:
            opponent = Player("Snapshot", "eval_agent")
            opponent.q_table = random.choice(snapshots)
            opponent.epsilon = 0.0
        else:
            # Fallback if no snapshots yet
            opponent = Player("Random", "random")

        game = Game([p_agent, opponent])
        game.play_game()

        # Combine terminal reward with move-level shaping.
        if game.winner == 0:
            terminal_reward = WIN_REWARD
        elif game.winner == 1:
            terminal_reward = LOSS_REWARD
        else:
            terminal_reward = DRAW_REWARD

        mark = p_agent.marker if p_agent.marker is not None else " X "
        shaped_reward = terminal_reward + episode_shaping_reward(p_agent.states, mark)

        if p_agent.strategy == "agent":
            p_agent.q_table_update(shaped_reward)

        game.reset_board()
        p_agent.decay_epsilon()

        # Print progress every 10,000 games
        if game_idx % 10000 == 0:
            print(f"Completed {game_idx} games, epsilon: {p_agent.epsilon:.4f}")

        # Snapshot current policy periodically
        if game_idx % SNAPSHOT_INTERVAL == 0:
            snapshots.append(p_agent.q_table.copy())
            if len(snapshots) > MAX_SNAPSHOTS:
                snapshots.pop(0)

    # Save the trained Q-table
    with open(Q_TABLE_OUTPUT, "wb") as f:
        pickle.dump(p_agent.q_table, f)
    print(f"Q-table saved to {Q_TABLE_OUTPUT}")

    # Evaluation versus random opponent only (for comparability)
    p_agent.epsilon = 0.0
    eval_random = Player("EvalRandom", "random")
    game_eval = Game([p_agent, eval_random])
    wins = 0
    ties = 0
    losses = 0
    for _ in range(1000):
        game_eval.play_game()
        if game_eval.winner == 0:
            wins += 1
        elif game_eval.winner == 1:
            losses += 1
        else:
            ties += 1
        game_eval.reset_board()

    print(" ----- AGENT SUMMARY ----- ")
    print("Wins: ", wins)
    print("Losses: ", losses)
    print("Ties: ", ties)


if __name__ == "__main__":
    main()