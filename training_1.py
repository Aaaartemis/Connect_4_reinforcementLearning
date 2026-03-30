import random
import pickle

from game import Game
from player import Player


TOTAL_GAMES = 20_000_000

# Opponent mix: Random 50%, Self-snapshot 20%, Heuristic 20%, Minimax 10%
OPPONENT_MODES = [
    ("random", 0.5),
    ("self_snapshot", 0.2),
    ("heuristic", 0.2),
    ("minimax", 0.1),
]

SNAPSHOT_INTERVAL = 200_000
MAX_SNAPSHOTS = 5


def choose_mode():
    r = random.random()
    cumulative = 0.0
    for mode, prob in OPPONENT_MODES:
        cumulative += prob
        if r <= cumulative:
            return mode
    return OPPONENT_MODES[-1][0]


def main():
    p_agent = Player("Agent", "agent", 0.9, 0.9995)
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

        if game.winner == 0:
            if p_agent.strategy == "agent":
                p_agent.q_table_update(1.0)
        elif game.winner == 1:
            if p_agent.strategy == "agent":
                p_agent.q_table_update(-1.0)
        else:
            if p_agent.strategy == "agent":
                p_agent.q_table_update(0.2)

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
    with open("q_table.pkl", "wb") as f:
        pickle.dump(p_agent.q_table, f)
    print("Q-table saved to q_table.pkl")

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