# Evaluation versus random opponent only (for comparability)
import pickle
import sys
from game import Game
from player import Player

q_table_filename = "q_table_final.pkl"

try:
    with open(q_table_filename, "rb") as f:
        loaded_q_table = pickle.load(f)
except OSError:
    print(f"Could not open {q_table_filename}. Train the agent first by running training.py.")
    sys.exit(1)

# Agent is player 0 (X), human is player 1 (O)
agent = Player("Agent", "agent")

agent.epsilon = 0.0
eval = Player("Eval", "random")
game_eval = Game([agent, eval])
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
