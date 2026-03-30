# 4x4 Connect 4 Reinforcement Learning Agent

This project implements a reinforcement learning agent that learns to play 4x4 Connect 4 (a variant of Tic-Tac-Toe on a 4x4 grid) using Tabular Q-learning with reward shaping. The agent is trained against various opponents and can be played against through a graphical interface.

## Features

- **Tabular Q-Learning Agent**: Trained using reinforcement learning with epsilon-greedy exploration
- **Multiple Opponent Types**: Random, heuristic, minimax, and self-play snapshots
- **Reward Shaping**: Enhanced learning through intermediate rewards for creating threats
- **Graphical Interface**: Play against the trained AI using Pygame
- **Evaluation Tools**: Test agent performance against different opponents

## Project Structure

- `game.py` - Core game logic for 4x4 Connect 4
- `player.py` - Player implementations (human, random, Q-agent, heuristic, minimax)
- `training_final.py` - Main training script with reward shaping
- `play_against_ai.py` - Pygame GUI for human vs AI gameplay
- `evaluation.py` - Agent evaluation against random opponent
- `shaping.py` - Reward shaping utilities
- `training_1.py` & `training_2.py` - Alternative training approaches

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pygame numpy
   ```

## Usage

### Training the Agent

Run the training script to train the Q-learning agent:

```bash
python training_final.py
```

This will train the agent for 40 million games against a mix of opponents (40% random, 20% heuristic, 20% minimax, 20% self-play snapshots). The trained Q-table will be saved as `q_table_training2_create_only.pkl`.

### Playing Against the AI

After training, or using the pre-trained model by unzipping the q_table_final.zip file, play against the AI using the graphical interface:

```bash
python play_against_ai.py
```

**Controls:**
- Click on empty cells to make your move (O)
- Press 'R' to restart the game
- Press 'Esc' to quit

The AI plays as X and moves first.

### Evaluation

Evaluate the trained agent's performance:

```bash
python evaluation.py
```

This runs 1000 games against a random opponent and reports win/loss/tie statistics.

## Game Rules

4x4 Connect 4 is played on a 4x4 grid where players take turns placing pieces. The first player to get 4 in a row (horizontally, vertically, or diagonally) wins. If the board fills up without a winner, it's a draw.

## Training Details

- **Algorithm**: Q-learning with Monte Carlo updates
- **Exploration**: Epsilon-greedy with exponential decay (starts at 0.9, decays to 0.01)
- **Opponents**: Mixed curriculum learning with different opponent types
- **Reward Shaping**: Additional rewards for creating open 2-in-a-row and 3-in-a-row threats
- **Snapshots**: Periodic snapshots of the Q-table for self-play

## Technical Details

- **State Representation**: Tuple of board state (16 cells)
- **Action Space**: Valid moves (empty cells)
- **Q-Table**: Dictionary mapping states to action values
- **Learning Rate**: 0.1 (alpha)
- **Discount Factor**: Not used (Monte Carlo updates)

## Dependencies

- Python 3.7+
- Pygame
- NumPy
- Pickle (built-in)

## License

This project is for educational purposes. Feel free to use and modify the code.
