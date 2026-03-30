import random
import numpy as np


class Player:
    """
    Generic player that can act as:
    - human: asks for console input (used only in CLI mode)
    - random: uniform random move
    - agent: learning Q-agent (epsilon-greedy during training)
    - eval_agent: fixed Q-agent (greedy, no learning)
    - heuristic: hand-crafted opponent (win/block, otherwise random)
    - minimax: depth-limited minimax / alpha-beta opponent
    """

    def __init__(self, name, strategy, epsilon=0.9, decay_rate=0.999):
        self.name = name
        self.strategy = strategy
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.states = []  # (state, act_id) pairs visited this episode
        # marker is set by Game: " X " for player 0, " O " for player 1
        self.marker = None

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)

    def _get_valid_moves(self, board):
        return [i for i, cell in enumerate(board) if cell == "   "]

    def _check_winner_on_board(self, board):
        win_conditions = [
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
        for (i, j, k, l) in win_conditions:
            if board[i] != "   " and board[i] == board[j] == board[k] == board[l]:
                # return 0 if X wins, 1 if O wins
                return 0 if board[i] == " X " else 1
        if "   " not in board:
            return 2  # draw
        return None

    def _offensive_move_score(self, board, move):
        """Prefer moves that create stronger own-line pressure."""
        my_mark = self.marker if self.marker is not None else " X "
        temp = list(board)
        temp[move] = my_mark

        lines = [
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

        score = 0.0
        center_bonus = {5: 0.15, 6: 0.15, 9: 0.15, 10: 0.15}
        score += center_bonus.get(move, 0.0)

        for line in lines:
            if move not in line:
                continue
            marks = [temp[idx] for idx in line]
            if any(v not in ("   ", my_mark) for v in marks):
                continue
            my_count = sum(1 for v in marks if v == my_mark)
            if my_count == 4:
                score += 10.0
            elif my_count == 3:
                score += 2.0
            elif my_count == 2:
                score += 0.5

        return score

    def _select_q_move(self, board, explore):
        valid_moves = self._get_valid_moves(board)
        state = tuple(board)
        if state not in self.q_table:
            self.q_table[state] = [0 for _ in range(len(valid_moves))]

        my_mark = self.marker if self.marker is not None else " X "

        # Always take a forced immediate win.
        for move in valid_moves:
            temp = list(board)
            temp[move] = my_mark
            winner = self._check_winner_on_board(temp)
            if winner == (0 if my_mark == " X " else 1):
                move_id = valid_moves.index(move)
                if self.strategy == "agent":
                    self.states.append((state, move_id))
                return move

        if explore and random.random() < self.epsilon:
            move_id = random.randint(0, len(valid_moves) - 1)
            move = valid_moves[move_id]
        else:
            q_values = self.q_table[state]
            best_q = max(q_values)
            best_ids = [i for i, q in enumerate(q_values) if q == best_q]
            if len(best_ids) == 1:
                move_id = best_ids[0]
            else:
                scored = []
                for idx in best_ids:
                    m = valid_moves[idx]
                    scored.append((self._offensive_move_score(board, m), idx))
                top_score = max(s for s, _ in scored)
                top_ids = [idx for s, idx in scored if s == top_score]
                move_id = random.choice(top_ids)
            move = valid_moves[move_id]

        # Only track states when this player is actually learning
        if self.strategy == "agent":
            self.states.append((state, move_id))
        return move

    def _heuristic_move(self, board):
        valid_moves = self._get_valid_moves(board)
        if not valid_moves:
            return 0

        my_mark = self.marker
        opp_mark = " O " if my_mark == " X " else " X "

        # 1) Win if possible
        for move in valid_moves:
            temp = list(board)
            temp[move] = my_mark
            if self._check_winner_on_board(temp) is not None:
                # If setting my_mark there immediately ends with my win, take it
                if self._check_winner_on_board(temp) == (0 if my_mark == " X " else 1):
                    return move

        # 2) Block opponent's immediate win
        for move in valid_moves:
            temp = list(board)
            temp[move] = opp_mark
            if self._check_winner_on_board(temp) is not None:
                if self._check_winner_on_board(temp) == (0 if opp_mark == " X " else 1):
                    return move

        # 3) Otherwise, random
        return random.choice(valid_moves)

    def _minimax(self, board, depth, maximizing, alpha, beta, my_mark):
        winner = self._check_winner_on_board(board)
        if winner is not None or depth == 0:
            if winner is None or winner == 2:
                return 0  # draw or non-terminal at depth limit
            # Map winner (0 for X, 1 for O) to payoff relative to my_mark
            my_is_x = my_mark == " X "
            if winner == (0 if my_is_x else 1):
                return 1000 - (16 - board.count("   "))  # prefer faster wins
            else:
                return -1000 + (16 - board.count("   "))  # prefer slower losses

        valid_moves = self._get_valid_moves(board)
        if maximizing:
            value = -float("inf")
            for move in valid_moves:
                temp = list(board)
                temp[move] = my_mark
                value = max(
                    value,
                    self._minimax(
                        temp, depth - 1, False, alpha, beta, my_mark
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            opp_mark = " O " if my_mark == " X " else " X "
            for move in valid_moves:
                temp = list(board)
                temp[move] = opp_mark
                value = min(
                    value,
                    self._minimax(
                        temp, depth - 1, True, alpha, beta, my_mark
                    ),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _minimax_move(self, board, depth=4):
        valid_moves = self._get_valid_moves(board)
        if not valid_moves:
            return 0
        my_mark = self.marker
        best_value = -float("inf")
        best_move = valid_moves[0]
        alpha = -float("inf")
        beta = float("inf")
        for move in valid_moves:
            temp = list(board)
            temp[move] = my_mark
            value = self._minimax(temp, depth - 1, False, alpha, beta, my_mark)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_move

    def perform_move(self, board):
        valid_moves = self._get_valid_moves(board)

        if self.strategy == "human":
            print("Available moves: ", valid_moves)
            print("Move :")
            move = int(input())
            if move in valid_moves:
                return move
            print("Invalid move. Try again.")
            return self.perform_move(board)

        if self.strategy == "random":
            return random.choice(valid_moves)

        if self.strategy == "agent":
            return self._select_q_move(board, explore=True)

        if self.strategy == "eval_agent":
            return self._select_q_move(board, explore=False)

        if self.strategy == "heuristic":
            return self._heuristic_move(board)

        if self.strategy == "minimax":
            return self._minimax_move(board)

        # Fallback to random if unknown strategy
        return random.choice(valid_moves)

    def q_table_update(self, reward):
        """
        Monte Carlo-style update: use the final reward for all (state, action)
        pairs visited in the episode (no bootstrap on the same state).
        """
        alpha = 0.1
        for (state, act_id) in self.states:
            # Ensure table entry exists (for safety)
            if state not in self.q_table:
                continue
            current_q = self.q_table[state][act_id]
            self.q_table[state][act_id] = current_q + alpha * (reward - current_q)
        self.states.clear()