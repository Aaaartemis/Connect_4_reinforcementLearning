import sys
import pickle

import pygame

from game import Game
from player import Player


WINDOW_SIZE = 500
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
LINE_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
X_COLOR = (200, 0, 0)
O_COLOR = (0, 0, 200)


def draw_board(screen, board, font):
    screen.fill(BG_COLOR)

    # Grid lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (i * CELL_SIZE, 0),
            (i * CELL_SIZE, WINDOW_SIZE),
            2,
        )
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (0, i * CELL_SIZE),
            (WINDOW_SIZE, i * CELL_SIZE),
            2,
        )

    # Pieces
    for idx, cell in enumerate(board):
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        x = col * CELL_SIZE + CELL_SIZE // 2
        y = row * CELL_SIZE + CELL_SIZE // 2

        if cell == " X ":
            text = font.render("X", True, X_COLOR)
            rect = text.get_rect(center=(x, y))
            screen.blit(text, rect)
        elif cell == " O ":
            text = font.render("O", True, O_COLOR)
            rect = text.get_rect(center=(x, y))
            screen.blit(text, rect)

    pygame.display.flip()


def main():
    try:
        with open("q_table.pkl", "rb") as f:
            loaded_q_table = pickle.load(f)
    except OSError:
        print("Could not open q_table.pkl. Train the agent first by running training.py.")
        sys.exit(1)

    # Agent is player 0 (X), human is player 1 (O)
    agent = Player("Agent", "agent")
    agent.q_table = loaded_q_table
    agent.epsilon = 0.0
    human = Player("Human", "human")
    game = Game([agent, human])

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("4x4 Tic-Tac-Toe - Play vs Agent")
    font = pygame.font.SysFont(None, CELL_SIZE // 2)

    running = True
    human_turn = False  # Agent moves first (X), consistent with training

    while running:
        draw_board(screen, game.board, font)

        if not game.game_over:
            if not human_turn:
                # Agent move
                move = agent.perform_move(game.board)
                game.board[move] = " X "
                game.check_winner()
                human_turn = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not game.game_over and human_turn:
                    mx, my = event.pos
                    col = mx // CELL_SIZE
                    row = my // CELL_SIZE
                    idx = row * GRID_SIZE + col
                    if 0 <= idx < 16 and game.board[idx] == "   ":
                        game.board[idx] = " O "
                        game.check_winner()
                        human_turn = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset_board()
                    human_turn = False

        if game.game_over:
            # Simple end-screen overlay
            result_text = "Draw!"
            if game.winner == 0:
                result_text = "Agent (X) wins!"
            elif game.winner == 1:
                result_text = "You (O) win!"

            overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))

            text = font.render(result_text + " (Press R to restart, Esc to quit)", True, (255, 255, 255))
            rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            screen.blit(text, rect)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset_board()
                        human_turn = False

    pygame.quit()


if __name__ == "__main__":
    main()