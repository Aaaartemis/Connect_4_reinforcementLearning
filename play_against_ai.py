import sys
import pickle

import pygame

from game import Game
from player import Player

q_table_filename = "q_table_final.pkl"

WINDOW_SIZE = 600
TITLE_BAR_HEIGHT = 70
MARGIN = 25
BOARD_SIZE = WINDOW_SIZE - TITLE_BAR_HEIGHT - (2 * MARGIN)
GRID_SIZE = 4
CELL_SIZE = BOARD_SIZE // GRID_SIZE

# Theme
BG_TOP = (15, 23, 42)  # slate-900
BG_BOTTOM = (2, 6, 23)  # navy-950
TITLE_BG = (30, 41, 59)  # slate-800
TITLE_FG = (226, 232, 240)  # gray-200
BOARD_BG = (15, 23, 42)  # slate-900
BOARD_BORDER = (148, 163, 184)  # slate-400
GRID_LINE = (71, 85, 105)  # slate-600

X_COLOR = (220, 38, 38)  # red-600
O_COLOR = (59, 130, 246)  # blue-600
OVERLAY_ALPHA = 120

BOARD_RADIUS = 18
TITLE_RADIUS = 16


def make_vertical_gradient(size: int) -> pygame.Surface:
    """Create a one-time background gradient surface."""
    surf = pygame.Surface((size, size))
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (size, y))
    return surf


def draw_board(screen: pygame.Surface, board, board_rect: pygame.Rect) -> None:
    """Draws the board and pieces."""
    # Board background + border
    pygame.draw.rect(screen, BOARD_BG, board_rect, border_radius=BOARD_RADIUS)
    pygame.draw.rect(screen, BOARD_BORDER, board_rect, width=3, border_radius=BOARD_RADIUS)

    # Grid lines
    for i in range(1, GRID_SIZE):
        x = board_rect.x + i * CELL_SIZE
        y = board_rect.y + i * CELL_SIZE
        pygame.draw.line(screen, GRID_LINE, (x, board_rect.y), (x, board_rect.y + board_rect.height), 3)
        pygame.draw.line(screen, GRID_LINE, (board_rect.x, y), (board_rect.x + board_rect.width, y), 3)

    # Pieces (draw shapes instead of relying on large fonts)
    center_pad = max(6, int(CELL_SIZE * 0.07))
    # Keep pieces comfortably inside each cell (smaller than before).
    piece_extent = max(16, int(CELL_SIZE * 0.28))
    x_width = max(4, int(CELL_SIZE * 0.07))
    o_width = max(4, int(CELL_SIZE * 0.085))

    for idx, cell in enumerate(board):
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        cx = board_rect.x + (col * CELL_SIZE) + CELL_SIZE // 2
        cy = board_rect.y + (row * CELL_SIZE) + CELL_SIZE // 2

        marker = cell.strip()
        if marker == "X":
            p = piece_extent
            pygame.draw.line(screen, X_COLOR, (cx - p, cy - p), (cx + p, cy + p), x_width)
            pygame.draw.line(screen, X_COLOR, (cx + p, cy - p), (cx - p, cy + p), x_width)
        elif marker == "O":
            pygame.draw.circle(screen, O_COLOR, (cx, cy), piece_extent, o_width)


def main():
    try:
        with open(q_table_filename, "rb") as f:
            loaded_q_table = pickle.load(f)
    except OSError:
        print(f"Could not open {q_table_filename}. Train the agent first by running training.py.")
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
    gradient_bg = make_vertical_gradient(WINDOW_SIZE)

    # Fonts sized to fit inside the board overlay
    title_font = pygame.font.SysFont(None, 35)
    turn_font = pygame.font.SysFont(None, 28)
    overlay_title_font = pygame.font.SysFont(None, 44)
    overlay_body_font = pygame.font.SysFont(None, 24)

    running = True
    human_turn = False  # Agent moves first (X), consistent with training

    board_rect = pygame.Rect((WINDOW_SIZE - BOARD_SIZE) // 2, TITLE_BAR_HEIGHT + MARGIN, BOARD_SIZE, BOARD_SIZE)
    header_rect = pygame.Rect(0, 0, WINDOW_SIZE, TITLE_BAR_HEIGHT)

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset_board()
                    human_turn = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not game.game_over and human_turn and board_rect.collidepoint(event.pos):
                    mx, my = event.pos
                    col = (mx - board_rect.x) // CELL_SIZE
                    row = (my - board_rect.y) // CELL_SIZE
                    idx = row * GRID_SIZE + col
                    if 0 <= idx < 16 and game.board[idx].strip() == "":
                        game.board[idx] = " O "
                        game.check_winner()
                        human_turn = False

        # Agent move (X) when it's their turn.
        if not game.game_over and not human_turn:
            move = agent.perform_move(game.board)
            game.board[move] = " X "
            game.check_winner()
            human_turn = True

        # Background
        screen.blit(gradient_bg, (0, 0))

        # Title bar
        pygame.draw.rect(screen, TITLE_BG, header_rect, border_radius=TITLE_RADIUS)
        pygame.draw.line(
            screen,
            BOARD_BORDER,
            (0, TITLE_BAR_HEIGHT),
            (WINDOW_SIZE, TITLE_BAR_HEIGHT),
            2,
        )

        title_surf = title_font.render("4x4 Tic-Tac-Toe vs Agent", True, TITLE_FG)
        screen.blit(title_surf, title_surf.get_rect(center=(WINDOW_SIZE // 2, TITLE_BAR_HEIGHT // 2 - 5)))

        if game.game_over:
            turn_text = "Game Over"
        else:
            turn_text = "Your move (O)" if human_turn else "Agent thinking (X)..."
        turn_surf = turn_font.render(turn_text, True, TITLE_FG)
        # Keep the status text inside the title bar so it never overlaps the grid.
        screen.blit(turn_surf, turn_surf.get_rect(center=(WINDOW_SIZE // 2, TITLE_BAR_HEIGHT - 18)))

        # Board + pieces
        draw_board(screen, game.board, board_rect)

        # Game over overlay (inside the board area)
        if game.game_over:
            if game.winner == 0:
                title_line = "Agent (X) wins!"
            elif game.winner == 1:
                title_line = "You (O) win!"
            else:
                title_line = "Draw!"

            overlay = pygame.Surface((board_rect.width, board_rect.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, OVERLAY_ALPHA))
            screen.blit(overlay, board_rect.topleft)

            x_center = board_rect.x + board_rect.width // 2
            y_center = board_rect.y + board_rect.height // 2

            # Title (with shadow)
            shadow = overlay_title_font.render(title_line, True, (0, 0, 0))
            title_surf = overlay_title_font.render(title_line, True, (255, 255, 255))
            screen.blit(shadow, shadow.get_rect(center=(x_center + 2, y_center + 2)))
            screen.blit(title_surf, title_surf.get_rect(center=(x_center, y_center - 12)))

            body_lines = ["Press 'R' to Restart", "Press 'Esc' to Quit"]
            y = y_center + 22
            for line in body_lines:
                surf = overlay_body_font.render(line, True, (255, 255, 255))
                screen.blit(surf, surf.get_rect(center=(x_center, y)))
                y += overlay_body_font.get_linesize()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()