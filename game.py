# Class for game logic

class Game:
    def __init__(self, players):
        self.players = players
        self.board = ["   " for x in range(16)]
        self.game_over = False
        self.winner = ""
        # Assign markers so players can reason about X/O when needed
        if len(self.players) >= 1 and hasattr(self.players[0], "marker"):
            self.players[0].marker = " X "
        if len(self.players) >= 2 and hasattr(self.players[1], "marker"):
            self.players[1].marker = " O "

    # Func to print the board
    def display_board(self):
        for x in range(4):
            print("|".join(self.board[x*4:(x+1)*4]))
            if x < 3:
                print("-" * 15)
        print("\n")

    # Main game logic 
    def play_game(self):
        while(True):
            move = self.players[0].perform_move(self.board)
            self.board[move] = " X "
            # self.display_board()
            self.check_winner()
            if self.game_over == True:
                break
            move = self.players[1].perform_move(self.board)
            self.board[move] = " O "
            # self.display_board()
            self.check_winner()
            if self.game_over == True:
                break

    # Winning conditions
    def check_winner(self):
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
            (3, 6, 9, 12)
        ]

        for (i, j, k, l) in win_conditions:
            if self.board[i] != "   " and self.board[i] == self.board[j] == self.board[k] == self.board[l]:
                if self.board[i] == " X ":
                    self.winner = 0
                    self.game_over = True
                else:
                    self.winner = 1
                    self.game_over = True

            
        if "   " not in self.board:
            self.game_over = True
            self.winner = 2

    # Reset the board for a new game
    def reset_board(self):
        self.board = ["   " for x in range(16)]
        self.game_over = False
        self.winner = ""