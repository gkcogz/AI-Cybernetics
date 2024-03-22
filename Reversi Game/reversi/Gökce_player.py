import time


# I have chosen Minimax with Alpha-Beta-Pruning while search depth = 3.

class MyPlayer:
    '''We create MyPlayer class and write it's attributes and define
    it's methods. Then, we implement Minimax with alpha-beta pruning.'''


    # Here, we initilialize a new instance of the class with specific attributes.
    def __init__(self, my_color, opponent_color, board_size=8):
        self.name = 'gokce_ogu'
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size


    # This method evaluates and returns a score for the given board state.
    # I here calculate what my score will be, based on board.
    def evaluate_board(self, board):
        my_pieces = 0
        opponent_pieces = 0
        for row in board:
            for cell in row:
                if cell == self.my_color:
                    my_pieces += 1
                elif cell == self.opponent_color:
                    opponent_pieces += 1
        return my_pieces - opponent_pieces

    
    # This method simulates making a move on the board for a given color
    # and returns the resulting board state.
    def simulate_move(self, board, move, color):

        # A copy of the board.
        simulated_board = [row[:] for row in board]

        # Places the piece of the given color at the specified move location
        # on the simulated board.
        simulated_board[move[0]][move[1]] = color

        # All eight possible directions.
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            x, y = move[0] + dx, move[1] + dy

            # A list to keep track of opponent pieces that could be flipped.
            pieces_to_flip = []
            while 0 <= x < self.board_size and 0 <= y < self.board_size and simulated_board[x][y] == self.opponent_color:
                pieces_to_flip.append((x, y))
                x += dx
                y += dy

            # If a closing piece is found, opponent_color is changed to color. 
            # Otherwise, the chain is closed.
            if 0 <= x < self.board_size and 0 <= y < self.board_size and simulated_board[x][y] == color:
                for px, py in pieces_to_flip:
                    simulated_board[px][py] = color
        return simulated_board

    # We end the game, if there are no valid moves left.
    def is_game_over(self, board):
        return not self.get_all_valid_moves(board, self.my_color) and not self.get_all_valid_moves(board, self.opponent_color)

    # Minimax for maximizing player.
    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board)
        
        # We check if the current recursion level is trying to maximize the player's score.
        if maximizingPlayer:

            # Alpha is set to -inf.
            maxEval = float('-inf')

            # We iterate through all valid moves for the opponent.
            for move in self.get_all_valid_moves(board, self.my_color):
                simulated_board = self.simulate_move(board, move, self.my_color)

                # Depth is decreased, the move is simulated and changing to minimizing player (opponent).
                eval = self.minimax(simulated_board, depth - 1, alpha, beta, False)
                
                # Max. score updated.
                maxEval = max(maxEval, eval)

                # Alpha is updated.
                alpha = max(alpha, eval)

                # We check if the current branch can be pruned.
                if beta <= alpha:
                    break
            return maxEval
        
        # Minimax for minimizing player.
        # Syntax is analogous to the previous one.
        else:
            minEval = float('inf')
            for move in self.get_all_valid_moves(board, self.opponent_color):
                simulated_board = self.simulate_move(board, move, self.opponent_color)
                eval = self.minimax(simulated_board, depth - 1, alpha, beta, True)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval

    # We define the move method, which selects the best move for the player
    # based on the Minimax algorithm.
    def move(self, board):

        # We start time recording and initialize the parameters.
        start_time = time.time()
        best_score = float('-inf')  # Correctly define best_score before using it
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        valid_moves = self.get_all_valid_moves(board, self.my_color)

        if not valid_moves:
            return None

        # Iterating through each valid move.
        for move in valid_moves:
            simulated_board = self.simulate_move(board, move, self.my_color)

            # The player is set as the maximizing player.
            # The depth is set to 3.
            score = self.minimax(simulated_board, 3, alpha, beta, True)  
            
            # We update 'best_score' and 'best_move' if a better score is found.
            if score > best_score:
                best_score = score
                best_move = move

            if time.time() - start_time > 4.5:  # Stop if approaching time limit
                break

        return best_move


    def get_all_valid_moves(self, board, color):
        valid_moves = []

        # We iterate over each cell on the board.
        for x in range(self.board_size):
            for y in range(self.board_size):

                # We check if an empty cell is a valid move (x, y) for the given color
                # and add it to the list if so.
                if board[x][y] == -1:  # Empty cell
                    if self.__is_correct_move((x, y), board, color):
                        valid_moves.append((x, y))
        return valid_moves


    def __is_correct_move(self, move, board, color):

        # We check if placing a piece at move(x,y) is valid for the chosen color.
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            if self.__confirm_direction(move, dx, dy, board, color):
                return True
        return False

    # We check if moving in a specific direction from the given move will 
    # capture any of the opponent's pieces.
    def __confirm_direction(self, move, dx, dy, board, color):
        posx, posy = move[0]+dx, move[1]+dy

        # A valid direction must start with an opponent's piece adjacent to the move position.
        if (posx < 0 or posx >= self.board_size or posy < 0 or posy >= self.board_size) or board[posx][posy] != self.opponent_color:
            return False
        posx += dx
        posy += dy
        while 0 <= posx < self.board_size and 0 <= posy < self.board_size:
            
            # If an empty space is found, return false.
            if board[posx][posy] == -1:
                return False
            
            # If player's piece is found, return true.
            if board[posx][posy] == color:
                return True
            posx += dx
            posy += dy

        # If the end of the board is reached without finding a piece
        # of the player's color, the direction is invalid for capturing.
        return False
