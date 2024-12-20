from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
import random
from helpers import get_valid_moves, get_directions, check_endgame

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Optimized Othello AI Agent with various improvements.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.initialized = False
        self.max_depth = {
            6: 6,
            8: 5,
            10: 4,
            12: 3
            }
        self.best_move_default = None
        self.corners = None
        self.board_size = None
        self.zobrist_table = None
        self.transposition_table = {}
        self.base_weights = None
        self.time_limit = 1.8  # Time limit per move
        self.start_time = None
        self.move_ordering_cache = {}
        self.move_cache = {}
        self.time_used = 0
        self.weights = None
        self.weight_options = {
            6: {
                # 6x6
                'piece_difference': 100,
                'corner_occupancy': 800,
                'corner_closeness': 900,
                'mobility': 100,
                'frontier_tiles': 100,
                'disk_squares': 15,
                'stability': 900 
            },
            8: {
                # 8x8
                'piece_difference': 100,
                'corner_occupancy': 800,
                'corner_closeness': 900,
                'mobility': 100,
                'frontier_tiles': 100,
                'disk_squares': 15,
                'stability': 900 
            },
            10: {
                # 10x10
                'piece_difference': 100,
                'corner_occupancy': 800,
                'corner_closeness': 1000,
                'mobility': 350,
                'frontier_tiles': 300,
                'disk_squares': 10,
                'stability': 1000
            },
            12: {
                # 12x12
                'piece_difference': 100,
                'corner_occupancy': 800,
                'corner_closeness': 1000,
                'mobility': 350,
                'frontier_tiles': 300,
                'disk_squares': 10.5,
                'stability': 1000
            }
        }
        self.move_stack = []

    def step(self, chess_board, player, opponent):
        """
        Determine the best move using iterative deepening and alpha-beta pruning.
        """
        self.start_time = time.time()

        if not self.initialized:
            self.initialize(chess_board)
            
        self.best_move_default = None    
        best_move = None
        # Iterative deepening Alpha-Beta Search
        for depth in range(1, self.max_depth[self.board_size] + 1):
            remaining_time = self.time_limit - (time.time() - self.start_time)
            if remaining_time <= 0:
                break
            move, _ = self.alpha_beta_search(chess_board, depth, player, opponent,
                                                float('-inf'), float('inf'), True)
            if move is not None:
                best_move = move

        time_delta = time.time() - self.start_time
        self.time_used += time_delta
        print(f"My AI's turn took {time_delta:.4f} seconds.")
        return best_move if best_move else self.best_move_default
    
    
    def initialize(self, chess_board):
        ''' Initalize agent/board data to be used throughout game '''
        if self.board_size is None: 
            self.board_size = chess_board.shape[0]
            self.weights = self.weight_options[self.board_size]

        if self.zobrist_table is None:
            self.zobrist_table = self.initialize_zobrist(self.board_size)
            self.transposition_table = {}

        if self.base_weights is None:
            self.base_weights = generate_base_weights(self.board_size)
            self.corners = [(0, 0), (0, self.board_size - 1),
                            (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        self.initialized = True

    def alpha_beta_search(self, board, depth, player, opponent, alpha, beta, maximizing):
        """
        Alpha-beta pruning with iterative deepening and transposition tables.
        """
        if time.time() - self.start_time > self.time_limit:
            return None, self.evaluate_board(board, player, opponent)

        if depth == 0:
            return None, self.evaluate_board(board, player, opponent)

        hash_key = self.compute_zobrist_hash(board)
        alpha_orig = alpha

        if hash_key in self.transposition_table: 
            entry = self.transposition_table[hash_key]
            if entry['depth'] >= depth:
                if entry['flag'] == 'exact':
                    return entry['move'], entry['value']
                elif entry['flag'] == 'lower':
                    alpha = max(alpha, entry['value'])
                elif entry['flag'] == 'upper':
                    beta = min(beta, entry['value'])
                if alpha >= beta:
                    return entry['move'], entry['value']

        legal_moves = self.get_cached_valid_moves(board, player if maximizing else opponent)
        if not self.best_move_default: self.best_move_default = legal_moves[0]

        if not legal_moves:
            # If there are no legal moves, pass the turn to the opponent
            if maximizing:
                _, value = self.alpha_beta_search(board, depth - 1, player, opponent, alpha, beta, False)
            else:
                _, value = self.alpha_beta_search(board, depth - 1, player, opponent, alpha, beta, True)
            return None, value

        best_move = None
        if maximizing:
            value = float('-inf')
            legal_moves = self.order_moves(board, legal_moves, player)
            for move in legal_moves:
                self.make_move(board, move, player)
                _, score = self.alpha_beta_search(board, depth - 1, player, opponent,
                                                  alpha, beta, False)
                self.undo_move(board)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float('inf')
            legal_moves = self.order_moves(board, legal_moves, opponent)
            for move in legal_moves:
                self.make_move(board, move, opponent)
                _, score = self.alpha_beta_search(board, depth - 1, player, opponent,
                                                  alpha, beta, True)
                self.undo_move(board)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break

        flag = ''
        if value <= alpha_orig:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        
        self.transposition_table[hash_key] = {'depth': depth, 'value': value,
                                              'flag': flag, 'move': best_move}
        return best_move, value


    def make_move(self, board, move, player):
        """
        Make a move on the board and record the flipped discs.
        """
        flipped = self.simulate_move(board, move, player)
        self.move_stack.append((move, flipped))

    def undo_move(self, board):
        """
        Undo the last move made on the board.
        """
        move, flipped = self.move_stack.pop()
        board[move[0], move[1]] = 0
        for x, y in flipped:
            board[x, y] = 3 - board[x, y]

    def simulate_move(self, board, move, player):
        """
        Simulate making a move and return the list of flipped discs.
        """
        flipped_discs = []
        r, c = move
        board[r, c] = player

        for direction in get_directions():
            flipped = self.check_direction(board, move, player, direction)
            if flipped:
                flipped_discs.extend(flipped)

        for x, y in flipped_discs:
            board[x, y] = player

        return flipped_discs

    def check_direction(self, board, move, player, direction):
        """
        Check a direction for discs to flip.
        """
        flipped = []
        r, c = move
        dx, dy = direction
        r += dx
        c += dy
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == 0:
                return []
            elif board[r, c] == player:
                return flipped
            else:
                flipped.append((r, c))
            r += dx
            c += dy
        return []
    
    
    def evaluate_board(self, board, player, opponent):
        """
        Optimized evaluation function with dynamic weights.
        inspired heavily by: https://github.com/kartikkukreja/blog-codes/blob/master/src/Heuristic%20Function%20for%20Reversi%20(Othello).cpp
        """
        # -> first we check if this is a losing state
        # -> want to avoid this at all costs
        # -> inspired by "Don't Lose Evaluation Heuristic"
        #    mentioned in the project spec
        scores = {} 
        is_end, scores[1], scores[2] = check_endgame(board, player, opponent)
        if is_end and scores[opponent] > scores[player]: return float('-inf')

        size = self.board_size
        my_tiles = opp_tiles = 0
        my_front_tiles = opp_front_tiles = 0
        # p: piece difference 
        # c: corner difference
        # l: corner closeness (want to avoid C-squares)
        # m: mobility 
        # f: frontier discs (want to avoid adjacency to empty squares)
        # d: sums of board-weights taken up to far (see base_weights)
        # s: discs that cannot be flipped
        p, c, l, m, f, d = 0, 0, 0, 0, 0, 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for i in range(size):
            for j in range(size):
                if board[i, j] == player:
                    d += self.base_weights[i][j]
                    my_tiles += 1
                elif board[i, j] == opponent:
                    d -= self.base_weights[i][j]
                    opp_tiles += 1

                if board[i, j] != 0:
                    for dx, dy in directions:
                        x, y = i + dx, j + dy
                        if 0 <= x < size and 0 <= y < size and board[x, y] == 0:
                            if board[i, j] == player:
                                my_front_tiles += 1
                            else:
                                opp_front_tiles += 1
                            break

        if my_tiles > opp_tiles:
            p = 1.0 * my_tiles / (my_tiles + opp_tiles)
        elif my_tiles < opp_tiles:
            p = -1.0 * opp_tiles / (my_tiles + opp_tiles)
        else:
            p = 0

        if my_front_tiles > opp_front_tiles:
            f = -1.0 * my_front_tiles / (my_front_tiles + opp_front_tiles)
        elif my_front_tiles < opp_front_tiles:
            f = 1.0 * opp_front_tiles / (my_front_tiles + opp_front_tiles)
        else:
            f = 0

        my_corners = opp_corners = 0
        corners = self.corners
        for x, y in corners:
            if board[x, y] == player:
                my_corners += 1
            elif board[x, y] == opponent:
                opp_corners += 1
        c = 1 * (my_corners - opp_corners)

        my_close = opp_close = 0

        for x, y in corners:
            if board[x, y] == 0:
                for dx, dy in directions:
                    adj_x, adj_y = x + dx, y + dy
                    if 0 <= adj_x < size and 0 <= adj_y < size:
                        if board[adj_x, adj_y] == player:
                            my_close += 1
                        elif board[adj_x, adj_y] == opponent:
                            opp_close += 1
        l = -1 * (my_close - opp_close)

        my_moves = len(self.get_cached_valid_moves(board, player))
        opp_moves = len(self.get_cached_valid_moves(board, opponent))
        if my_moves > opp_moves:
            m = 1.0 * my_moves / (my_moves + opp_moves)
        elif my_moves < opp_moves:
            m = -1.0 * opp_moves / (my_moves + opp_moves)
        else:
            m = 0

        # Compute stability
        my_stable_discs = self.compute_stability(board, player)
        opp_stable_discs = self.compute_stability(board, opponent)
        if my_stable_discs + opp_stable_discs != 0:
            s = 1 * (my_stable_discs - opp_stable_discs) / (my_stable_discs + opp_stable_discs)
        else:
            s = 0
            
        score = (
                self.weights['piece_difference'] * p +
                 self.weights['corner_occupancy'] * c +
                 self.weights['corner_closeness'] * l +
                 self.weights['mobility'] * m +
                 self.weights['frontier_tiles'] * f +
                 self.weights['disk_squares'] * d +
                 self.weights['stability'] * s
                 )
        return score

    def order_moves(self, board, moves, player):
        """
        Order moves based on the evaluation score after making the move.
        """
        # Compute a cache key that includes the board hash and player
        hash_key = (self.compute_zobrist_hash(board), player)
        # Check if the move ordering is already cached
        if hash_key in self.move_ordering_cache:
            return self.move_ordering_cache[hash_key]
        
        opponent = 3 - player  # Assuming players are labeled as 1 and 2
        move_scores = []

        for move in moves:
            # make move and compute its score
            self.make_move(board, move, player)
            score = self.compute_stability(board, player)
            # score = self.evaluate_board(board, player, opponent)
            self.undo_move(board)

            # Append the move and its score to the list
            move_scores.append((move, score))
        
        # Sort the moves based on their evaluation scores in descending order
        sorted_moves = sorted(move_scores, key=lambda x: x[1], reverse=True)

        # Extract the moves from the sorted list
        ordered_moves = [move for move, score in sorted_moves]
        # Cache the ordered moves
        self.move_ordering_cache[hash_key] = ordered_moves
        # Return only the sorted moves, excluding their scores
        return ordered_moves

    def simulate_move_count(self, board, move, player):
        """
        Simulate a move and return the number of discs that would be flipped.
        Does not modify the original board.
        """
        temp_board = board.copy()
        flipped_discs = []
        r, c = move
        temp_board[r, c] = player  # Temporarily place the piece

        for direction in get_directions():
            flipped = self.check_direction_static(temp_board, (r, c), player, direction)
            if flipped:
                flipped_discs.extend(flipped)

        return len(flipped_discs)

    def check_direction_static(self, board, move, player, direction):
        """
        Check a direction for discs to flip without modifying the board.
        """
        flipped = []
        r, c = move
        dx, dy = direction
        r += dx
        c += dy
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == 0:
                return []
            elif board[r, c] == player:
                return flipped
            else:
                flipped.append((r, c))
            r += dx
            c += dy
        return []

    def get_cached_valid_moves(self, board, player):
        """
        Get valid moves using cache to avoid redundant computations.
        """
        hash_key = (self.compute_zobrist_hash(board), player)
        if hash_key in self.move_cache:
            return self.move_cache[hash_key]
        moves = get_valid_moves(board, player)
        self.move_cache[hash_key] = moves
        return moves

    def get_capture_moves(self, board, player):
        """
        Get moves that capture opponent's discs.
        """
        moves = self.get_cached_valid_moves(board, player)
        return [move for move in moves if self.simulate_move_count(board, move, player) > 0]

    def initialize_zobrist(self, size):
        """
        Initialize Zobrist hashing table.
        """
        table = {}
        for i in range(size):
            for j in range(size):
                for k in [0, 1, 2]:
                    table[(i, j, k)] = random.getrandbits(64)
        return table

    def compute_zobrist_hash(self, board):
        """
        Compute the Zobrist hash for the board.
        """
        h = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = board[i, j]
                h ^= self.zobrist_table[(i, j, piece)]
        return h
    
    def compute_stability(self, board, player):
        """
        Compute the number of stable discs for the given player.
        """
        size = self.board_size
        stable = np.zeros((size, size), dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Initialize stability from corners
        corners = self.corners
        for corner in corners:
            x, y = corner
            if board[x, y] == player:
                stable[x, y] = True

        # Continue expanding stable discs
        changed = True
        while changed:
            changed = False
            for i in range(size):
                for j in range(size):
                    if stable[i, j]:
                        continue
                    if board[i, j] != player:
                        continue
                    is_stable = True
                    for dx, dy in directions:
                        x, y = i, j
                        while True:
                            x += dx
                            y += dy
                            if x < 0 or x >= size or y < 0 or y >= size:
                                break
                            if board[x, y] == 0:
                                is_stable = False
                                break
                            if board[x, y] != player and not stable[x, y]:
                                is_stable = False
                                break
                        if not is_stable:
                            break
                    if is_stable:
                        stable[i, j] = True
                        changed = True
        return np.sum(stable)


def generate_base_weights(n):
    """
    Generates an n x n board with integer weights for each entry based on predefined layer patterns.

    Parameters:
    n (int): Size of the board (must be one of 6, 8, 10, 12).

    Returns:
    list of lists: The generated board with assigned weights.
    """
    # Predefined layer weights for specific grid sizes
    return  {
        6: [
            [100, -10, 10, 10, -10, 100],
            [-10, -20,  3,  3, -20, -10],
            [ 10,   3,  3,  3,   3,  10],
            [ 10,   3,  3,  3,   3,  10],
            [-10, -20,  3,  3, -20, -10],
            [100, -10, 10, 10, -10, 100]
        ],
        8: [
            [100, -10, 11, 6, 6, 11, -10, 100],
            [-10, -20, 1, 2, 2, 1, -20, -10],
            [10, 1, 5, 4, 4, 5, 1, 10],
            [6, 2, 4, 2, 2, 4, 2, 6],
            [6, 2, 4, 2, 2, 4, 2, 6],
            [10, 1, 5, 4, 4, 5, 1, 10],
            [-10, -20, 1, 2, 2, 1, -20, -10],
            [100, -10, 11, 6, 6, 11, -10, 100]
        ],
        10: [
            [100, -10,  11,   6,   6,   6,   6,  11, -10, 100],
            [-10, -20,   2,   1,   1,   1,   1,   2, -20, -10],
            [ 11,   2,   5,   4,   4,   4,   4,   5,   2,  11],
            [  6,   1,   4,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   4,   1,   6],
            [ 11,   2,   5,   4,   4,   4,   4,   5,   2,  11],
            [-10, -20,   2,   1,   1,   1,   1,   2, -20, -10],
            [100, -10,  11,   6,   6,   6,   6,  11, -10, 100]
        ],
        12: [
            [100, -10,  11,   6,   6,   6,   6,   6,   6,  11, -10, 100],
            [-10, -20,   2,   1,   1,   1,   1,   1,   1,   2, -20, -10],
            [ 11,   2,   5,   4,   4,   4,   4,   4,   4,   5,   2,  11],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [  6,   1,   4,   3,   3,   3,   3,   3,   3,   4,   1,   6],
            [ 11,   2,   5,   4,   4,   4,   4,   4,   4,   5,   2,  11],
            [-10, -20,   2,   1,   1,   1,   1,   1,   1,   2, -20, -10],
            [100, -10,  11,   6,   6,   6,   6,   6,   6,  11, -10, 100]
        ]
    }[n]