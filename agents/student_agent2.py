# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent2")
class StudentAgent2(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent2, self).__init__()
        self.name = "StudentAgent2"
        self.heuristic_time = 0  # Initialize a global variable to track heuristic calculation time

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        """
        # Reset the heuristic time before starting alpha-beta search
        self.heuristic_time = 0

        max_depth = self.get_depth(chess_board)
        best_move, _ = self.alpha_beta_search(
            chess_board, max_depth, player, opponent, float('-inf'), float('inf'), True
        )

        # Print the total time spent on heuristic calculations for this move
        print(f"Total heuristic calculation time for this move: {self.heuristic_time:.4f} seconds.")
        return best_move

    def get_depth(self, chess_board):
        """
        Determine the search depth based on the board size or game state.
        Adjust this as necessary for larger boards or late game stages.
        """
        board_size = chess_board.shape[0]
        # Depth scales inversely with board size to manage complexity
        if board_size == 6:
            return 6
        elif board_size == 8:
            return 5
        elif board_size == 10:
            return 4
        else:
            return 3

    def alpha_beta_search(self, chess_board, depth, player, opponent, alpha, beta, maximizing):
        """
        Perform alpha-beta pruning minimax search.

        Parameters:
        - chess_board: Current board state (2D numpy array).
        - depth: Fixed depth for all branches.
        - player: Current player (1 for black, 2 for white).
        - opponent: Opponent player (1 for black, 2 for white).
        - alpha: Alpha value for pruning.
        - beta: Beta value for pruning.
        - maximizing: Boolean, True if this is the maximizing player's turn.

        Returns:
        - Tuple (best_move, best_score): The best move and its evaluation score.
        """
        legal_moves = get_valid_moves(chess_board, player if maximizing else opponent)

        # Base case: terminal state or depth reached
        if depth == 0 or not legal_moves:
            is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
            heuristic = self.evaluate_early_game if np.sum(chess_board != 0) / chess_board.size < 0.6 else self.evaluate_late_game
            
            # Track heuristic calculation time
            start_time = time.time()
            score = heuristic(chess_board, player, opponent)
            self.heuristic_time += time.time() - start_time  # Accumulate time spent
            return None, score

        best_move = None

        if maximizing:
            value = float('-inf')
            for move in legal_moves:
                simulated_board = deepcopy(chess_board)
                execute_move(simulated_board, move, player)
                _, move_value = self.alpha_beta_search(simulated_board, depth - 1, player, opponent, alpha, beta, False)
                if move_value > value:
                    value = move_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Prune
            return best_move, value

        else:
            value = float('inf')
            for move in legal_moves:
                simulated_board = deepcopy(chess_board)
                execute_move(simulated_board, move, opponent)
                _, move_value = self.alpha_beta_search(simulated_board, depth - 1, player, opponent, alpha, beta, True)
                if move_value < value:
                    value = move_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Prune
            return best_move, value

    def evaluate_early_game(self, chess_board, player, opponent):
        """
        Early game heuristic incorporating coin parity, mobility, corners captured, and stability.
        """
        coin_parity = self.calculate_coin_parity(chess_board, player, opponent)
        mobility = self.calculate_mobility(chess_board, player, opponent)
        corners = self.calculate_corners(chess_board, player, opponent)
        stability = self.calculate_stability(chess_board, player, opponent)

        # Center control (4x4 region)
        center_start = chess_board.shape[0] // 2 - 2
        center_end = center_start + 4
        center = chess_board[center_start:center_end, center_start:center_end]
        center_control = np.sum(center == player) * 25

        # Weighted sum of heuristics
        return 0.3 * coin_parity + 0.4 * mobility + 0.2 * corners + 0.1 * stability + 0.2 * center_control

    def evaluate_late_game(self, chess_board, player, opponent):
        """
        Late game heuristic focusing on coin parity, corners captured, and stability.
        """
        is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
        if is_endgame:
            return 1e6 if player_score > opponent_score else -1e6 if player_score < opponent_score else 0

        coin_parity = self.calculate_coin_parity(chess_board, player, opponent)
        mobility = self.calculate_mobility(chess_board, player, opponent)
        corners = self.calculate_corners(chess_board, player, opponent)
        stability = self.calculate_stability(chess_board, player, opponent)

        # Flipping pieces as an added heuristic in late game
        flips = sum(count_capture(chess_board, move, player) for move in get_valid_moves(chess_board, player))
        flip_score = flips * 10

        # Weighted sum of heuristics
        return 0.5 * coin_parity + 0.2 * mobility + 0.2 * corners + 0.1 * stability + 0.2 * flip_score

    def calculate_coin_parity(self, chess_board, player, opponent):
        """
        Calculate coin parity heuristic.
        """
        player_score = np.sum(chess_board == player)
        opponent_score = np.sum(chess_board == opponent)
        if player_score + opponent_score != 0:
            return 100 * (player_score - opponent_score) / (player_score + opponent_score)
        return 0

    def calculate_mobility(self, chess_board, player, opponent):
        """
        Calculate mobility heuristic.
        """
        player_moves = len(get_valid_moves(chess_board, player))
        opponent_moves = len(get_valid_moves(chess_board, opponent))
        if player_moves + opponent_moves != 0:
            return 100 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        return 0

    def calculate_corners(self, chess_board, player, opponent):
        """
        Calculate corners captured heuristic.
        """
        corners = [(0, 0), (0, chess_board.shape[1] - 1), 
                   (chess_board.shape[0] - 1, 0), (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
        player_corners = sum(1 for corner in corners if chess_board[corner] == player)
        opponent_corners = sum(1 for corner in corners if chess_board[corner] == opponent)
        if player_corners + opponent_corners != 0:
            return 100 * (player_corners - opponent_corners) / (player_corners + opponent_corners)
        return 0

    def calculate_stability(self, chess_board, player, opponent):
        """
        Calculate stability heuristic based on stable coins (corners).
        """
        corners = [(0, 0), (0, chess_board.shape[1] - 1), 
                   (chess_board.shape[0] - 1, 0), (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
        player_stability = sum(1 for corner in corners if chess_board[corner] == player)
        opponent_stability = sum(1 for corner in corners if chess_board[corner] == opponent)
        if player_stability + opponent_stability != 0:
            return 100 * (player_stability - opponent_stability) / (player_stability + opponent_stability)
        return 0