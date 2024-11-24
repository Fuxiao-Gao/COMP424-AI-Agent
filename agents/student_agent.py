# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

# global variable to count the current depth
depth = 0
MAX_PLAYER = 1
MIN_PLAYER = 2
@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Goal: choose your best move for the current player using alpha-beta pruning
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).
    return a tuple (r,c) as the position of your next move
    """
    start_time = time.time()
    MAX_PLAYER = player
    MIN_PLAYER = opponent

    alpha = float('-inf')
    beta = float('inf')
    depth = np.sum(chess_board != 0)
    max_depth = self.get_dynamic_max_depth(chess_board.shape[0], depth)
    # print(f"depth: {depth}, max_depth: {max_depth}")

    best_move, _ = self.minimax_alpha_beta(max_depth, chess_board, alpha, beta, player)

    time_taken = time.time() - start_time
    # print("My AI's turn took ", time_taken, "seconds.")

    return best_move

  def minimax_alpha_beta(self, max_depth, chess_board, alpha, beta, player):
    '''
    Perform minimax search with alpha-beta pruning to find the best move
    :param max_depth: The maximum depth of the search tree.
    :param chess_board: 2D numpy array representing the game board.
    :param alpha: The best score that the maximizing player can guarantee.
    :param beta: The best score that the minimizing player can guarantee.
    :param player: The current player (MAX_PLAYER or MIN_PLAYER).
    :return tuple: The best move and the best score.
    '''
    if max_depth == 0 or check_endgame(chess_board, player, MIN_PLAYER)[0]:
      # calculate the depth of the search tree
      depth = np.sum(chess_board != 0)
      return None, self.evaluate_board(chess_board, depth)
    
    moves = get_valid_moves(chess_board, player)
    best_move = None
    
    if player == MAX_PLAYER:
      best_score = float('-inf')
      for move in moves:
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        _, score = self.minimax_alpha_beta(max_depth - 1, new_board, alpha, beta, MIN_PLAYER)
        if score > best_score:
          best_score = score
          best_move = move
        alpha = max(alpha, best_score)
        if beta <= alpha:
          break
    else:
      best_score = float('inf')
      for move in moves:
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)
        _, score = self.minimax_alpha_beta(max_depth - 1, new_board, alpha, beta, MAX_PLAYER)
        if score < best_score:
          best_score = score
          best_move = move
        beta = min(beta, best_score)
        if beta <= alpha:
          break
        
    return best_move, best_score
    
  def evaluate_board(self, chess_board, depth):
    '''
    Evaluate the board state based on multiple factors.
    Positive score for the MAX_PLAYER and negative score for the MIN_PLAYER
    :param chess_board: 2D numpy array representing the game board.
    :param depth: The current depth of the search tree.
    :return int: The evaluated score of the board.
    '''
    # Get dynamic weights based on depth and board size
    weights = self.get_dynamic_weights(depth, chess_board)
    w_coin, w_mobility, w_corner, w_stability = weights
    
    coin_parity_score = self.coin_parity(chess_board)
    mobility_score = self.mobility(chess_board)
    corner_potential_score = self.corner_potential(chess_board)
    stability_score = self.stability(chess_board)
    
    # Calculate the heuristic score
    heuristic_score = w_coin * coin_parity_score + w_mobility * mobility_score + w_corner * corner_potential_score + w_stability * stability_score
    
    return heuristic_score

  def coin_parity(self, chess_board):
    '''
    Calculate the coin parity of the player.
    Goal: maximize the difference between your discs and the opponent's discs
    '''
    # compute the number of discs for the player and the opponent
    player_discs = np.sum(chess_board == MAX_PLAYER)
    opponent_discs = np.sum(chess_board == MIN_PLAYER)
    
    # calculate the coin parity score
    total_possible_discs = chess_board.size
    coin_parity_score = (player_discs - opponent_discs) / max(1, player_discs + opponent_discs) * (player_discs + opponent_discs) / total_possible_discs
    return coin_parity_score
  
  def mobility(self, chess_board):
    '''
    Calculate the mobility of the player.
    Goal: maximize the number of legal moves avaliable to you while minimizing the opponent's moves
    '''
    # compute the number of legal moves for the player and the opponent
    player_legal_moves = get_valid_moves(chess_board, MAX_PLAYER)
    opponent_legal_moves = get_valid_moves(chess_board, MIN_PLAYER)
    
    # calculate the mobility score
    mobility_score = (len(player_legal_moves) - len(opponent_legal_moves)) / max(1, len(player_legal_moves) + len(opponent_legal_moves))
    return mobility_score
  
  def corner_potential(self, chess_board):
    '''
    Calculate the corner potential of the player.
    Goal: C-Squares -- penalize occupying these squares early unless adjacent corners are already stable
          X-Squares -- reward occupying these squares early unless adjacent corners are already stable
    '''
    # find the C-squares, X-squares and corners on the board
    c_squares = self.find_c_squares(chess_board)
    x_squares = self.find_x_squares(chess_board)
    
    # calculate the raw corner score
    raw_corner_potential_score = 0

    for c_square in c_squares:
      if chess_board[c_square] == MAX_PLAYER:
        raw_corner_potential_score -= 12
      elif chess_board[c_square] == MIN_PLAYER:
        raw_corner_potential_score += 12

    for x_square in x_squares:
      if chess_board[x_square] == MAX_PLAYER:
        raw_corner_potential_score -= 20
      elif chess_board[x_square] == MIN_PLAYER:
        raw_corner_potential_score += 20
    
    # normalize the raw corner pontential score
    min_corner_potential_score = len(c_squares) * -12 + len(x_squares) * -20
    max_corner_potential_score = len(c_squares) * 12 + len(x_squares) * 20
    
    corner_potential_score = 2 * (raw_corner_potential_score - min_corner_potential_score) / max(1, max_corner_potential_score - min_corner_potential_score) - 1
    return corner_potential_score
  
  def stability(self, chess_board):
    '''
    Calculate the stability of the player.
    Goal: reward discs that cannot be flipped by the opponent
    '''
    stability_score = 0
    stable_discs = set()
    
    # Directions for checking stability (horizontal, vertical, and diagonal)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # Check each corner and its connected discs
    corners = self.find_corners(chess_board)
    for corner in corners:
        if chess_board[corner] == MAX_PLAYER:
            stability_score += 1  # Corner disc is always stable
            stable_discs.add(corner)
            # Recursively check connected discs
            self.check_stability(chess_board, corner, stable_discs, directions)
    
    # Normalize the stability score
    total_discs = chess_board.size
    stability_score = 2 * len(stable_discs) / total_discs - 1
    
    return stability_score

  def check_stability(self, chess_board, position, stable_discs, directions):
    '''
    Recursively check the stability of discs connected to a stable disc.
    '''
    row, col = position
    for direction in directions:
        d_row, d_col = direction
        new_row, new_col = row + d_row, col + d_col
        while 0 <= new_row < chess_board.shape[0] and 0 <= new_col < chess_board.shape[1]:
            if chess_board[new_row, new_col] == MAX_PLAYER and (new_row, new_col) not in stable_discs:
                stable_discs.add((new_row, new_col))
                self.check_stability(chess_board, (new_row, new_col), stable_discs, directions)
            else:
                break
            new_row += d_row
            new_col += d_col
    
  def find_x_squares(self, chess_board):
    '''
    Find the X-squares on the board <-- diagnoally adjacent to the corners
    4 X-squares on the board for N x N board
    '''
    
    board_size = chess_board.shape[0]
    x_squares = [
      (1, 1), (1, board_size - 2),
      (board_size - 2, 1), (board_size - 2, board_size - 2)
    ]
    return x_squares

  def find_c_squares(self, chess_board):
    '''
    Find the C-squares on the board <-- directly adjacent to the corners
    for N >= 4, total c-squares = 8 for any N x N board
    '''
    
    board_size = chess_board.shape[0]
    c_squares = [
        (0, 1), (0, board_size - 2),
        (1, 0), (1, board_size - 1),
        (board_size - 2, 0), (board_size - 2, board_size - 1),
        (board_size - 1, 1), (board_size - 1, board_size - 2)
    ]
    return c_squares
  
  def find_corners(self, chess_board):
    '''
    Find the corners on the board
    '''
    board_size = chess_board.shape[0]
    corners = [
        (0, 0), (0, board_size - 1),
        (board_size - 1, 0), (board_size - 1, board_size - 1)
    ]
    return corners
  
  def find_edges(self, chess_board):
    '''
    Find the edges on the board
    '''
    board_size = chess_board.shape[0]
    edges = []
    for i in range(1, board_size - 1):
      edges.append((0, i))
      edges.append((board_size - 1, i))
      edges.append((i, 0))
      edges.append((i, board_size - 1))
    return

  def get_dynamic_weights(self, depth, chess_board):
    '''
    Get the dynamic weights for the evaluation function based on the depth and board size.
    Early game: mobility and coin parity are more important
    Mid game: stability and corner potential are more important
    Late game: stability and corner potential are most important
    w_coin, w_mobility, w_corner, w_stability
    '''
    board_size = chess_board.shape[0]
    total_possible_discs = board_size ** 2
    
    # Scale depth thresholds based on the board size
    early_game_depth = total_possible_discs * 0.25 
    mid_game_depth = total_possible_discs * 0.75  
    late_game_depth = total_possible_discs
    
    # Define the weights based on depth and board size
    if depth < early_game_depth:
      w_coin = 0.4
      w_mobility = 0.35 - 0.1 * (depth / early_game_depth) 
      w_corner = 0.15
      w_stability = 0.1  
    elif depth < mid_game_depth: 
      transition = (depth - early_game_depth) / (mid_game_depth - early_game_depth)  # Transition factor [0, 1]
      w_coin = 0.3 - 0.1 * transition
      w_mobility = 0.3 - 0.1 * transition
      w_corner = 0.2 + 0.1 * transition
      w_stability = 0.2 + 0.1 * transition 
    else:  
      w_coin = 0.1
      w_mobility = 0.2
      w_corner = 0.3
      w_stability = 0.4
      
    return [w_coin, w_mobility, w_corner, w_stability]

  def get_dynamic_max_depth(self, board_size, depth):
    '''
    Get the dynamic max depth based on the board size.
    the max depth decreases as the game progresses
    '''
    if depth < board_size * 2: # Early game
      return min(3, max(4, board_size//2))
    elif depth < board_size * 6: # Mid game
      return min(3, max(3, board_size//3))
    else: # Late game
      return min(3, max(2, board_size//4))
    
    