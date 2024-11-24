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
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).
    return a tuple (r,c) as the position of your next move
    """
    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    
    alpha = float('-inf')
    beta = float('inf')
    depth = np.sum(chess_board != 0) - 4 # depth is the number of turns made so far
    max_depth = self.get_dynamic_max_depth(chess_board.shape[0], depth)
    print(f"depth: {depth}, max_depth: {max_depth}")
    best_move = None
    best_score = float('-inf')
    
    # Get all valid moves for the player
    valid_moves = get_valid_moves(chess_board, player)
    
    if not valid_moves:
        return None # No valid moves available, pass turn
    
    # Iterate through all valid moves
    for move in valid_moves:
        simulated_board = deepcopy(chess_board)
        execute_move(simulated_board, move, player)
        score = self.alpha_beta_pruning(simulated_board, player, opponent, max_depth - 1, depth, alpha, beta)
        
        if score > best_score:
            best_score = score
            best_move = move
        
        alpha = max(alpha, score)
        if beta <= alpha:
            break
    
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    
    return best_move
  
  def alpha_beta_pruning(self, chess_board, player, opponent, max_depth, depth, alpha, beta):
    if max_depth == 0 or check_endgame(chess_board, player, opponent):
        return self.evaluate_board(chess_board, player, opponent, depth)
    
    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
        return self.alpha_beta_pruning(chess_board, opponent, player, max_depth - 1, alpha, beta)
    
    if player == self.max_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = execute_move(chess_board, move, player)
            eval = self.alpha_beta_pruning(new_board, opponent, player, max_depth - 1, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_board = execute_move(chess_board, move, player)
            eval = self.alpha_beta_pruning(new_board, opponent, player, max_depth - 1, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval
    
  def evaluate_board(self, chess_board, player, opponent, depth):
    # Get dynamic weights based on depth and board size
    weights = self.get_dynamic_weights(depth, chess_board)
    w_coin, w_mobility, w_corner, w_stability = weights
    
    # Calculate the scores for both the player and opponent
    coin_parity_score = self.coin_parity(chess_board, player)
    mobility_score = self.mobility(chess_board, player)
    corner_potential_score = self.corner_potential(chess_board, player)
    stability_score = self.stability(chess_board, player)
    
    opponent_coin_parity_score = self.coin_parity(chess_board, opponent)
    opponent_mobility_score = self.mobility(chess_board, opponent)
    opponent_corner_potential_score = self.corner_potential(chess_board, opponent)
    opponent_stability_score = self.stability(chess_board, opponent)
    
    my_score = w_coin * coin_parity_score + w_mobility * mobility_score + w_corner * corner_potential_score + w_stability * stability_score
    opponent_score = w_coin * opponent_coin_parity_score + w_mobility * opponent_mobility_score + w_corner * opponent_corner_potential_score + w_stability * opponent_stability_score
    heuristic_score = my_score - opponent_score
    
    return heuristic_score

  def coin_parity(self, chess_board, player):
    '''
    Calculate the coin parity of the player.
    Goal: maximize the difference between your discs and the opponent's discs
    '''
    # compute the number of discs for the player and the opponent
    player_discs = np.sum(chess_board == player)
    opponent_discs = np.sum(chess_board == 3 - player)
    
    # calculate the coin parity score
    total_possible_discs = chess_board.size
    coin_parity_score = (player_discs - opponent_discs) / max(1, player_discs + opponent_discs) * (player_discs + opponent_discs) / total_possible_discs
    return coin_parity_score
  
  def mobility(self, chess_board, player):
    '''
    Calculate the mobility of the player.
    Goal: maximize the number of legal moves avaliable to you while minimizing the opponent's moves
    '''
    # compute the number of legal moves for the player and the opponent
    player_legal_moves = get_valid_moves(chess_board, player)
    opponent_legal_moves = get_valid_moves(chess_board, 3 - player)
    
    # calculate the mobility score
    mobility_score = (len(player_legal_moves) - len(opponent_legal_moves)) / max(1, len(player_legal_moves) + len(opponent_legal_moves))
    return mobility_score
  
  def corner_potential(self, chess_board, player):
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
      if chess_board[c_square] == player:
        raw_corner_potential_score -= 12
      elif chess_board[c_square] == 3 - player:
        raw_corner_potential_score += 12

    for x_square in x_squares:
      if chess_board[x_square] == player:
        raw_corner_potential_score -= 20
      elif chess_board[x_square] == 3 - player:
        raw_corner_potential_score += 20
    
    # normalize the raw corner pontential score
    min_corner_potential_score = len(c_squares) * -12 + len(x_squares) * -20
    max_corner_potential_score = len(c_squares) * 12 + len(x_squares) * 20
    
    corner_potential_score = 2 * (raw_corner_potential_score - min_corner_potential_score) / max(1, max_corner_potential_score - min_corner_potential_score) - 1
    return corner_potential_score
  
  def stability(self, chess_board, player):
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
        if chess_board[corner] == player:
            stability_score += 1  # Corner disc is always stable
            stable_discs.add(corner)
            # Recursively check connected discs
            self.check_stability(chess_board, player, corner, stable_discs, directions)
    
    # Normalize the stability score
    total_discs = chess_board.size
    stability_score = 2 * len(stable_discs) / total_discs - 1
    
    return stability_score

  def check_stability(self, chess_board, player, position, stable_discs, directions):
    '''
    Recursively check the stability of discs connected to a stable disc.
    '''
    row, col = position
    for direction in directions:
        d_row, d_col = direction
        new_row, new_col = row + d_row, col + d_col
        while 0 <= new_row < chess_board.shape[0] and 0 <= new_col < chess_board.shape[1]:
            if chess_board[new_row, new_col] == player and (new_row, new_col) not in stable_discs:
                stable_discs.add((new_row, new_col))
                self.check_stability(chess_board, player, (new_row, new_col), stable_discs, directions)
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
    
    # Scale depth thresholds based on the board size
    early_game_depth = board_size * 2  
    mid_game_depth = board_size * 6   
    late_game_depth = board_size * 10 
    
    # Define the weights based on depth and board size
    if depth < early_game_depth:
        return [0.4, 0.3, 0.2, 0.1]  
    elif depth < mid_game_depth: 
        return [0.25, 0.25, 0.25, 0.25] 
    else:  
        return [0.1, 0.2, 0.3, 0.4]

  def get_dynamic_max_depth(self, board_size, depth):
    '''
    Get the dynamic max depth based on the board size.
    the max depth decreases as the game progresses
    '''
    if depth < board_size * 2: # Early game
      return max(4, int(board_size / 2))
    elif depth < board_size * 6: # Mid game
      return max(3, int(board_size / 3))
    else: # Late game
      return max(2, int(board_size / 4))
    
    