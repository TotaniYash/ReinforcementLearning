"""
Tabular Q-learning for 3x3 Tic-Tac-Toe.

This version trains X as the learning agent against a random O opponent.
It uses a Q-table dictionary keyed by (state, move), where:

- state is a tuple representation of the board
- move is an integer from 0 to 8

Board representation:
    0  = empty square
    1  = X / learning agent
   -1  = O / random opponent

Index map:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

Board = np.ndarray
State = Tuple[int, ...]
QTable = Dict[Tuple[State, int], float]


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

def legal_moves(board: Board) -> np.ndarray:
    """Return the indices of all empty squares."""
    return np.where(board == 0)[0]


def make_move(board: Board, move: int, player: int) -> Board:
    """Return a new board after player makes move."""
    if move not in legal_moves(board):
        raise ValueError(f"Illegal move: {move}")

    new_board = board.copy()
    new_board[move] = player
    return new_board


def winner(board: Board) -> int:
    """
    Return the winner of the board.

    Returns:
        1  if X wins
       -1  if O wins
        0  if there is no winner yet or the game is a draw
    """
    winning_lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]

    for line in winning_lines:
        line_sum = board[line[0]] + board[line[1]] + board[line[2]]
        if line_sum == 3:
            return 1
        if line_sum == -3:
            return -1

    return 0


def game_over(board: Board) -> bool:
    """Return True if the game has ended."""
    return winner(board) != 0 or len(legal_moves(board)) == 0


def reward(board: Board) -> int:
    """
    Return the reward from X's perspective.

    X win: +1
    O win: -1
    Draw or unfinished game: 0
    """
    if game_over(board):
        return winner(board)
    return 0


# ---------------------------------------------------------------------
# Agent utilities
# ---------------------------------------------------------------------

def board_state(board: Board) -> State:
    """Convert a NumPy board into a tuple so it can be used as a dictionary key."""
    return tuple(board.tolist())


def get_q(q_table: QTable, board: Board, move: int) -> float:
    """Return Q(board, move), or 0.0 if it has not been seen before."""
    return q_table.get((board_state(board), int(move)), 0.0)


def max_q(q_table: QTable, board: Board) -> float:
    """Return the maximum Q-value among legal moves from this board."""
    moves = legal_moves(board)
    if len(moves) == 0:
        return 0.0

    best_value = -float("inf")
    for move in moves:
        value = get_q(q_table, board, int(move))
        if value > best_value:
            best_value = value

    return best_value


def best_move(q_table: QTable, board: Board) -> int:
    """Return the legal move with the highest Q-value."""
    moves = legal_moves(board)
    if len(moves) == 0:
        raise ValueError("No legal moves available.")

    best_action = None
    best_value = -float("inf")

    for move in moves:
        value = get_q(q_table, board, int(move))
        if value > best_value:
            best_value = value
            best_action = int(move)

    return best_action


def random_move(board: Board) -> int:
    """Return a random legal move."""
    moves = legal_moves(board)
    if len(moves) == 0:
        raise ValueError("No legal moves available.")

    return int(np.random.choice(moves))


def choose_move(q_table: QTable, board: Board, epsilon: float) -> int:
    """
    Choose a move using epsilon-greedy action selection.

    With probability epsilon, choose randomly.
    With probability 1 - epsilon, choose the best known move.
    """
    if np.random.rand() < epsilon:
        return random_move(board)
    return best_move(q_table, board)


# ---------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------

def update_q(
    q_table: QTable,
    board: Board,
    move: int,
    step_reward: float,
    next_board: Board,
    alpha: float,
    gamma: float,
) -> None:
    """Apply one Q-learning update."""
    old_value = get_q(q_table, board, move)
    future_value = max_q(q_table, next_board)
    new_value = old_value + alpha * (step_reward + gamma * future_value - old_value)

    q_table[(board_state(board), int(move))] = new_value


def train_one_game(q_table: QTable, alpha: float, gamma: float, epsilon: float) -> Board:
    """
    Train X for one game against a random O opponent.

    This version updates X immediately after X moves.
    It also punishes X's previous move if O wins immediately afterward.
    It does not yet fully propagate values from later turns back to the opening move.
    """
    board = np.zeros(9, dtype=int)
    player = 1

    last_x_state = None
    last_x_move = None

    while not game_over(board):
        if player == 1:
            move = choose_move(q_table, board, epsilon)

            last_x_state = board
            last_x_move = move

            next_board = make_move(board, move, player)
            step_reward = reward(next_board)
            update_q(q_table, board, move, step_reward, next_board, alpha, gamma)

            board = next_board
            player = -player

        else:
            move = random_move(board)
            board = make_move(board, move, player)

            if winner(board) == -1 and last_x_state is not None and last_x_move is not None:
                update_q(q_table, last_x_state, last_x_move, -1, board, alpha, gamma)

            player = -player

    return board


def train_many_games(n_games: int, alpha: float = 0.1, gamma: float = 0.9) -> QTable:
    """
    Train X over many games using decaying epsilon.

    Epsilon starts near 1.0 and decays to a floor of 0.05.
    """
    q_table: QTable = {}

    for i in range(n_games):
        epsilon = max(0.05, 1 - i / n_games)
        train_one_game(q_table, alpha, gamma, epsilon)

    return q_table


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def play_trained_game(q_table: QTable) -> Board:
    """
    Play one evaluation game.

    X uses the learned Q-table greedily.
    O plays randomly.
    No learning happens here.
    """
    board = np.zeros(9, dtype=int)
    player = 1

    while not game_over(board):
        if player == 1:
            move = best_move(q_table, board)
        else:
            move = random_move(board)

        board = make_move(board, move, player)
        player = -player

    return board


def evaluate_agent(q_table: QTable, n_games: int) -> Tuple[int, int, int]:
    """
    Evaluate the trained X agent against random O.

    Returns:
        (x_wins, o_wins, draws)
    """
    x_wins = 0
    o_wins = 0
    draws = 0

    for _ in range(n_games):
        board = play_trained_game(q_table)
        result = winner(board)

        if result == 1:
            x_wins += 1
        elif result == -1:
            o_wins += 1
        else:
            draws += 1

    return x_wins, o_wins, draws


def print_empty_board_q_values(q_table: QTable) -> None:
    """Print the learned Q-values for each opening move."""
    board = np.zeros(9, dtype=int)

    for move in legal_moves(board):
        print(f"Move {move}: {get_q(q_table, board, int(move)):.4f}")


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------

def main() -> None:
    np.random.seed(0)

    q_table = train_many_games(n_games=50_000, alpha=0.1, gamma=0.9)

    print(f"Number of learned state-action pairs: {len(q_table)}")
    print("Evaluation over 1,000 games:", evaluate_agent(q_table, 1_000))

    print("\nOpening move Q-values:")


if __name__ == "__main__":
    main()
