import numpy as np
import time
import chess
from .eval import get_evaluation


def order_moves(board):
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    scored_moves = []

    for move in board.legal_moves:
        score = 0

        if board.is_capture(move):
            score += 10
        if move.to_square in center_squares:
            score += 5

        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if chess.square_rank(move.to_square) in [0, 7]:
                score += 15

        scored_moves.append((score, move))

    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in scored_moves]


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return get_evaluation(board)

    if maximizing_player:
        max_eval = -np.inf
        for move in order_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = np.inf
        for move in order_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def iterative_deepening(board, max_depth, time_limit=3):
    start_time = time.time()

    maximizing = (board.turn == chess.WHITE)

    best_move = None
    best_score = -999999 if maximizing else 999999

    for depth in range(1, max_depth + 1):

        if time.time() - start_time > time_limit:
            break

        depth_best_move = None
        depth_best_score = -np.inf if maximizing else np.inf

        for move in order_moves(board):
            board.push(move)

            # correct maximizing logic
            score = minimax(board, depth - 1, -np.inf, np.inf, not maximizing)

            board.pop()

            if maximizing:
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
            else:
                if score < depth_best_score:
                    depth_best_score = score
                    depth_best_move = move

        if depth_best_move is not None:
            best_move = depth_best_move
            best_score = depth_best_score

        print(f"Depth {depth}: Best move {best_move}, eval {best_score}")

    return best_move, best_score
