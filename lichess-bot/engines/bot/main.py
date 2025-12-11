import chess
import numpy as np

from .opening import play_opening
from .minimax import minimax, iterative_deepening


def get_move(board, depth):

    # 1️⃣ Try opening book first
    opening_move = play_opening(board)
    if opening_move:
        print("PLAYING OPENING MOVE:", opening_move)
        return opening_move

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    print("Thinking with iterative deepening...")

    # 2️⃣ Correct tuple unpack
    try:
        best_move, best_score = iterative_deepening(
            board,
            max_depth=depth,
            time_limit=3
        )
    except Exception as e:
        print("Iterative deepening error:", e)
        best_move = None

    # 3️⃣ Accept the move only if valid
    if best_move is not None and best_move in legal_moves:
        print("CHOSEN MOVE:", best_move)
        return best_move

    # 4️⃣ Fallback minimax search
    print("⚠️ Deepening failed, using minimax fallback...")

    best_score = -np.inf if board.turn == chess.WHITE else np.inf
    fallback_move = None

    maximizing = (board.turn == chess.WHITE)

    for move in legal_moves:
        board.push(move)
        try:
            score = minimax(board, depth - 1, -np.inf, np.inf, not maximizing)
        except Exception:
            score = None
        board.pop()

        if score is None:
            continue

        if maximizing:
            if score > best_score:
                best_score = score
                fallback_move = move
        else:
            if score < best_score:
                best_score = score
                fallback_move = move

    # 100% guaranteed safe return
    if fallback_move:
        print("FALLBACK MOVE:", fallback_move)
        return fallback_move

    print("❗ All searches failed — returning first legal move")
    return legal_moves[0]
