import pandas as pd
import chess
import chess.pgn
import random
import os
import requests

# Public Lichess Opening Explorer API
EXPLORER_URL = "https://explorer.lichess.ovh/master"

def play_opening(board):
    """
    Try to play an opening move based on local CSV or, if unavailable,
    query Lichess Opening Explorer.
    """
    next_opening_moves = []

    # 1️⃣ Quick rule: play e4 as White’s first move if starting fresh
    if board.turn == chess.WHITE and board.fullmove_number == 1:
        return chess.Move.from_uci("e2e4")

    # 2️⃣ Local CSV-based openings
    new_board = chess.Board()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'openings.csv')

    if os.path.exists(file_path):
        chess_openings = pd.read_csv(file_path)
        chess_openings = chess_openings["moves"].tolist()

        for opening in chess_openings:
            moves_in_opening = opening.split()
            for index, move in enumerate(moves_in_opening):
                try:
                    new_board.push_san(move)
                    if board == new_board and index + 1 < len(moves_in_opening):
                        next_move = board.parse_san(moves_in_opening[index + 1]).uci()
                        next_opening_moves.append(next_move)
                except Exception:
                    break
            new_board.reset()

    # If we found a match locally, pick one and return
    if next_opening_moves:
        chosen = random.choice(next_opening_moves)
        return chess.Move.from_uci(chosen)

    # 3️⃣ No match in CSV — fall back to Lichess database
    try:
        params = {"fen": board.fen(), "moves": 8, "topGames": 0}
        r = requests.get(EXPLORER_URL, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            moves = data.get("moves", [])
            if moves:
                # pick most popular move overall
                best = max(moves, key=lambda m: m.get("white", 0) + m.get("black", 0))
                uci = best.get("uci")
                if uci:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        print(f"Using Lichess DB move: {uci}")
                        return move
    except Exception as e:
        print("Lichess API unavailable:", e)

    # 4️⃣ If nothing found, return None
    return None
