import chess.pgn
import pandas as pd
from tqdm import tqdm

def extract_features(board):
    """Extracts simple numeric features from a chess position."""
    material = sum(piece_value(p) for p in board.piece_map().values())
    mobility = len(list(board.legal_moves))
    turn = 1 if board.turn == chess.WHITE else -1
    return [material, mobility, turn]

def piece_value(piece):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    return values[piece.piece_type] if piece.color == chess.WHITE else -values[piece.piece_type]

def build_dataset(pgn_path, max_games=2000, max_positions=3):
    """
    Extracts features and results from PGN file.
    Each game contributes up to `max_positions` random positions.
    """
    data = []
    print(f"Reading {pgn_path} ...")

    with open(pgn_path, encoding="utf-8") as pgn:
        for game_index in tqdm(range(max_games)):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            result = game.headers.get("Result")
            if result == "1-0":
                y = 1
            elif result == "0-1":
                y = -1
            else:
                y = 0

            board = game.board()
            positions = []
            for move in game.mainline_moves():
                board.push(move)
                positions.append(extract_features(board))

            # Take a few random positions from each game
            for pos in positions[:max_positions]:
                data.append(pos + [y])

    df = pd.DataFrame(data, columns=["material", "mobility", "turn", "result"])
    df.to_csv("training_positions.csv", index=False)
    print(f"✅ Saved {len(df)} positions to training_positions.csv")

if __name__ == "__main__":
    build_dataset(r"C:\ChessAIProject\lichess-bot\datasets\lichess_db_standard_rated_2025-09.pgn")
