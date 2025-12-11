import os
import chess
import torch
import torch.nn as nn
import numpy as np
from .material import get_material
from . import positions

# ===============================
#  Neural Network Definition
# ===============================
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ===============================
#  Load Model
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "trained_nn_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_nn = None

if os.path.exists(model_path):
    trained_nn = ChessNet().to(device)
    trained_nn.load_state_dict(torch.load(model_path, map_location=device))
    trained_nn.eval()
    print(f"🧠 Neural network model loaded from {model_path}")
else:
    print("⚠️ Neural network model not found. Using fallback evaluation.")

# ===============================
#  Feature Extraction (same as training)
# ===============================
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def material_balance(board: chess.Board) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    score = 0
    for sq, piece in board.piece_map().items():
        v = values.get(piece.piece_type, 0)
        score += v if piece.color == chess.WHITE else -v
    return score


def mobility_diff(board: chess.Board) -> int:
    board_turn_backup = board.turn

    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))

    board.turn = board_turn_backup
    return white_moves - black_moves


def center_control_diff(board: chess.Board) -> int:
    score = 0
    for sq in CENTER_SQUARES:
        w = len(board.attackers(chess.WHITE, sq))
        b = len(board.attackers(chess.BLACK, sq))
        score += (w - b)
    return score


def king_safety(board: chess.Board, color: bool) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    defenders = len(board.attackers(color, king_sq))
    pawn_shield = 0
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type == chess.PAWN:
            pawn_shield += 1
    return defenders + pawn_shield


def pawn_structure_score(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(p) for p in pawns]
    file_set = set(files)

    doubled = sum(files.count(f) > 1 for f in file_set)

    isolated = 0
    for f in file_set:
        if all(abs(f - other) > 1 for other in file_set if other != f):
            isolated += 1

    return -(doubled + isolated)


def piece_activity(board: chess.Board, color: bool) -> int:
    activity = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(piece_type, color):
            activity += len(board.attacks(sq))
    return activity


def bishop_pair(board: chess.Board, color: bool) -> int:
    return 1 if len(board.pieces(chess.BISHOP, color)) >= 2 else 0


def passed_pawns(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    count = 0
    for sq in pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        is_passed = True
        for ep in enemy_pawns:
            ef = chess.square_file(ep)
            er = chess.square_rank(ep)
            if abs(ef - file) <= 1:
                if color == chess.WHITE and er > rank:
                    is_passed = False
                    break
                if color == chess.BLACK and er < rank:
                    is_passed = False
                    break
        if is_passed:
            count += 1
    return count


def game_phase(board: chess.Board) -> float:
    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    total = 0
    for sq, piece in board.piece_map().items():
        if piece.piece_type in values:
            total += values[piece.piece_type]
    return 1.0 - min(total, 40) / 40.0


def extract_features(board: chess.Board) -> np.ndarray:
    mat = material_balance(board)
    mob = mobility_diff(board)
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    center = center_control_diff(board)

    ks_white = king_safety(board, chess.WHITE)
    ks_black = king_safety(board, chess.BLACK)

    ps_white = pawn_structure_score(board, chess.WHITE)
    ps_black = pawn_structure_score(board, chess.BLACK)

    act_white = piece_activity(board, chess.WHITE)
    act_black = piece_activity(board, chess.BLACK)

    bp_white = bishop_pair(board, chess.WHITE)
    bp_black = bishop_pair(board, chess.BLACK)

    pp_white = passed_pawns(board, chess.WHITE)
    pp_black = passed_pawns(board, chess.BLACK)

    phase = game_phase(board)

    feats = [
        mat / 40.0,
        mob / 60.0,
        turn,
        center / 16.0,
        ks_white / 10.0,
        ks_black / 10.0,
        ps_white / 8.0,
        ps_black / 8.0,
        act_white / 50.0,
        act_black / 50.0,
        float(bp_white),
        float(bp_black),
        pp_white / 8.0,
        pp_black / 8.0,
        phase
    ]

    return np.array([feats], dtype=np.float32)


# ===============================
#  Evaluation Function
# ===============================
def get_evaluation(board: chess.Board) -> float:
    # Terminal positions
    if board.is_checkmate():
        return -9999 if board.turn == chess.WHITE else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Neural Network evaluation
    if trained_nn is not None:
        try:
            features = extract_features(board)
            with torch.no_grad():
                x = torch.tensor(features, device=device)
                output = trained_nn(x).item()
            # scale back from [-10,10]-ish to centipawns
            return float(output * 1000.0)
        except Exception as e:
            print("⚠️ NN evaluation failed, fallback:", e)

    # Fallback: simple material + pawn-square
    total_material = get_material(board)
    pawnsq = sum([positions.pawn[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq += sum([-positions.pawn[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])
    return total_material + pawnsq
