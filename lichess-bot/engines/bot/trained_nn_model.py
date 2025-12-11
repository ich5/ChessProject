import os
import chess
import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
PGN_PATH = "datasets/lichess_db_standard_rated_2025-09.pgn"
STOCKFISH_PATH = r"C:\Users\gaire\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

GAMES_PER_RUN = 50000      # How many games THIS run will process
DEPTH = 3                 # Stockfish depth
MODEL_PATH = "trained_nn_model.pth"
SAVE_INTERVAL = 1000       # Save partial model every N games
EPOCHS = 40                # Training epochs per run
BATCH_SIZE = 4096          # Mini-batch size


# ===============================
# FEATURE HELPERS (15 FEATURES)
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
    # Squares around the king
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type == chess.PAWN:
            pawn_shield += 1
    return defenders + pawn_shield


def pawn_structure_score(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(p) for p in pawns]
    file_set = set(files)

    # doubled pawns
    doubled = sum(files.count(f) > 1 for f in file_set)

    # isolated pawns (no pawn on adjacent files)
    isolated = 0
    for f in file_set:
        if all(abs(f - other) > 1 for other in file_set if other != f):
            isolated += 1

    # negative because these are weaknesses
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
            # same file or adjacent files
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
    # Simple heuristic: based on non-pawn material
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
    # opening ~ 40, endgame ~ 0
    return 1.0 - min(total, 40) / 40.0  # 0 = opening, 1 = endgame


def extract_features(board: chess.Board) -> np.ndarray:
    """
    Returns a 15-dimensional feature vector (as Python list).
    All features are roughly normalized to [-1, 1].
    """

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

    features = [
        mat / 40.0,              # 1 material
        mob / 60.0,              # 2 mobility diff
        turn,                    # 3 side to move
        center / 16.0,           # 4 center control diff
        ks_white / 10.0,         # 5 king safety white
        ks_black / 10.0,         # 6 king safety black
        ps_white / 8.0,          # 7 pawn structure white
        ps_black / 8.0,          # 8 pawn structure black
        act_white / 50.0,        # 9 activity white
        act_black / 50.0,        # 10 activity black
        float(bp_white),         # 11 bishop pair white (0/1)
        float(bp_black),         # 12 bishop pair black (0/1)
        pp_white / 8.0,          # 13 passed pawns white
        pp_black / 8.0,          # 14 passed pawns black
        phase                    # 15 game phase (0..1)
    ]
    return features


# ===============================
# MODEL DEFINITION
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
# TRAINING LOOP
# ===============================
def train_nn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Training on {str(device).upper()}")

    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    loss_fn = nn.MSELoss()

    # 🔁 Resume from existing model if present
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("🔁 Resumed training from existing model checkpoint.")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    X, y = [], []

    with open(PGN_PATH, encoding="utf-8", errors="ignore") as pgn:
        for i in tqdm(range(GAMES_PER_RUN), desc="Processing games"):
            game = chess.pgn.read_game(pgn)
            if not game:
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                feats = extract_features(board)

                info = engine.analyse(board, limit=chess.engine.Limit(depth=DEPTH))
                score = info["score"].white().score(mate_score=10000)
                if score is not None:
                    X.append(feats)
                    y.append(score / 1000.0)  # normalize to about [-10, 10]

            if i > 0 and i % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"💾 Saved partial model at game {i}")

    engine.quit()

    print(f"\n✅ Collected {len(X)} samples.")
    if len(X) == 0:
        print("⚠️ No samples collected, aborting training.")
        return

    X = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32).unsqueeze(1).to(device)

    print("🚀 Starting full training...")
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(1, EPOCHS + 1):
        # Shuffle each epoch
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Mini-batch training
        total_loss = 0.0
        for start in range(0, num_samples, BATCH_SIZE):
            end = start + BATCH_SIZE
            xb = X[start:end]
            yb = y[start:end]

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end - start)

        scheduler.step()
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.6f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Model checkpoint saved at epoch {epoch}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Final model saved as {MODEL_PATH}")


if __name__ == "__main__":
    train_nn_model()
