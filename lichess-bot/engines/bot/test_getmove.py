import chess
from engines.bot.main import get_move

# Create a custom position (random midgame, not in opening book)
board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
print(board)

move = get_move(board, depth=2)
print("Final chosen move:", move)
