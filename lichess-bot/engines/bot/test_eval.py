import chess
from engines.bot.eval import get_evaluation


board = chess.Board()

print("Starting position:")
print(board)
print("Evaluation:", get_evaluation(board))

board.push_san("e4")
print("\nAfter 1.e4:")
print(board)
print("Evaluation:", get_evaluation(board))

board.push_san("c5")
print("\nAfter 1...c5:")
print(board)
print("Evaluation:", get_evaluation(board))
