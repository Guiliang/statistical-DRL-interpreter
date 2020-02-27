"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""

from collections import namedtuple
from random import choice
from mimic_learner.mcts.monte_carlo_tree_search import MCTS, Node

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(_TTTB, Node):
    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            self.make_move(i) for i, value in enumerate(self.tup) if value is None
        }

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(self.tup) if value is None]
        return self.make_move(choice(empty_spots))

    def reward(self):
        if not self.terminal:
            raise RuntimeError("reward called on nonterminal board {0}".format(self))
        if self.winner is self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError("reward called on nonterminal board {0}".format(self))
        if self.turn is (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError("board has unknown winner type {0}".format(self.winner))

    def is_terminal(self):
        return self.terminal

    def make_move(self, index):
        tup = self.tup[:index] + (self.turn,) + self.tup[index + 1:]
        turn = not self.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
                "\n  1 2 3\n"
                + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
                + "\n"
        )


def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        board = board.make_move(index)
        print(board.to_pretty_string())
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game()
