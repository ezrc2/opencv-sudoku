class Sudoku:

    def __init__(self, board):
        self.board = board
        self.size = len(board)

    def find_next_empty(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == -1:
                    return i, j
        return -1, -1


if __name__ == "__main__":
    test_board = [
        [-1, 2, -1, 9, -1, -1, -1, -1, 6],
        [-1, -1, -1, 3, 8, 4, 9, -1, -1],
        [8, 3, -1, -1, -1, -1, -1, 7, -1],
        [-1, 5, -1, -1, 9, -1, -1, -1, -1],
        [-1, 1, -1, 7, -1, 2, -1, 3, -1],
        [-1, -1, -1, -1, 4, -1, -1, 9, -1],
        [-1, 7, -1, -1, -1, -1, -1, 2, 8],
        [-1, -1, 6, 1, 2, 8, -1, -1, -1],
        [2, -1, -1, -1, -1, 5, -1, 4, -1]
    ]
    print(test_board)
    Sudoku(test_board)
    print(test_board)