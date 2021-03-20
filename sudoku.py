import numpy as np

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

    def is_valid_position(self, number, row, col):
        # Check if number is in the row and column
        board_row = self.board[row]
        board_col = [self.board[i][col] for i in range(self.size)]
        if number in board_row or number in board_col:
            return False

        # Check 3x3 sub-grid
        grid_size = int(self.size ** 0.5)
        sub_row = (row // grid_size) * grid_size
        sub_col = (col // grid_size) * grid_size
        for i in range(sub_row, sub_row + grid_size):
            for j in range(sub_col, sub_col + grid_size):
                if self.board[i][j] == number:
                    return False
        
        return True

    def solve(self):
        row, col = self.find_next_empty()
        if row == col == -1:
            return True

        for num in range(1, self.size + 1):
            if self.is_valid_position(num, row, col):
                self.board[row][col] = num
                if self.solve():
                    return True

            self.board[row][col] = -1
        
        return False

def print_board(board):
    for row in board:
        print(row)
    print("")

if __name__ == "__main__":
    test_board = np.array([
        [-1, 2, -1, 9, -1, -1, -1, -1, 6],
        [-1, -1, -1, 3, 8, 4, 9, -1, -1],
        [8, 3, -1, -1, -1, -1, -1, 7, -1],
        [-1, 5, -1, -1, 9, -1, -1, -1, -1],
        [-1, 1, -1, 7, -1, 2, -1, 3, -1],
        [-1, -1, -1, -1, 4, -1, -1, 9, -1],
        [-1, 7, -1, -1, -1, -1, -1, 2, 8],
        [-1, -1, 6, 1, 2, 8, -1, -1, -1],
        [2, -1, -1, -1, -1, 5, -1, 4, -1]
    ])
    sudoku = Sudoku(test_board)
    sudoku.solve()
    print_board(sudoku.board)