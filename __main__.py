# Determines a sudoku grid by asking the user questions
import numpy as np


def pretty_print_indices(indices):
    return """{} {} {}\n{} {} {}\n{} {} {}
    """.format(*[i if i in indices else "" for i in range(1,10)])

# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self):
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = np.ones((9,9,9), dtype='int8')

    def take_into_account_new_entry(self, row, col, entry):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """
        print("Setting ({}, {}) to {}".format(row, col, entry))

        # TODO: use cleaner numpy indexing
        for j in range(0, 9):
            if j == col:
                continue
            print(self.grid[row][j])
            self.grid[row][j][entry - 1] = 0

        # TODO: use cleaner numpy indexing
        for i in range(0, 9):
            if i == row:
                continue
            print(self.grid[i][col])
            self.grid[i][col][entry - 1] = 0

      
        row_subgrid_idx, col_subgrid_idx = (row // 3, col // 3)

        self.grid[row_subgrid_idx: 3 * (row_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[row_subgrid_idx: 3 * (row_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[col_subgrid_idx: 3 * (col_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[col_subgrid_idx: 3 * (col_subgrid_idx + 1) + 1][entry:][entry - 1] = 0

    def determine_cell(self, row, col, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """
        
        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(indices) == 2:
            answer = input("Is the value {} ? ".format(indices[0] + 1))
            idx = 0 if answer == "yes" else 1
            val = indices[idx] + 1
            # take_into_account_new_entry takes in the value of the entry, not the index, so
            # here, we add 1 to indices[0] when passing it in
            self.take_into_account_new_entry(row, col, val)
            return val

        # Otherwise, use simple binary search to determine what the value is in a minimal number of questions
        pivot = len(indices) // 2
        answer = input("Is the value for the ({}, {}) cell higher than {} ? ".format(row, col, indices[pivot] + 1))
        if answer == "yes":
            # get rid of
            indices = indices[pivot + 1:]
            print("Was higher: ", indices)
        else:
            indices = indices[:pivot + 1]

            print("Was lower: ", indices)

        return self.determine_cell(row, col, indices)
        # strike whichever half was eliminated
        # call itself again


    def viable_indices(self, row, col):
        # any entry with 1 is a viable index
        return np.where(self.grid[row][col] == 1)[0]
    
    def __repr__(self):
        strings = []
        for i in range(1, 9):
            buffer = []
            for j in range(1, 9):
                arr = [idx + 1 for idx in self.viable_indices(i, j)]
                buffer.append(pretty_print_indices(arr).split("\n"))
            strings.append(buffer)
        # strings[i][j] are the nicely printed viable options for i, j
        # all the ones in the same row need to be collected
        
        massive_str = ""

        for row in strings:
            massive_str += "|" + "|".join([x[0] for x in row]) + "|\n"
            massive_str += "|" + "|".join([x[1] for x in row]) + "|\n"
            massive_str += "|" +  "|".join([x[2] for x in row]) + "|\n"
            massive_str += "-" * 49 + "\n"

        return massive_str


sg = SudokuGrid()
print(sg)
# print(sg.determine_cell(0, 0, sg.viable_indices(0,0)))
# print(sg.determine_cell(0, 1, sg.viable_indices(0,1)))
# # sg.take_into_account_new_entry(2, 2, 1)
