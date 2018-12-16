# Determines a sudoku grid by asking the user questions
import numpy as np

# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self):
        # a list (rows) of lists (cell in the row) of lists (viable possibilities)
        # the postcondition is that self.grid will basically be just a list (rows) of lists (cell values in the row)

        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = np.ones((9,9,9), dtype='int8')

    def take_into_account_new_entry(self, row, col, entry):
        print("Setting ({}, {}) to {}".format(row, col, entry))

        # strike this entry from all those in the same row as this one

        # TODO: make this not suck i.e. make numpy indexing work somehow
        for j in range(0, 9):
            if j == col:
                continue
            print(self.grid[row][j])
            self.grid[row][j][entry - 1] = 0


        # self.grid[row][:col][entry - 1] = 0
        # self.grid[row][col:][entry - 1] = 0

        # strike this entry from all those in the same column as this one

        # TODO: make this not suck i.e. make numpy indexing work somehow
        for i in range(0, 9):
            if i == row:
                continue
            print(self.grid[i][col])
            self.grid[i][col][entry - 1] = 0 # IndexError: index 9 is out of bounds for axis 0 with size 9

        # self.grid[:row][col][entry - 1] = 0 #IndexError: index 2 is out of bounds for axis 0 with size 2
        # self.grid[row:][col][entry - 1] = 0

        # strike this entry from all those in the same subgrid as this one
        row_subgrid_idx, col_subgrid_idx = (row // 3, col // 3)

        self.grid[row_subgrid_idx: 3 * (row_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[row_subgrid_idx: 3 * (row_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[col_subgrid_idx: 3 * (col_subgrid_idx + 1) + 1][entry - 1] = 0
        self.grid[col_subgrid_idx: 3 * (col_subgrid_idx + 1) + 1][entry:][entry - 1] = 0

    def determine_cell(self, row, col, indices):
        # Extract the viable possibilities for this cell
        # indices = self.viable_indices(row, col)
        if len(indices) == 2:
            answer = input("Is the value {} ? ".format(indices[0] + 1))

            if answer == "yes":
                #take_into_account_new_entry takes in the value of the entry, not the index, so
                #here, we add 1 to indices[0] when passing it in
                # self.take_into_account_new_entry(self, row, col, indices[0] + 1)
                # return indices[0]
                idx = 0
                #  (l 17)   self.grid[row][:col][entry - 1] = 0
                # IndexError: index 5 is out of bounds for axis 0 with size 0
            else:
                idx = 1

            val = indices[idx] + 1
            self.take_into_account_new_entry(row, col, val)
            return val

        # Then, use simple binary search to determine what the value is in a minimal number of questions
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
        return np.where(self.grid[row][col] == 1)[0]



sg = SudokuGrid()
print(sg.determine_cell(0,0, sg.viable_indices(0,0)))
# sg.take_into_account_new_entry(2, 2, 1)
