import numpy as np
import logging


def pretty_print_indices(indices):
    return """{} {} {}\n{} {} {}\n{} {} {}
    """.format(*[i if i in indices else " " for i in range(1,10)])

def convert_indices_to_options(indices):
    return [idx + 1 for idx in indices]

def find_empty_location(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                l[0]=row
                l[1]=col
                return True
    return False

def used_in_row(sudoku, row, num):
    for i in range(9):
        if sudoku[row][i] == num:
            return True
    return False

def used_in_col(sudoku,col,num):
    for i in range(9):
        if(sudoku[i][col] == num):
            return True
    return False


def used_in_box(sudoku,row,col,num):
    for i in range(3):
        for j in range(3):
            if(sudoku[i+row][j+col] == num):
                return True
    return False


def check_location_is_safe(arr,row,col,num):
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num)


# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self, interrogator, grid = np.ones((9,9,9), dtype='int8')):
        logging.debug("Instantiating SudokuGrid")
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = grid
        self.interrogator = interrogator

    def take_into_account_new_entry(self, row: int, col: int, entry: int):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """
        logging.debug("Setting ({}, {}) to {}".format(row, col, entry))

        self.grid[row][col][:entry-1] = 0
        self.grid[row][col][entry:] = 0

        # TODO: use cleaner numpy indexing
        for j in range(0, 9):
            if j == col:
                continue
            # print(self.grid[row][j])
            self.grid[row][j][entry - 1] = 0

        # TODO: use cleaner numpy indexing
        for i in range(0, 9):
            if i == row:
                continue
            # print(self.grid[i][col])
            self.grid[i][col][entry - 1] = 0



        for i in range(row - (row % 3), row + 3-(row % 3) ):
            for j in range(col - (col % 3), col + 3-(col % 3) ):
                if i == row and j == col:
                    continue
                self.grid[i][j][entry - 1] = 0

        return entry


    def determine_cell(self, row, col, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """
        options = convert_indices_to_options(indices)

        if len(options) == 0:
            logging.critical("No options for the {} cell".format((row, col)))

        logging.debug("Determining the value of ({}, {}) with possibilities {}".format(row, col, options))

        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(options) == 2:
            # answer = self.interrogator.ask("Is the value {} ? ".format(options[0]))
            answer = self.interrogator.ask( ((row, col), "==", options[0] ))
            logging.debug(answer)
            idx = 0 if answer == True else 1
            return self.take_into_account_new_entry(row, col, options[idx])
        elif len(options) == 1:
            return self.take_into_account_new_entry(row, col, options[0])

        # Otherwise, use simple binary search to determine what the value is in a minimal number of questions
        pivot = round(len(indices) / 2)
        # answer = self.interrogator.ask("Is the value for the ({}, {}) cell greater than or equal to {} ? ".format(row, col, options[pivot]))
        answer = self.interrogator.ask(( (row, col), ">=", options[pivot]))
        logging.debug(answer)
        indices = indices[pivot:] if answer == True else indices[:pivot]

        return self.determine_cell(row, col, indices)
        # strike whichever half was eliminated
        # call itself again


    def viable_indices(self, row, col):
        # any entry with 1 is a viable index
        return np.where(self.grid[row][col] == 1)[0]

    def viable_options(self, row, col):
        return convert_indices_to_options(self.viable_indices(row, col))

    def __repr__(self):
        strings = []
        for i in range(self.grid.shape[0]):
            strings.append([pretty_print_indices( self.viable_options(i,j) ).split("\n") for j in range(self.grid.shape[1])])

        line_separator = "|" + "-" * 17 + "||" + "-" * 17 + "||" + "-" * 17 + "|\n"
        massive_str = line_separator

        for idx, row in enumerate(strings):
            for i in range(3):
                grouped_strings = [ x[i] for x in row]
                grouped_strings.insert(3, "")
                grouped_strings.insert(7, "")
                massive_str += "|" + "|".join(grouped_strings) + "|\n"
            massive_str += line_separator

            if idx == 2 or idx == 5:
                massive_str += line_separator

        return massive_str

    @property
    def completed(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # if it is completed, then there should only be 1 nonzero value here
                if np.count_nonzero(self.grid[i][j]) != 1:
                    return False
        return True

    @property
    def collapsed_grid(self):
        grid = []
        for i in range(9):
            buffer = []
            for j in range(9):
                if np.count_nonzero(self.grid[i][j] == 1) > 1:
                    buffer.append(0)
                else:
                    buffer.append( 1 + np.where(self.grid[i][j]==1)[0][0])
                # if np.count_nonzero(self.grid[i][j] == 1) > 1:
                #     buffer.append(0)
                # else:
                #     buffer.append(np.where(self.grid[i][j] == 1)[0][0])
            grid.append(buffer)
        return np.array(grid)

    def solve(self, arr):
        print(arr)
        l=[0,0]

        if(not find_empty_location(arr,l)):
            return True

        row=l[0]
        col=l[1]

        for num in range(1,10):

            if(check_location_is_safe(arr,row,col,num)):

                arr[row][col]=num

                if(self.solve(arr)):
                    return True

                arr[row][col] = 0

        return False
