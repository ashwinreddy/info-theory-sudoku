# Determines a sudoku grid by asking the user questions
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

def pretty_print_indices(indices): 
    return """{} {} {}\n{} {} {}\n{} {} {}
    """.format(*[i if i in indices else " " for i in range(1,10)])

def convert_indices_to_options(indices):
    return [idx + 1 for idx in indices]

# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self, grid = np.ones((9,9,9), dtype='int8')):
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = grid
        self.questions_asked = 0

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

      
        row_subgrid_idx, col_subgrid_idx = (row // 3, col // 3)
        for i in range(row_subgrid_idx, 3 *(row_subgrid_idx+1) ):
            for j in range(col_subgrid_idx, 3*(col_subgrid_idx +1)):
                if i == row and j == col:
                    continue
                self.grid[i][j][entry - 1] = 0
    
    def ask_question(self, prompt):
        answer = input(prompt)
        self.questions_asked += 1
        return answer

    def determine_cell(self, row, col, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """

       
        options = convert_indices_to_options(indices)

        logging.debug("Determining the value of ({}, {}) with possibilities {}".format(row, col, options))
        
        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(options) == 2:
            logging.debug("There are only two possibilities, {} and {}".format(indices[0]+1, indices[1]+1))
            answer = self.ask_question("Is the value {} ? ".format(options[0]))
            idx = 0 if answer == "yes" else 1
            val = indices[idx] + 1
            # take_into_account_new_entry takes in the value of the entry, not the index, so
            # here, we add 1 to indices[0] when passing it in
            self.take_into_account_new_entry(row, col, val)
            return val

        # Otherwise, use simple binary search to determine what the value is in a minimal number of questions
        pivot = round(len(indices) / 2)
        answer = self.ask_question("Is the value for the ({}, {}) cell greater than or equal to {} ? ".format(row, col, options[pivot]))
        indices = indices[pivot:] if answer == "yes" else indices[:pivot]

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
 
sg = SudokuGrid()
# sg.determine_cell(0, 0, sg.viable_indices(0,0))
# sg.determine_cell(0, 1, sg.viable_indices(0,1))