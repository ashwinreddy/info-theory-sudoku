import numpy as np
import logging


def pretty_print_indices(indices):
    return """{} {} {}\n{} {} {}\n{} {} {}
    """.format(*[i if i in indices else " " for i in range(1,10)])

def indices_to_options(indices):
    return [idx + 1 for idx in indices]

def solve(sg, cells):
    i = 0
    while not sg.completed:
        coordinate = np.unravel_index(cells[i], (9,9))
        sg.determine_cell(coordinate, sg.viable_indices(coordinate))
        i += 1
    print(sg)

# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self, interrogator, grid = np.ones((9,9,9), dtype='int8')):
        logging.debug("Instantiating SudokuGrid")
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = grid
        self.interrogator = interrogator
    
    def find_empty_location(self):
        array = np.sum(self.grid,axis=2) > 1
        i = 0
        for elem in np.nditer(array):
            if elem == True:
                return np.unravel_index(i, (9,9))
            i += 1
        return False
    
    def count_solutions(self):
        pass
        # cell = self.find_empty_location()
        # viable_solution = np.argmax(self.grid[cell] == 1)
        # grid_array_copy = np.copy(self.grid)
        # sg = SudokuGrid(grid_array_copy)
        # sg.assign_cell(cell[0], cell[1], viable_solution)
        # return sg.count_solutions()
        
    
    def assign_cell(self, coordinate, entry):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """

        row = coordinate[0]
        col = coordinate[1]

        self.grid[coordinate][:entry-1] = 0
        self.grid[coordinate][entry:] = 0

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


    def determine_cell(self, coordinate, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """
        options = indices_to_options(indices)
        row = coordinate[0]
        col = coordinate[1]

        if len(options) == 0:
            logging.critical("No options for the {} cell".format(coordinate))

        logging.debug("Determining the value of {} with possibilities {}".format(coordinate, options))

        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(options) == 2:
            # answer = self.interrogator.ask("Is the value {} ? ".format(options[0]))
            answer = self.interrogator.ask( (coordinate, "==", options[0] ))
            logging.debug(answer)
            idx = 0 if answer == True else 1
            return self.assign_cell(coordinate, options[idx])
        elif len(options) == 1:
            return self.assign_cell(coordinate, options[0])

        # Otherwise, use simple binary search to determine what the value is in a minimal number of questions
        pivot = round(len(indices) / 2)
        # answer = self.interrogator.ask("Is the value for the ({}, {}) cell greater than or equal to {} ? ".format(row, col, options[pivot]))
        answer = self.interrogator.ask(( coordinate, ">=", options[pivot]))
        logging.debug(answer)
        indices = indices[pivot:] if answer == True else indices[:pivot]

        return self.determine_cell(coordinate, indices)
        # strike whichever half was eliminated
        # call itself again


    def viable_indices(self, coordinate):
        # any entry with 1 is a viable index
        return np.where(self.grid[coordinate] == 1)[0]
        
    def __repr__(self):
        strings = []
        for i in range(self.grid.shape[0]):
            strings.append([pretty_print_indices( indices_to_options(self.viable_indices((i,j))) ).split("\n") for j in range(self.grid.shape[1])])

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

    # def count_solutions(self):
