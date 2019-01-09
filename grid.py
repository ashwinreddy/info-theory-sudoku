import numpy as np
import logging


def indices_to_options(indices):
    return list(map(lambda x: x + 1, indices))

def solve_by_backtracking(sg, mode = "lower"):
    empty_location = sg.find_empty_location()

    if empty_location == False:
        return True

    if mode == "lower":
        values = range(1, 10)
    elif mode == "upper":
        values = range(9, 0, -1)

    for num in values:
        vi = sg.viable_indices(empty_location)
        if num in vi:
            
            sg.assign_cell(empty_location, num)

            if solve_by_backtracking(sg, mode):
                return True
            
            sg.grid[empty_location] = vi

    return False

class Grid(object):
    def __init__(self, grid):
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = grid

    def __getitem__(self, coordinate):
        return self.grid[coordinate]
    
    def find_empty_location(self):
        array = np.sum(self.grid, axis=2) > 1
        i = 0
        for elem in np.nditer(array):
            if elem == True:
                return np.unravel_index(i, (9,9))
            i += 1
        return False
    
    def eliminate_option_for_cell(self, coordinate, entry):
        self[coordinate][entry - 1] = 0    
    

    def assign_cell(self, coordinate, entry):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """
        logging.debug("Assigning entry {} the value {}".format(coordinate, entry))
        # self.num_cells_determined += 1
        row = coordinate[0]
        col = coordinate[1]

        self[coordinate][:entry-1] = 0
        self[coordinate][entry:] = 0

        for j in range(0, 9):
            self.eliminate_option_for_cell((row, j), entry)

        for i in range(0, 9):
            self.eliminate_option_for_cell((i, col), entry)

        for i in range(- (row % 3),  3 -(row % 3) ):
            for j in range(- (col % 3), + 3 - (col % 3) ):
                self.eliminate_option_for_cell((row + i, col + j), entry)

        self[coordinate][entry - 1] = 1

        return entry


    def viable_indices(self, coordinate):
        """
        Returns a list of indices (entries - 1) from the grid representing possible values for that cell
        """
        # any entry with 1 is a viable index
        vi = np.where(self[coordinate] == 1)[0]
        # logging.debug("Viable indices for {}: {}".format(coordinate, vi))
        return vi
    
    def __sub__(self, other_grid):
        return self.grid - other_grid.grid
    
