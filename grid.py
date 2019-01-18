"""
Developed by Ashwin Rammohan & Ashwin Reddy

Implements the Grid class, which wraps around a 9x9x9 array representing a Sudoku grid
"""


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
        self.num_cells_determined = 0

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
    

    def assign_cell(self, coordinate, entry, increment_cells_determined = True):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """
        # logging.debug("Assigning entry {} the value {}".format(coordinate, entry))
        if increment_cells_determined:
            self.num_cells_determined += 1

        row = coordinate[0]
        col = coordinate[1]

        for j in range(9):
            self.eliminate_option_for_cell((row, j), entry)

        for i in range(9):
            self.eliminate_option_for_cell((i, col), entry)

        for i in range(- (row % 3),  3 -(row % 3) ):
            for j in range(- (col % 3), + 3 - (col % 3) ):
                self.eliminate_option_for_cell((row + i, col + j), entry)
        
        for i in range(9):
            self.grid[coordinate][i] = 0
        self.grid[coordinate][entry - 1] = 1
        

        return entry
    
    def double_clear(self):
        for cell in self.determined_cells:
            vi = self.viable_indices(cell)
            if len(vi) == 1:
                self.assign_cell(cell, vi[0] + 1, increment_cells_determined=False)

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
    

    def __repr__(self):
        def pretty_print_indices(indices):
            return """{} {} {}\n{} {} {}\n{} {} {}""".format(*[i if i in indices else " " for i in range(1,10)])
        
        strings = [[pretty_print_indices( indices_to_options(self.viable_indices((i,j)))).split("\n") for j in range(self.grid.shape[1])] for i in range(9)]

        line_separator = "|" + ("-" * 17 + "||")*2  + "-" * 17 + "|\n"
        massive_str = line_separator

        for idx, row in enumerate(strings):
            for i in range(3):
                grouped_strings = [ x[i] for x in row]
                grouped_strings.insert(3, "")
                grouped_strings.insert(7, "")
                massive_str += "|" + "|".join(grouped_strings) + "|\n"
            massive_str += 2 * line_separator if idx == 2 or idx == 5 else line_separator

        return massive_str

    @property
    def shortcircuited(self):
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) == 0:
                    return (i, j)
        return False
    

    @property
    def undetermined_cells(self):
        undetermined_cells = []
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) != 1:
                    undetermined_cells.append((i,j))
        return undetermined_cells
    
    @property
    def determined_cells(self):
        determined_cells = []
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) == 1:
                    determined_cells.append((i,j))
        return determined_cells