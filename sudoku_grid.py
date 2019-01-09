import numpy as np
import logging
from grid import Grid, indices_to_options

class Tree:
    def __init__(self, children):
        self.children = children

# This class will keep track of all the known and unknown cells
class SudokuGrid(Grid):
    def __init__(self, questioner = None, checkpoint_frequency = None, grid = np.ones((9,9,9), dtype='int8')):
        super(SudokuGrid, self).__init__(grid)
        
        self.questioner = questioner
        self.num_cells_determined = 0
        self.record_checkpoint()
        self.checkpoint_frequency = checkpoint_frequency
    
    def __eq__(self, other_grid):
        return np.array_equal(self.grid, other_grid)
    
    def rewind_to_last_checkpoint(self):
        logging.warning("Rewinding back to checkpoint copy")
        self.checkpoint_frequency
        self.grid = np.copy(self.ckpt_copy)

        if self.num_cells_determined % self.checkpoint_frequency == 0 and self.num_cells_determined > 0:
            self.num_cells_determined -= self.checkpoint_frequency
        else:
            self.num_cells_determined -= self.num_cells_determined - (self.num_cells_determined // self.checkpoint_frequency) * self.checkpoint_frequency
    
    def record_checkpoint(self):
        logging.debug("Recording checkpoint")
        self.ckpt_copy = np.copy(self.grid)
    
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
  
    @property
    def collapsed_grid(self):
        # logging.critical(self.grid.shape)
        # print(self.grid)
        grid = [[0 if np.count_nonzero(self[i][j] == 1) > 1 else 1 + np.where(self[i][j] == 1)[0][0] for j in range(9) ] for i in range(9)]
        return np.array(grid)


    def determine_cell(self, coordinate, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """
        if len(indices) == 0:
            logging.warning("Rewinding because cell has no options")
            print(self.shortcircuited)
            self.rewind_to_last_checkpoint()
            return

        options = indices_to_options(indices)
        logging.debug("Determining the value of {} with possibilities {}".format(coordinate, options))

        

        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(options) == 2:
            answer = self.questioner.ask_cdq( (coordinate, "==", options[1] ))
            logging.debug(answer)
            return self.assign_cell(coordinate, options[bool(answer)])
        elif len(options) == 1:
            return self.assign_cell(coordinate, options[0])

        # Otherwise, use simple binary search to determine what the value is in a minimal number of questions
        pivot = round(len(indices) / 2)

        answer = self.questioner.ask_cdq(( coordinate, ">=", options[pivot]))
        logging.debug(answer)
        # determine the cell with the new valid possibilities
        return self.determine_cell(coordinate, indices[pivot:] if answer == True else indices[:pivot])
        


    def viable_indices(self, coordinate):
        """
        Returns a list of indices (entries - 1) from the grid representing possible values for that cell
        """
        # any entry with 1 is a viable index
        vi = np.where(self[coordinate] == 1)[0]
        # logging.debug("Viable indices for {}: {}".format(coordinate, vi))
        return vi

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
    def completed(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # if it is completed, then there should only be 1 nonzero value here
                if np.count_nonzero(self[i][j]) != 1:
                    return False
        return True

    
    @property
    def first_undetermined_cell(self):
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) != 1:
                    return (i,j)
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
    def fully_determined(self):
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) != 1:
                    return False
        return True

    @property
    def shortcircuited(self):
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) == 0:
                    return (i, j)
        return False
    
    def count_solutions(self):
        print("Counting solutions")
        while len(self.undetermined_cells) > 0:
            print("Shortcircuited", self.shortcircuited)
            # determining the ith cell's value
            cell = self.undetermined_cells[0]
            print(cell)
            for value in self.viable_indices(cell):
                self.assign_cell(cell, value)
    
    @property
    def tree(self):
        undetermined_cells = self.undetermined_cells
        
        tree = Tree([])

        for cell in undetermined_cells:
            self.viable_indices(cell)