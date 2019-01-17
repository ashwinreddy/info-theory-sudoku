import numpy as np
import logging
from grid import Grid, indices_to_options

class Tree(object):
    def __init__(self, cell, grid_copy, children, parent, decision):
        self.cell      = cell
        self.grid_copy = grid_copy
        self.decision  = decision
        self.children  = children
        self.parent    = parent
    
    def __repr__(self):
        return "Tree(cell = {}, decision = {}, children = {})".format(self.cell, self.decision, self.children)

    def max_depth(self):
        if len(self.children) == 0:
            return 0
        elif type(self.children[0]) is not Tree:
            return 1
        else:
            return 1 + max([child.max_depth() for child in self.children])
        
    def find_solutions(self):
        if len(self.children) == 0:
            return Grid(self.grid_copy)
        else:
            return [child.find_solutions() for child in self.children]


def valid_children(grid, cell):
    # print("Finding valid children for \n{}".format(grid))
    # print("for ", cell)
    children = []
    for index in grid.viable_indices(cell):
        grid_copy = Grid(np.copy(grid.grid))
        entry = index + 1
        # print("Assigning cell {} value/entry {}".format(cell, entry))
        grid_copy.assign_cell(cell, entry)
        grid_copy.double_clear()
        # print(grid_copy)
        
        if not grid_copy.shortcircuited:
            if len(grid_copy.undetermined_cells) > 0:
                child_cell = grid_copy.undetermined_cells[0]
                grandchildren = valid_children(Grid(np.copy(grid_copy.grid)), child_cell)
                child = Tree(cell = child_cell, grid_copy = grid_copy.grid, children = grandchildren, parent = grid, decision = entry )
            else:
                child = Tree(cell = None, grid_copy = grid_copy.grid, children = [], parent = grid, decision = entry)
            children.append(child)
        else:
            pass
            # print("This grid did shortcircuit, so {} will not be included".format(entry))
            
            
    return children

# This class will keep track of all the known and unknown cells
class SudokuGrid(Grid):
    def __init__(self, questioner = None, checkpoint_frequency = None, grid = np.ones((9,9,9), dtype='int8')):
        self.grid = grid
        self.num_cells_determined = 0
        self.questioner = questioner
        self.record_checkpoint()
        self.checkpoint_frequency = checkpoint_frequency
        logging.info("Created SudokuGrid. {} cells determined".format(self.num_cells_determined))
    
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
        logging.info("Determining the value of {} with possibilities {}".format(coordinate, options))

        

        # When there are only 2 possibilities, ask which one it is, and return the answer
        if len(options) == 2:
            answer = self.questioner.ask_cdq( (coordinate, "==", options[1] ))
            logging.debug(answer)
            entry = self.assign_cell(coordinate, options[bool(answer)])
            self.double_clear()
            return entry
        elif len(options) == 1:
            entry = self.assign_cell(coordinate, options[0])
            self.double_clear()
            return entry

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
    def fully_determined(self):
        for i in range(9):
            for j in range(9):
                if np.count_nonzero(self[i][j]) != 1:
                    return False
        return True

   
    
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
        if len(self.undetermined_cells) == 0:
            return None

        grid = np.copy(self.grid)
        cell = self.undetermined_cells[0]
        tree = Tree(cell=cell, grid_copy=grid, children=valid_children(Grid(grid), cell), parent=None, decision=None)
        return tree
    

    def deepcopy(self):
        return SudokuGrid(grid=np.copy(self.grid))