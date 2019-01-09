import numpy as np
import logging
from grid import Grid, indices_to_options

# retrieved from http://code.activestate.com/recipes/578948-flattening-an-arbitrarily-nested-list-in-python/
def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

class Tree(object):
    def __init__(self, cell, children):
        self.cell      = cell
        self.children  = children
    
    def __repr__(self):
        return "Tree({}, {})".format(self.cell, self.children)
    
    def fill_children(self, sg):
        print("Filling children for {}".format(self))
        children_with_trees = []
        for child in self.children:
            print("Here is a viable index for this node {}: {}".format(self.cell, child))
            sg_copy = sg.deepcopy()
            print("Assigning {} {}".format(self.cell, child))
            sg_copy.assign_cell(self.cell, child)
            next_undetermined_cell = sg_copy.undetermined_cells[0]
            vi = sg_copy.viable_indices(next_undetermined_cell)
            children_with_trees.append(Tree(next_undetermined_cell, vi))
        self.children = children_with_trees
        return self
    
    def find_leaves(self):
        print("Finding leaves for {}".format(self))
        if type(self.children[0]) is not Tree:
            return [self]
        else:
            return [x.find_leaves() for x in self.children]
            

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
        tree = Tree(self.undetermined_cells[0], self.viable_indices(self.undetermined_cells[0]))
        print("The root node: {}".format(tree))
        tree.fill_children(self)
        for child in flatten(tree.find_leaves()):
            child.fill_children(self)
        # return tree.fill_children(self)
    

    def deepcopy(self):
        return SudokuGrid(grid=np.copy(self.grid))