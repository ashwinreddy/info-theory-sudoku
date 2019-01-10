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
    def __init__(self, cell, grid_copy, children, parent):
        self.cell      = cell
        self.grid_copy = grid_copy
        self.children  = children
        self.parent    = parent
    
    def __repr__(self):
        return "Tree({}, {})".format(self.cell, self.children)
    
    def fill_children(self):
        print("Filling children for {}".format(self))
        children_with_trees = []
        if self.parent is not None:
            sg = self.parent.grid_copy
        else:
            sg = self.grid_copy
        
        print("Here is the reference grid which all these children I am about to fill will use (initially)", sg)
        
        for child in self.children:
            print("Start of loop. Now considering hypothetically choosing {} for the cell {}".format(child + 1, self.cell))
            sg_copy = sg.deepcopy()
            print("Assigning {} {}".format(self.cell, child + 1))
            sg_copy.assign_cell(self.cell, child + 1)
            print("Here is this child {} 's unique grid".format(child + 1), sg_copy)

            if not sg_copy.shortcircuited:
                next_undetermined_cell = sg_copy.undetermined_cells[0]
                print("Next undetermined cell ", next_undetermined_cell)
                vi = sg_copy.viable_indices(next_undetermined_cell)

                print(sg_copy)
                print("Adding a tree for {} with indices {}".format(next_undetermined_cell, vi))
                new_tree = Tree(next_undetermined_cell, sg_copy, vi, self)
                print("I am adding a new tree", new_tree)
                print("This new tree has a grid copy", new_tree.grid_copy)
                children_with_trees.append(new_tree)
            else:
                print("Shortcircuited. Therefore, eliminating option {} for cell {}".format(child+1, self.cell))
                sg_copy.eliminate_option_for_cell(self.cell, child + 1)
        
        self.children = children_with_trees
        return self
    
    def find_leaves(self):
        # print("Finding leaves for {}".format(self))
        if len(self.children) == 0 or type(self.children[0]) is not Tree:
            return [self]
        else:
            return [x.find_leaves() for x in self.children]
        
    def max_depth(self):
        if type(self.children[0]) is not Tree:
            return 1
        else:
            return 1 + max([child.max_depth() for child in self.children])
            

# This class will keep track of all the known and unknown cells
class SudokuGrid(Grid):
    def __init__(self, questioner = None, checkpoint_frequency = None, grid = np.ones((9,9,9), dtype='int8')):
        super(SudokuGrid, self).__init__(grid)
        self.questioner = questioner
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
        # print(self)
        # print(self.viable_indices(self.undetermined_cells[0]))
        logging.debug("Creating a tree, rooted with the first undetermined cell {}".format(self.undetermined_cells[0]))
        tree = Tree(self.undetermined_cells[0], SudokuGrid(grid=np.copy(self.grid)), self.viable_indices(self.undetermined_cells[0]), None)
        logging.debug("Filling children")
        tree.fill_children()
        print("Just filled the children for the first time. Tree has depth of ", tree.max_depth())
        
        def unfinished_leaves():
            leaves = flatten(tree.find_leaves())
            print("This is an exhaustive list of leaves", leaves, "with ", len(leaves), " leaves")
            unfinished_leaves = []
            for leaf in leaves:
                print("Checking if leaf {} creates a complete solution".format(leaf))
                print("You can check if the leaf creates a complete soln for yourself below")
                print(leaf.grid_copy)
                
                if not leaf.grid_copy.completed and not leaf.grid_copy.shortcircuited:
                    print("This leaf does not create a complete soln")
                    unfinished_leaves.append(leaf)
            print("END OF UNFINISHED LEAVES")
            return unfinished_leaves
        
        ul = unfinished_leaves()

        while len(ul) != 0:
            print("How deep the tree is", tree.max_depth())
            print("There are leaves to expand:", ul)

            for child in ul:
                print("Filling in the children for ", child)
                child.fill_children()

        return tree
        # return tree.fill_children(self)
    

    def deepcopy(self):
        return SudokuGrid(grid=np.copy(self.grid))