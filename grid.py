import numpy as np
import logging

def indices_to_options(indices):
    return [idx + 1 for idx in indices]

def solve(sg, cells, assume_lying, checkpoint_frequency, interactive_mode = False):
    logging.debug("Solving grid")
    i = 0

    # If we're assuming that the person can lie, then the lie has not yet been caught.
    # If we assume that they won't ever lie, then this is effectively as if we already caught the lie.
    has_lie_been_caught = False if assume_lying else True
    
    while not sg.completed:
        logging.debug("Questions asked: {}. Cells Determined: {}. Lie Caught: {}".format(sg.questioner.questions_asked, sg.num_cells_determined, has_lie_been_caught))
        if sg.num_cells_determined % checkpoint_frequency == 0 and sg.num_cells_determined >= checkpoint_frequency and not has_lie_been_caught:
            logging.info("Time for a checkpoint!")
            has_lie_been_caught = checkpoint(sg)
            if has_lie_been_caught:
                i -= checkpoint_frequency

        coordinate = np.unravel_index(cells[i], (9,9))
        sg.determine_cell(coordinate, sg.viable_indices(coordinate))
        i += 1
        if interactive_mode:
            input("? ")

    print("Algorithm has solved grid, as shown below")
    print(sg)
    print("{} questions were asked".format(sg.questioner.questions_asked))

def checkpoint(sg):
    rewinding_required = sg.questioner.ask_if_rewinding_required(sg.collapsed_grid)
    logging.info("Grid wrong according to Anu: {}".format(rewinding_required))
    if rewinding_required:    
        lied_on_ckpt = sg.questioner.ask_if_lied_on_checkpoint()
        logging.debug("Lied on checkpoint: {}".format(lied_on_ckpt))
        if lied_on_ckpt:
            sg.record_checkpoint()
        else:
            logging.warn("A LIE HAS BEEN CAUGHT! Let's figure out which of the last {} questions was answered with a lie. Stop checkpointing after this".format(sg.checkpoint_frequency))
            sg.rewind_to_last_checkpoint()
            sg.num_cells_determined -= sg.checkpoint_frequency
        
        logging.info(sg.collapsed_grid)
        return rewinding_required
    else:
        logging.debug("There haven't been any lies since the last checkpoint. Recording this grid so far as valid, and moving on with the assumption that a lie could show up later.")
        sg.record_checkpoint()
        return False
    



# This class will keep track of all the known and unknown cells
class SudokuGrid(object):
    def __init__(self, questioner, checkpoint_frequency):
        logging.debug("Instantiating SudokuGrid")
        # (i, j, k): i is the row, j is the column, k is viability (1 if viable, 0 otherwise)
        self.grid = np.ones((9,9,9), dtype='int8')
        self.questioner = questioner
        self.num_cells_determined = 0
        self.record_checkpoint()
        self.checkpoint_frequency = checkpoint_frequency
    
    def rewind_to_last_checkpoint(self):
        logging.warning("Rewinding back to checkpoint copy")
        self.grid = np.copy(self.ckpt_copy)
    
    def record_checkpoint(self):
        self.ckpt_copy = np.copy(self.grid)
    

    def __getitem__(self, coordinate):
        return self.grid[coordinate]
    
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

    def eliminate_option_for_cell(self, coordinate, entry):
        self[coordinate][entry - 1] = 0    
    
    def assign_cell(self, coordinate, entry):
        """
        Changes the grid's values so that (row, col)'s only viable option is entry.
        Then strikes the value of entry from all neighbors (same row, same column, same 3x3 grid)
        """
        self.num_cells_determined += 1

        row = coordinate[0]
        col = coordinate[1]

        self[coordinate][:entry-1] = 0
        self[coordinate][entry:] = 0

        for j in range(0, 9):
            self.eliminate_option_for_cell((row, j), entry)

        for i in range(0, 9):
            self.eliminate_option_for_cell((i, col), entry)

        for i in range(row - (row % 3), row + 3-(row % 3) ):
            for j in range(col - (col % 3), col + 3-(col % 3) ):
                self.eliminate_option_for_cell((i,j), entry)

        self[coordinate][entry - 1] = 1

        return entry


    def determine_cell(self, coordinate, indices):
        """
        Uses binary search to determine what (row, col)'s value is, given a list of viable indices
        """
        options = indices_to_options(indices)
        logging.debug("Determining the value of {} with possibilities {}".format(coordinate, options))

        if len(options) == 0:
            logging.warning("Rewinding because cell has no options")
            self.rewind_to_last_checkpoint()

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
        # any entry with 1 is a viable index
        return np.where(self[coordinate] == 1)[0]

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
    def collapsed_grid(self):
        grid = [[0 if np.count_nonzero(self[i][j] == 1) > 1 else 1 + np.where(self[i][j] == 1)[0][0] for j in range(9) ] for i in range(9)]
        return np.array(grid)

    # def count_solutions(self):
