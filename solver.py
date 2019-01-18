import numpy as np
import logging
from sudoku_grid import SudokuGrid
from grid import Grid

def solve(sg, cells, assume_lying, checkpoint_frequency, checkpointing_method, interactive_mode = False):
    logging.debug("Initiating the solver method on the grid")
    i = 0

    # If we're assuming that the person can lie, then the lie has not yet been caught.
    # If we assume that they won't ever lie, then this is effectively as if we already caught the lie.
    has_lie_been_caught = False if assume_lying else True

    while not sg.completed:
        print("--------------------------------------------")
        logging.info("Questions asked: {}. Cells Determined: {}. Lie Caught: {}".format(sg.questioner.questions_asked, sg.num_cells_determined, has_lie_been_caught))
        
        if sg.num_cells_determined % checkpoint_frequency == 0 and sg.num_cells_determined >= checkpoint_frequency and not has_lie_been_caught:
            logging.info("Time for a checkpoint!")
            if sg.shortcircuited:
                sg.rewind_to_last_checkpoint()
            has_lie_been_caught = checkpoint(sg, checkpointing_method)
            if has_lie_been_caught:
                i -= checkpoint_frequency

        coordinate = np.unravel_index(cells[i], (9,9))
        sg.determine_cell(coordinate, sg.viable_indices(coordinate))
        sg.double_clear()
        
        i += 1
        if interactive_mode:
            input("? ")
        
        solution = sg.collapsed_grid

    print("Algorithm has solved grid, as shown below")
    print(solution, sg.questioner.grid)
    print("{} questions were asked".format(sg.questioner.questions_asked))

def checkpoint(sg, checkpointing_method):
    rewinding_required = sg.questioner.ask_if_rewinding_required(sg.collapsed_grid)
    logging.info("Grid wrong according to Anu: {}".format(rewinding_required))
    should_record_checkpoint = True
    if rewinding_required:
        
        if checkpointing_method == "dyl":
            should_record_checkpoint = sg.questioner.ask_if_lied_on_checkpoint()
        elif checkpointing_method == "repeat":
            rewinding_required_second_answer = sg.questioner.ask_if_rewinding_required(sg.collapsed_grid)

            if rewinding_required_second_answer:
                should_record_checkpoint = False
            else:
                rewinding_required_third_answer = sg.questioner.ask_if_rewinding_required(sg.collapsed_grid)

                num_trues = sum([rewinding_required, rewinding_required_second_answer, rewinding_required_third_answer])

                if num_trues == 2:
                    should_record_checkpoint = False
                elif num_trues == 1:
                    should_record_checkpoint = True


        if should_record_checkpoint:
            sg.record_checkpoint()
        else:
            logging.warn("A LIE HAS BEEN CAUGHT! Let's figure out which of the last {} questions was answered with a lie. Stop checkpointing after this".format(sg.checkpoint_frequency))
            sg.rewind_to_last_checkpoint()
        
        logging.info(sg.collapsed_grid)
        return rewinding_required
    else:
        logging.debug("There haven't been any lies since the last checkpoint. Recording this grid so far as valid, and moving on with the assumption that a lie could show up later.")
        sg.record_checkpoint()
        return False
    
