import numpy as np
import logging
from backtracking import solve_by_backtracking
from sudoku_grid import SudokuGrid
import sys
# from grid import solve_by_backtracking


def solve(sg, cells, assume_lying, checkpoint_frequency, interactive_mode = False):
    logging.debug("Initiating the solver method on the grid")
    i = 0

    # If we're assuming that the person can lie, then the lie has not yet been caught.
    # If we assume that they won't ever lie, then this is effectively as if we already caught the lie.
    has_lie_been_caught = False if assume_lying else True

    grid_copies_1 = []
    grid_copies_2 = []
    
    while not sg.completed:
        logging.debug("Questions asked: {}. Cells Determined: {}. Lie Caught: {}".format(sg.questioner.questions_asked, sg.num_cells_determined, has_lie_been_caught))
        if sg.num_cells_determined % checkpoint_frequency == 0 and sg.num_cells_determined >= checkpoint_frequency and not has_lie_been_caught:
            logging.info("Time for a checkpoint!")
            if sg.shortcircuited:
                sg.rewind_to_last_checkpoint()
            has_lie_been_caught = checkpoint(sg)
            if has_lie_been_caught:
                i -= checkpoint_frequency

        coordinate = np.unravel_index(cells[i], (9,9))
        sg.determine_cell(coordinate, sg.viable_indices(coordinate))
        
        if sg.num_cells_determined >= 17:
            print("There are more than 17 cells determined. Showing tree")
            # print(sg.tree)
        
        if sg.num_cells_determined == 72:
            print(sg.num_cells_determined)
            print(sg)
            sys.exit()
            # sg.count_solutions()
            # grid_copy1 = np.copy(sg.collapsed_grid)
            # grid_copy2 = np.copy(sg.collapsed_grid)
            # sol_grid_1 = solve_by_backtracking(grid_copy1, mode = "lower")
            # sol_grid_2 = solve_by_backtracking(grid_copy2, mode = "upper")
            # lower = int("".join([str(x) for x in grid_copy1.flatten()]))
            # upper = int("".join([str(x) for x in grid_copy2.flatten()]))
            
            # print("Lower", lower, "\nUpper", upper)
            # # print(grid_copy1, sg.questioner.grid)
            # if np.array_equal(grid_copy1, sg.questioner.grid):
            #     solution = grid_copy1
            #     print("Found it with grid 1!", sg.questioner.questions_asked)
                
            # if np.array_equal(grid_copy2, sg.questioner.grid):
            #     solution = grid_copy2
            #     print("Found it with grid 2!", sg.questioner.questions_asked)
                
            #     # print("----------------------")
            #     # print(grid_copy1, grid_copy2)
            # if np.array_equal(grid_copy1, grid_copy2):
            #     grid_copies_1.append(lower)
            #     grid_copies_2.append(upper)
            #     if len(grid_copies_1) >= 2:
            #         print("Difference between this lower grid and last lower grid", grid_copies_1[-1] - grid_copies_1[-2])
            #         print("difference between this upper grid and last upper grid", grid_copies_2[-1] - grid_copies_2[-2])
            #     solution = grid_copy1
            #     print("Convergent backtracking completed", sg.questioner.questions_asked)
                

        
        i += 1
        if interactive_mode:
            input("? ")
        
        solution = sg.collapsed_grid

    print("Algorithm has solved grid, as shown below")
    print(solution, sg.questioner.grid)
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
        
        logging.info(sg.collapsed_grid)
        return rewinding_required
    else:
        logging.debug("There haven't been any lies since the last checkpoint. Recording this grid so far as valid, and moving on with the assumption that a lie could show up later.")
        sg.record_checkpoint()
        return False
    
