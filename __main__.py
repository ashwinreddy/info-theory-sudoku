# Determines a sudoku grid by asking the user questions
import numpy as np

from grid import SudokuGrid
from interrogator import Interrogator

import logging
import toml

logging.basicConfig(level=logging.DEBUG)

with open('config.toml') as f:
    config = toml.loads(f.read())
    logging.debug(config)


grid = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
interrogator = Interrogator(config['ask_user_mode'])
interrogator.set_grid(grid)
sg = SudokuGrid(interrogator)

order_of_questioning = list(range(81))

cell = 0
while not sg.completed:
    i, j = np.unravel_index(order_of_questioning[cell], grid.shape)
    sg.determine_cell(i, j, sg.viable_indices(i, j))
    # flat_idx += 1
    cell += 1

print(sg)

print(interrogator.questions_asked)