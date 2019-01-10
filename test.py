import numpy as np
import logging
from sudoku_grid import SudokuGrid
from solver import solve
from questioner import Questioner

logging.basicConfig(level = logging.DEBUG)


test_grid   = np.loadtxt("test_grids/1.txt", delimiter=' ', dtype='int8')
questioner  = Questioner(False, False, False)
questioner.set_grid(test_grid)
sg          = SudokuGrid(questioner, 7)
questioning = np.arange(81)

for i in range(0, 70):
    i = np.unravel_index(i, (9,9))
    sg.determine_cell(i, sg.viable_indices(i))

print(sg)
print(sg.tree)