# Determines a sudoku grid by asking the user questions
import numpy as np
from grid import SudokuGrid, Interrogator
import logging
logging.basicConfig(level=logging.CRITICAL)

grid = np.loadtxt("test_grids/1.txt", delimiter=' ', dtype='int8')
interrogator = Interrogator(False)
interrogator.set_grid(grid)
sg = SudokuGrid(interrogator)

# np.unravel_index()
flat_idx = 0

while not sg.completed:
    i, j = np.unravel_index(flat_idx, grid.shape)
    sg.determine_cell(i, j, sg.viable_indices(i, j))
    flat_idx += 1
    # input("?")
print(interrogator.questions_asked)