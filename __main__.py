# Determines a sudoku grid by asking the user questions
import numpy as np

from grid import SudokuGrid
from interrogator import Interrogator

import logging
import toml

logging.basicConfig(level=logging.CRITICAL)

with open('config.toml') as f:
    config = toml.loads(f.read())
    logging.debug(config)


test_grid = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
interrogator = Interrogator(config['ask_user_mode'])
interrogator.set_grid(test_grid)
sg = SudokuGrid(interrogator)

def spiral_cw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)

# order_of_questioning = list(range(81))
order_of_questioning = []

for i in range(9):
    for j in range(9):
        if (i,j) not in order_of_questioning:
            order_of_questioning.append((i,j))

order_of_questioning = order_of_questioning

i = 0
while not sg.completed:
    # row, col = np.unravel_index(order_of_questioning[i], grid.shape)
    row, col = order_of_questioning[i]
    sg.determine_cell(row, col, sg.viable_indices(row, col))
    # flat_idx += 1
    i += 1
    # if i > 40:
    
    
    #         break

# print(sg.collapsed_grid)
print(interrogator.questions_asked)