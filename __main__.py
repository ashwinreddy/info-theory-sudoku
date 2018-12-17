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

# order_of_questioning = list(range(81))
order_of_questioning = []

for i in range(9):
    for j in range(9):
        if (i,j) not in order_of_questioning:
            order_of_questioning.append((i,j))

order_of_questioning = order_of_questioning[::-1]

print(sg)
i = 0
while not sg.completed:
    row, col = order_of_questioning[i]
    sg.determine_cell(row, col, sg.viable_indices(row, col))
    print(sg)
    i += 1

print(sg)
print(interrogator.questions_asked)
