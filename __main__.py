# Determines a sudoku grid by asking the user questions
import numpy as np

from grid import SudokuGrid, solve
from interrogator import Interrogator

import logging
import toml

logging_levels = {
    "debug": logging.DEBUG,
    "critical": logging.CRITICAL,
}

with open('config.toml') as f:
    config = toml.loads(f.read())
    logging.basicConfig(level=logging_levels[config['logging_level']])
    logging.debug(config)

test_grid = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
interrogator = Interrogator(config['ask_user_mode'])
interrogator.set_grid(test_grid)
sg = SudokuGrid(interrogator)


solve(sg, list(range(81)))