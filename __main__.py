# Determines a sudoku grid by asking the user questions
import numpy as np

from grid import SudokuGrid, solve
from questioner import Questioner

import logging
import toml

logging_levels = {
    "debug": logging.DEBUG,
    "critical": logging.CRITICAL,
    "info": logging.INFO,
}

with open('config.toml') as f:
    config = toml.loads(f.read())
    logging.basicConfig(level=logging_levels[config['logging_level']])
    logging.debug(config)

test_grid = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
questioner = Questioner(config['ask_user_mode'])
questioner.set_grid(test_grid)
sg = SudokuGrid(questioner)


solve(sg, np.arange(81), config['assume_lying'], config['checkpoint_frequency'], config['interactive_mode'])