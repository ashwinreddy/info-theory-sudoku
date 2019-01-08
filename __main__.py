# Determines a sudoku grid by asking the user questions
import numpy as np

from grid import SudokuGrid
from solver import solve
from questioner import Questioner

import logging
import toml

def main():
    with open('config.toml') as f:
        config                    = toml.loads(f.read())
        logging.basicConfig(level = logging.DEBUG)
        logging.info(config)
        ask_user_cdqs             = config['questioning']['ask_user_cdqs']
        ask_user_ckpt             = config['questioning']['ask_user_ckpt']
        computer_can_lie          = config['questioning']['computer_can_lie']
        assume_lying              = config['algorithm']['assume_lying']
        checkpoint_frequency      = config['algorithm']['checkpoint_frequency']
        step_mode                 = config['ui']['step_mode']    
        test_grid_fname           = config['algorithm']['test_grid']

    test_grid   = np.loadtxt(test_grid_fname, delimiter=' ', dtype='int8')
    questioner  = Questioner(ask_user_cdqs, ask_user_ckpt, computer_can_lie)
    questioner.set_grid(test_grid)
    sg          = SudokuGrid(questioner, checkpoint_frequency)
    questioning = np.arange(81)
    solve(sg, questioning, assume_lying, checkpoint_frequency, step_mode)

if __name__ == "__main__":
    main()
