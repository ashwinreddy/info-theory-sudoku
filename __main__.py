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


def main():

    with open('config.toml') as f:
        config                    = toml.loads(f.read())
        logging.basicConfig(level =logging_levels[config['logging']['logging_level']])
        logging.info(config)
        ask_user_cdqs             = config['questioning']['ask_user_cdqs']
        ask_user_ckpt             = config['questioning']['ask_user_ckpt']
        computer_can_lie          = config['questioning']['computer_can_lie']
        assume_lying              = config['algorithm']['assume_lying']
        checkpoint_frequency      = config['algorithm']['checkpoint_frequency']
        step_mode                 = config['ui']['step_mode']    


    test_grid  = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
    questioner = Questioner(ask_user_cdqs, ask_user_ckpt, computer_can_lie)
    questioner.set_grid(test_grid)
    sg         = SudokuGrid(questioner, checkpoint_frequency)
    solve(sg, np.arange(81), assume_lying, checkpoint_frequency, step_mode)

if __name__ == "__main__":
    main()
