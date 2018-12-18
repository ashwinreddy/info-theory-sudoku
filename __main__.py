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

def bool_to_word(proposition):
    return "" if proposition else "not"

with open('config.toml') as f:
    config = toml.loads(f.read())
    logging.basicConfig(level=logging_levels[config['logging_level']])
    ask_user_cell_determining_questions = config['ask_user_cell_determining_questions']
    ask_user_checkpointing_questions    = config['ask_user_checkpointing_questions']
    computer_can_lie                    = config['computer_can_lie']
    assume_lying                        = config['assume_lying']
    checkpoint_frequency                = config['checkpoint_frequency']
    step_mode                           = config['step_mode']
    
    logging.info("This program will {} ask you, the user, for questions that determine the cell's value.".format(bool_to_word(ask_user_cell_determining_questions)))
    logging.info("This program will {} ask you, the user, to validate the grid at checkpoints".format(bool_to_word(ask_user_checkpointing_questions)))
    logging.info("The computer is {} allowed to lie".format(bool_to_word(computer_can_lie)))
    logging.info("This algorithm will {} assume the computer is allowed to lie".format(bool_to_word(assume_lying)))
    logging.info("Checkpointing, if it occcurs, will happen after every {} questions".format(checkpoint_frequency))

test_grid = np.loadtxt("test_grids/2.txt", delimiter=' ', dtype='int8')
questioner = Questioner(ask_user_cell_determining_questions, ask_user_checkpointing_questions, computer_can_lie)
questioner.set_grid(test_grid)
sg = SudokuGrid(questioner)


solve(sg, np.arange(81), assume_lying, checkpoint_frequency, step_mode)