import logging
import random

operator_to_human_readable = {
    "==": "equal to",
    ">=": "greater than or equal to"
}

def ask_human_question(prompt):
    return input(prompt + " (y/n)") == "yes"

class Questioner(object):
    """
    Routes questions to user or program and answers appropriately (lying if it is allowed to do so)
    """
    def __init__(self, ask_user_cdqs, ask_user_ckpt, can_lie):
        self.ask_user_cdqs = ask_user_cdqs
        self.ask_user_ckpt = ask_user_ckpt
        self.questions_asked = 0
        self.can_lie = can_lie
        self.has_lied_yet = False
        
    def set_grid(self, grid):
        self.grid = grid
    
    def ask_cdq(self, question):
        """
        Routes a cell determining question appropriately
        """

        human_readable_question = "Is the value in {}th cell {} {} ? ".format(question[0], operator_to_human_readable[question[1]], question[2])
        logging.debug(human_readable_question)
        self.questions_asked += 1
        
        return ask_human_question(human_readable_question) if self.ask_user_cdqs else self._answer_question_as_computer(self._ask_computer_cdq(question))

    def ask_if_rewinding_required(self, collapsed_grid):
        human_readable_question = "Does your grid look like this?\n {} \n ? ".format(collapsed_grid)
        logging.debug(human_readable_question)
        self.questions_asked += 1
        return ask_human_question(human_readable_question) if self.ask_user_ckpt else not self._check_grid_matches_solution(collapsed_grid)
    
    def _ask_computer_cdq(self, question):
        """
        Answers a cell determining question using its local solution grid
        """
        question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
        answer = eval(question)
        return answer
    
    def _answer_question_as_computer(self, true_response):
        if self.can_lie and not self.has_lied_yet and self.questions_asked > 10 and random.random() > 0.9:
            self.has_lied_yet = True
            logging.warning("Lying on this question")
            return not true_response
        else:
            return true_response
    
    def _check_grid_matches_solution(self, incomplete_grid):
        for i in range(9):
            for j in range(9):
                if incomplete_grid[i][j] != 0 and incomplete_grid[i][j] != self.grid[i][j]:
                    return False
        return True
    

    def ask_if_lied_on_checkpoint(self):
        pass