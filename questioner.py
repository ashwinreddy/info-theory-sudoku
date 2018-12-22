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
        logging.info(human_readable_question)

        answer = ask_human_question(human_readable_question) if self.ask_user_cdqs else self._ask_computer_cdq(question)
        
        self.questions_asked += 1
        return self._answer_question_as_computer(answer)[0]
    
    def _ask_computer_cdq(self, question):
        """
        Answers a cell determining question using its local solution grid
        """
        question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
        answer = eval(question)
        return answer
    
    def _answer_question_as_computer(self, true_response):
        if self.can_lie and not self.has_lied_yet and random.random() > 0.7:
            logging.warning("Computer is going to lie on this question")
            self.has_lied_yet = True
            return (not true_response, False)
        else:
            return (true_response, True)
    
    def _check_grid_matches_solution(self, incomplete_grid):
        for i in range(9):
            for j in range(9):
                if incomplete_grid[i][j] != 0 and incomplete_grid[i][j] != self.grid[i][j]:
                    return False
        return True
    
    def is_rewinding_required(self, collapsed_grid):
        human_readable_question = "Does your grid look like this?\n {} \n ? ".format(collapsed_grid)
        logging.info(human_readable_question)

        rewindingRequired = False

        if self.ask_user_ckpt:
            rewindingRequired = ask_human_question(human_readable_question)
            # didLieOnCheckpointQuestion = ask_human_question("Did you just lie on this last question? ")
        else:
            rewindingRequired = self._check_grid_matches_solution(collapsed_grid)

        
        

        if didLieOnCheckpointQuestion:
            rewindingRequired = not rewindingRequired

        logging.info("Rewinding required: "+ str(rewindingRequired))
        
        self.questions_asked += 2
        return rewindingRequired
