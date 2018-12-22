import logging
import random

operator_to_human_readable = {
    "==": "equal to",
    ">=": "greater than or equal to"
}

def ask_human_question(prompt):
    return input(prompt + " (y/n)") == "yes"

class Questioner(object):
    def __init__(self, ask_user_cdqs, ask_user_ckpt, grid, can_lie):
        self.ask_user_cdqs = ask_user_cdqs
        self.ask_user_ckpt = ask_user_ckpt
        self.questions_asked = 0
        self.can_lie = can_lie
        self.has_lied_yet = False
        self.grid = grid
    
    def _ask_computer_cdq(self, question):
        question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
        answer = eval(question)
        return answer
    
    def ask_cdq(self, question):
        human_readable_question = "Is the value in {}th cell {} {} ? ".format(question[0], operator_to_human_readable[question[1]], question[2])
        logging.info(human_readable_question)

        answer = ask_human_question(human_readable_question) if self.ask_user_cdqs else self._ask_computer_cdq(question)
        
        self.questions_asked += 1
        return self._answer_question_as_computer(answer)[0]
    
    def _answer_question_as_computer(self, true_response):
        if self.can_lie and not self.has_lied_yet and random.random() > 0.7:
            logging.warning("Computer is going to lie on this question")
            self.has_lied_yet = True
            return (not true_response, False)
        else:
            return (true_response, True)
    
    def _ask_computer_checkpointing_question(self, collapsed_grid):
        rewindingRequired = False
        didLieOnCheckpointQuestion = False

        for i in range(9):
            for j in range(9):
                if collapsed_grid[i][j] != 0:
                    if collapsed_grid[i][j] != self.grid[i][j]:
                        rewindingRequired = True

        return rewindingRequired, didLieOnCheckpointQuestion
    
    def is_rewinding_required(self, collapsed_grid):
        human_readable_question = "Does your grid look like this?\n {} \n ? ".format(collapsed_grid)
        logging.info(human_readable_question)

        rewindingRequired = False
        didLieOnCheckpointQuestion = False

        if self.ask_user_ckpt:
            rewindingRequired = input(human_readable_question) == "yes"
            didLieOnCheckpointQuestion = input("Did you just lie on this last question? ") == "yes"
        else:
            rewindingRequired, didLieOnCheckpointQuestion = self._ask_computer_checkpointing_question(collapsed_grid)
        
        if didLieOnCheckpointQuestion:
            rewindingRequired = not rewindingRequired

        logging.info("Rewinding required: "+ str(rewindingRequired))
        
        self.questions_asked += 2
        return rewindingRequired
