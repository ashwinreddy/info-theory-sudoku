import logging

operator_to_human_readable = {
    "==": "equal to",
    ">=": "greater than or equal to"
}

class Questioner(object):
    def __init__(self, ask_user_mode):
        self.ask_user_mode = ask_user_mode
        self.questions_asked = 0
    
    def set_grid(self, grid):
        self.grid = grid
    
    def ask(self, question):
        human_readable_question = "Is the value in {}th cell {} {} ? ".format(question[0], operator_to_human_readable[question[1]], question[2])

        
        if self.ask_user_mode:
            user_answer = input(human_readable_question)
            answer = user_answer == "yes"
        else:
            question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
            answer = eval(question)
        self.questions_asked += 1
        return answer
    
    def is_rewinding_required(self, collapsed_grid):
        human_readable_question = "Does your grid look like this?\n {} \n ? ".format(collapsed_grid)
        rewindingRequired = input(human_readable_question) == "yes"
        checkForLie = input("Did you just lie on this last question? ") == "yes"
        self.questions_asked += 2
        rewindingRequired = not rewindingRequired ^ checkForLie
        return rewindingRequired
