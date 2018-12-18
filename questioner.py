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

        self.questions_asked += 1
        if self.ask_user_mode:
            user_answer = input(human_readable_question)
            return user_answer == "yes"
        else:
            question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
            return eval(question)