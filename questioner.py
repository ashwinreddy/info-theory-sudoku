import logging

operator_to_human_readable = {
    "==": "equal to",
    ">=": "greater than or equal to"
}

class Questioner(object):
    def __init__(self, ask_user_cell_determining_questions, ask_user_checkpointing_questions, can_lie):
        self.ask_user_cell_determining_questions = ask_user_cell_determining_questions
        self.ask_user_checkpointing_questions = ask_user_checkpointing_questions
        self.questions_asked = 0
        self.can_lie = can_lie
        if self.can_lie:
            logging.debug("Oh boy! I'm allowed to lie!")
    
    def set_grid(self, grid):
        self.grid = grid
    
    def ask(self, question):
        human_readable_question = "Is the value in {}th cell {} {} ? ".format(question[0], operator_to_human_readable[question[1]], question[2])


        if self.ask_user_cell_determining_questions:
            user_answer = input(human_readable_question)
            answer = user_answer == "yes"
        else:
            question = "{} {} {}".format(self.grid[question[0]], question[1], question[2])
            answer = eval(question)
        
        self.questions_asked += 1
        return answer
    
    def is_rewinding_required(self, collapsed_grid):
        human_readable_question = "Does your grid look like this?\n {} \n ? ".format(collapsed_grid)

        rewindingRequired = False
        didLieOnCheckpointQuestion = False

        if self.ask_user_checkpointing_questions:
            rewindingRequired = input(human_readable_question) == "yes"
            didLieOnCheckpointQuestion = input("Did you just lie on this last question? ") == "yes"
        else:
            print("Ask the computer if rewinding is required and if it lied.")
            for i in range(9):
                for j in range(9):
                    if collapsed_grid[i][j] != 0:
                        if collapsed_grid[i][j] != self.grid[i][j]:
                            rewindingRequired = True
            
            if not self.can_lie:
                didLieOnCheckpointQuestion = False

        rewindingRequired = rewindingRequired ^ (not didLieOnCheckpointQuestion)
        self.questions_asked += 2
        return rewindingRequired
