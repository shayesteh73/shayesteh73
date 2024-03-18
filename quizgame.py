
from Question import Question

question_prompts = ["What is 25+15?\n(a)50\n(b)60\n(c)40\n\n",
                    "What is 45-20?\n(a)25\n(b)35\n(c)15\n\n",
                    "What is 32*3?\n(a)106\n(b)96\n(c)56\n\n"]

question = [Question(question_prompts[0], "c"),
            Question(question_prompts[1], "a"),
            Question(question_prompts[2], "b")]

score = 0
for q in question:
    user_answer = input(q.prompt)
    if user_answer == q.answer:
        score += 1

print("Your score:", score)

  
