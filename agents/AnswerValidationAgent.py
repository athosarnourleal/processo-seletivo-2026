from agents.BasicGoogleAIAgent import BasicGoogleGenAIAgent
from docs.TraceManager import addToTraceJson


class AnswerValidationAgent(BasicGoogleGenAIAgent):

    def __init__(self):
        # 2. answer is in scope (philosophy, sociology, ENEM)
        super().__init__(
            name="validation agent",
            basic_instructions=
            """
            You are a answer validator you must receive a question and a response, and then determine if the answer is "valid" or "invalid", and then write the results

            conditions for a "valid" classification:
            - the response must answer most or all of the QUESTION
            - the response must contain source citations

            conditions for a "invalid" classification:
            - no source citations
            - the response does not answer the query

            RULES:
            - answer must be either: "valid" or "<message for user apologising for fail, and explaining what went wrong>"
            - do not write any markdown fences
            """,
            tools=[]
        )

    def validateAnswer(self, question: str, answer: str):
        """classify the answer for the question as valid or invalid

        args: question, answer

        return: True | content
        """

        response = self.invoke(
            query=f"QUESTION: {question} \n\nANSWER: {answer}",
        )

        trace_dict = {"response": response.content, "output": ""}
        if response.content == "valid":
            trace_dict["output"] = "True"
            addToTraceJson({"answer_validator_output": trace_dict})

            return True
        else:
            trace_dict["output"] = f"INVALID:{trace_dict}"
            addToTraceJson({"answer_validator_output": trace_dict})

            return response.content


answer_validator = AnswerValidationAgent()

