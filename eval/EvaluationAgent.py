import json
from agents.BasicGoogleAIAgent import BasicGoogleGenAIAgent


class EvalAgent(BasicGoogleGenAIAgent):

    def __init__(self):
        super().__init__(
            name="validation agent",
            basic_instructions=
            """
You are a performance judge, you must recieve a QUESTION, a CONTEXT, a CONTEXT FROM WEB SEARCH and a RESPONSE, then you 
must give a score ranging from 0.0 to 10.0 to these indicators:

indicator 1: faithfullness
- this indicator is any value between 10.0 and 0.0 according to how much the RESPONSE is faithfull to the CONTEXT or to the CONTEXT FROM WEB SEARCH
- the more information from the RESPONSE is sourced from the CONTEXT or from the CONTEXT FROM WEB SEARCH, the higher the faithfullness's score is
- the less information from the RESPONSE is sourced from the CONTEXT or from the CONTEXT FROM WEB SEARCH, the lower the faithfullness's score is

indicator 2: content_precision
- this indicator is any value between 10.0 and 0.0 according to how much the CONTEXT or the CONTEXT FROM WEB SEARCH have the information needed to respond the QUESTION
- the more able the CONTEXT or the CONTEXT FROM WEB SEARCH are to answer the QUESTION, the higher the content_precision's score is
- the less able the CONTEXT or the CONTEXT FROM WEB SEARCH are to answer the QUESTION, the lower the content_precision's score is

indicator 3: answer_relevancy
- this indicator is any value between 10.0 and 0.0 according to how much the RESPONSE is relevant to the QUESTION
- the more relevant the RESPONSE is to the QUESTION, the higher the answer_relevancy's score is
- the less relevant the RESPONSE is to the QUESTION, the lower the answer_relevancy's score is

response must be a json format that follows this structure:
{"faithfullness": <faithfullness indicator score>, "content_precision": <content_precision indicator score>, "answer_relevancy": <answer_relevancy indicator score>}

Rules:
- answer ONLY with the json object
- do not write any markdown fences
            """,
            tools=[]
        )
        self.metrics_used = ["faithfullness","content_precision","answer_relevancy"]

    def evaluate(self,
             question: str,
             context:  str,
             response: str
        ) -> dict:

        formated_sample = f"QUESTION: {question}\n\n\n CONTEXT: {context}\n\n\n RESPONSE: {response}"

        response = self.invoke(
            query= formated_sample
        )
        response_content = self.extractResponseContent(response= response)

        response_dict = json.loads(response_content)

        return response_dict

