from dotenv import load_dotenv
from agents.BasicGoogleAIAgent import BasicGoogleGenAIAgent
import json

from docs.TraceManager import addToTraceJson

load_dotenv()

class QueryValidationAgent(BasicGoogleGenAIAgent):

    def __init__(self):
        super().__init__(
            name="validation agent",
            basic_instructions=
            """
You are a validation and refactoring agent, your function is 
to receive the user query and return a json object with the classification:
# STEPS:
1. check if query is intelligible
2. check if query is in scope(science, physics, chemistry, biology, ENEM)
3. if the query is not intelligible, is not in scope or if there's cursing or offensive
language within the query then you should classify it as "DENIED" and respond with, precisely: 
    {"status": "DENIED", "message": <brief reasoning for why it was denied>}
4. if the query is not "DENIED" and it contains only greetings or apologies, you must classify it as "DONE" and respond with, strictly:
    {"status": "DONE", "message": <brief response to the query>}
5. in cases where the query is not classified as "DENIED" nor as "DONE", the response must strictly follow the following format:
    {"status": "PROCEED", "message": <reformulated query>}
the reformulated query must follow these requirements:
    - any ambiguity present in the original query must be clarified
    - if there is any writing mistakes, they should be corrected
    - Keep the reformulated query concise
    - Preserve the original language of the user
    
# RULES:
- respond with ONLY a json object containing two fields:
{"status": <classification("DENIED", "DONE" or "PROCEED")>, "message": <reformulated query, explanation for denial or basic ai answer to query>}
- NEVER write any text outside the json
- do not write any markdown fences
            """,
            tools=[]
        )

    def validateAndReformulateQuery(self, query: str) -> dict | None:
        response = self.invoke(
            query= f"QUERY: {query}\n\n"
        )
        response_content = self.extractResponseContent(response)

        response_dict = json.loads(response_content)

        addToTraceJson({"query_validation": response_dict})

        return response_dict

query_validator = QueryValidationAgent()

# EXEMPLOS USADOS NOS TESTES:
# Bom dia! Como você está? --> {'status': 'FINISHED', 'message': 'Bom dia! Estou bem, obrigado por perguntar. Como posso ajudar?'}
# Olá! Agora abgfe? enauldei embti --> {'status': 'DENIED', 'message': 'A query não é inteligível.'}
# quantos anos tem o jogador de futebol Pelé? --> {'status': 'DENIED', 'message': 'A consulta está fora do escopo de filosofia, sociologia ou ENEM.'}
# quais as principais coisas digas pelo filosofo aristoteles e por que? -> {'status': 'NEXT', 'message': 'Quais são as principais ideias de Aristóteles e seus fundamentos?'}