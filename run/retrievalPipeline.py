import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.QueryValidationAgent import query_validator
from agents.MainAgent import main_agent
from agents.AnswerValidationAgent import answer_validator
from database.VectorStoreManagement import vector_store_manager
from docs.TraceManager import addToTraceJson

def runRetrievalPipeline(raw_query: str) -> str:

    addToTraceJson(addition= {"raw_query_input": raw_query}, clear= True)

    # reformulate and validate query
    reformulation_dict = query_validator.validateAndReformulateQuery(query= raw_query)

    # move based on the outcome
    ref_query = ''
    match reformulation_dict["status"]:
        case "DENIED":
            addToTraceJson({"tokens used": query_validator.token_usage})

            return reformulation_dict["message"]
        case "DONE":
            addToTraceJson({"tokens used": query_validator.token_usage})

            return reformulation_dict["message"]
        case "PROCEED":
            ref_query = reformulation_dict["message"]

    addToTraceJson({"reformulated query": ref_query})

    # get context chunks
    retrieved_chunks = vector_store_manager.searchQuery(ref_query)

    # get llm answer
    response = main_agent.answerQuery(query= ref_query, context= retrieved_chunks)

    # validate answer
    response_validator_agent: True | str = answer_validator.validateAnswer(question= ref_query, answer= response)

    if response_validator_agent is not True:
        return f"INVALID: {response_validator_agent}"

    addToTraceJson({"final message": response})

    addToTraceJson({"tokens used": query_validator.token_usage + main_agent.token_usage + answer_validator.token_usage})

    return response

# EXAMPLE QUESTIONS INVALID/DONE:
# quantos anos tinha o jogador de Futebol pelé quando ele morreu?
# bom dia, posso fazer algumas perguntas?

# EXAMPLE QUESTIONS VALID(RAG ONLY):
# quais as principais caracteristicas do modelo de rutherford borh?
# o que são piramides ecologicas?
# qual a relação entre calor e temperatura?
# o que é o mutualismo?
# quais são as 3 leis de newton?

# EXAMPLE QUESTION VALID + WEB SEARCH:
# quais os tipos basicos de radiação? DONE
