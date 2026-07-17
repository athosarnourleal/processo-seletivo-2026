from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage

from agents.BasicGoogleAIAgent import BasicGoogleGenAIAgent
from langchain_tavily import TavilySearch

import json
from docs.TraceManager import addToTraceJson

# setup tavily tool
search_tool = TavilySearch(
    max_results= 3,
    include_url= True,
    include_usage=True,
    search_depth= "advanced",
    chunks_per_source= 3
)

class MainAgent(BasicGoogleGenAIAgent):

    def __init__(self):
        super().__init__(
            name="Main Agent",
            basic_instructions=
            """
you are a question answering agent,
your job is to receive a context and a question(query) and answer the question based 
ONLY on the given context and the web search results

#STEPS:
1. check if the CONTEXT or the CONTEXT FROM WEB SEARCH has the information needed to respond the query and construct your answer ONLY from them
2. in cases where you are NOT ABLE answer the query you must:
    - if available, you must call the web search tool
    - if the web search tool is unavailable, you must try to explain why you could not answer the query
3. if possible to answer the query, you must then provide a detailed answer for it.

RULES:
- when possible to answer question, you NEED to include source citations
- you must include Only the answer and the source citations
- do not include markdown fences
- answer and sources must have the same language as the query
            """,
            tools=[search_tool]
        )
        self.numberOfToolResponses = 0

    # noinspection PyTypeChecker
    def formatToolResponse(self, tool_message: ToolMessage) -> AnyMessage:
        """transformar os resultados do tavily search tool para um formato que poderia ser lido e interpretado mais facilmente"""

        raw_tool_content = json.loads(tool_message.content)

        raw_tool_content.update({"reason_for_call":"answer not found in original query"})

        # extract only results
        page_data = raw_tool_content["results"]

        formated_tool_content = "CONTEXT FROM WEB SEARCH:\n"
        for page in page_data:
            formated_data = f"\nweb page title:{page["title"]}\ncontents:\n{page["content"]}\nWEB PAGE SOURCE: {page["url"]}\n\n"
            formated_tool_content+=formated_data

        addToTraceJson({"formatted_tool_message": {"agent that called": self.NAME, "content": formated_tool_content, "usage": raw_tool_content["usage"]}})

        return SystemMessage(content= formated_tool_content)

    def answerQuery(self, query: str, context: str | list):

        response = self.invoke(
            query= query,
            context= f"CONTEXT: {context}"
        )

        response_content = self.extractResponseContent(response= response)
        print("MAIN AGENT:",response_content)

        return response_content

main_agent = MainAgent()
