import os
from dotenv import load_dotenv

load_dotenv()

from langchain.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class BasicGoogleGenAIAgent:

    def __init__(self,
        name: str,
        basic_instructions: str,
        tools: list,
        model: str= "gemini-3.5-flash"
    ) -> None:

        self.NAME = name
        self.SYSTEM_PROMPT = SystemMessage(content=basic_instructions)

        self.llm = ChatGoogleGenerativeAI(
            model= model,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.token_usage = 0

        if len(tools) > 0:
            self.llm_with_tools = self.llm.bind_tools(tools=tools)
            self.tools = {t.name : t for t in tools}
        else:
            self.llm_with_tools = None

    def formatToolResponse(self, tool_message: ToolMessage) -> AnyMessage:
        """formatação que pode ser customizada pelas classes filho"""
        return tool_message

    @staticmethod
    def extractResponseContent(response: AIMessage):
        content = response.content
        if str(content)[:1] == "[":
            # is a complex response(list[dict[Any, Any])
            content_dict = content[0]

            text = content_dict["text"]

            return text
        else:
            # is a simple response(str)
            return response.content

    def invoke(self,
        query: str | None = None,
        context: list | str | None = None,
        messages: list | None = None
    ) -> AIMessage:
        """invoke llm either as a simple text-only call, or as a call based on tool responses"""

        # gather messages(when needed)
        if messages is None:
            messages: list = [self.SYSTEM_PROMPT]
            if context is str:
                messages.append(SystemMessage(content=str(context)))
            elif context is not None:
                for message in context:
                    messages.append(message)
            messages.append(HumanMessage(content=query))

        # text only invoke
        if self.llm_with_tools is None:
            text_only_response = self.llm.invoke(messages)
            self.token_usage += text_only_response.usage_metadata["total_tokens"]
            return text_only_response

        # get answer with tools available
        initial_llm_response = self.llm_with_tools.invoke(messages)
        self.token_usage += initial_llm_response.usage_metadata["total_tokens"]

        if len(initial_llm_response.tool_calls) > 0:
            # execute tools
            for i, call in enumerate(initial_llm_response.tool_calls):
                tool_response: ToolMessage = self.tools[call["name"]].invoke(call)

                messages.append(self.formatToolResponse(tool_response))

            final_llm_response = self.llm.invoke(messages)
            self.token_usage += final_llm_response.usage_metadata["total_tokens"]

            return final_llm_response

        return initial_llm_response


