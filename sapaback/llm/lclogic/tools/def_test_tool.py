# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from sapaback.core.config import settings
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

llm = ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo',temperature=0)


#search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [

]
class CalculatorInput(BaseModel):
    question: str = Field()


class grTestInput(BaseModel):
    question: str = Field()
#
#
# tools.append(
#     Tool.from_function(
#         func=llm_math_chain.run,
#         name="Calculator",
#         description="useful for when you need to answer questions about math",
#         args_schema=CalculatorInput
#         # coroutine= ... <- you can specify an async method if desired as well
#     )
#
# )
# tools.append(
#     Tool.from_function(
#         func = lambda x: "'Greating Is You' by Greating. Greating age is 3 Years Old",
#         name="GreatingCustom",
#         description="useful for when you need to answer questions about Greating",
#
#         # coroutine= ... <- you can specify an async method if desired as well
#     )
#
# )
# 커스텀된 그리팅 툴
class CustomGreatingTool(BaseTool):
    name = "GreatingCustom"
    description = "useful for when you need to answer questions about Greating"
    args_schema: Type[BaseModel] = grTestInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        print("tools:str#############"+ query)

        return "'Greating은 현대그린푸드 서비스 입니다'. Greating은 올해로 3년 차 입니다."

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        print("tools:cal#############" + query)
        return llm_math_chain.run(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")


class SearchSchema(BaseModel):
    query: str = Field(description="should be a item name")
    #engine: str = Field(description="should be a search engine")
    #gl: str = Field(description="should be a item name")
   # hl: str = Field(description="should be a item category")


class CustomSearchdivTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"
    args_schema: Type[SearchSchema] = SearchSchema

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        #search_wrapper = SerpAPIWrapper(params={"engine": engine, "gl": gl, "hl": hl})
        print("tools:cal#############" + query)
        return query

    async def _arun(self, query: str, engine: str = "google", gl: str = "us", hl: str = "en",
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

tools = [CustomGreatingTool() ,CustomCalculatorTool(),CustomSearchdivTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)