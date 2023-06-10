from typing import Optional, Type

import aiohttp
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel, BaseSettings, Field
from sapaback.core.config import settings
from langchain.chat_models import ChatOpenAI
class GetHuggingFaceModelsToolSchema(BaseModel):

    query_params: Optional[dict] = Field(
        default=None, description="Optional search parameters"
    )

class GetNaverModelsTool(BaseTool, BaseSettings):
    """My custom tool."""

    name: str = "get_naver_cafe_models"
    #https: // openapi.naver.com / v1 / search / cafearticle.xml?query = %EC % A3 % BC % EC % 8


    description: str = """Tool that calls GET on <https: // openapi.naver.com/v1/search/cafearticle.json*> apis. Valid params include "query":"query"""
    args_schema: Type[GetHuggingFaceModelsToolSchema] = GetHuggingFaceModelsToolSchema
    base_url: str = "<https://openapi.naver.com/v1/search/cafearticle.json>"
    client_api_key: str = "ZwyD9fNIOKWYeZN0XsN2"
    client_api_secret_key: str = "b5ktdCu8cc"


    def _run(
        self,
        path: str = "",
        query_params: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Run the tool"""
        #
        return query_params

    # async def _arun(
    #     self,
    #     path: str = "",
    #     query_params: Optional[dict] = None,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> dict:
    #     """Run the tool asynchronously."""
    #
    #     async with aiohttp.ClientSession() as session:
    #         async with session.get(
    #             self.base_url + path, params=query_params, headers=self._headers
    #         ) as response:
    #             return await response.json()

#get_models_tool = GetNaverModelsTool()
# models = get_models_tool.run({"query_params": {"query": "gpt-j"}})
# print(models)

from langchain.agents import initialize_agent, AgentType

tools = [GetNaverModelsTool()] # Add any tools here
llm = ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo',temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)