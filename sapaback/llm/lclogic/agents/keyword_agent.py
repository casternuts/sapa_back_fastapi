#
# tools_agent.py
#
# zero-shot react agent that reply questions using available tools
# - Weater
# - Datetime
#
# get the question as a command line argument (a quoted sentence).
# $ py tools_agent.py What about the weather today in Genova, Italy
#
import sys

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

from langchain import LLMChain
from langchain.prompts import PromptTemplate

# import custom tools
from sapaback.llm.lclogic.tools.weather_tool import Weather
from sapaback.llm.lclogic.tools.datetime_tool import Datetime
from sapaback.core.config import settings
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
llm = ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo')

template='''\
Please answer the following questions with precision. \
If you are unable to find the required information after seeking assistance, \
please indicate that you do not know.
'''

prompt = PromptTemplate(input_variables=[], template=template)
search = GoogleSearchAPIWrapper(google_api_key=settings.google_search_api_key,
    google_cse_id=settings.google_search_engine_key)

def top5_results(query):
    return search.results(query, 5)
# create an instance of the custom langchain tool
google = Tool(
    name = "Google Search Snippets",
    description="가장 최신 내용을 구글 검색을 통해 가져온다",
    func=search.run
)
# debug
# print(prompt.format())
# sys.exit()


# Load the tool configs that are needed.
llm_keyword_chain = LLMChain(
    llm=llm,
    prompt=prompt

)

tools = [
    Datetime,
google

]

# Construct the react agent type.
keyword_agents = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print('question: ' + question)

        # run the agent
        keyword_agents.run(question)
    else:
        print('agent that answers questions using Weather and Datetime custom tools')
        print('usage: py tools_agent.py <question sentence>')
        print('example: py tools_agent.py what time is it?')