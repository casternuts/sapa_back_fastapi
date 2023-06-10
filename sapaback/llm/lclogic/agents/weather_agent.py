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
llm = ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo')

template='''\
Please answer the following questions with precision. \
If you are unable to find the required information after seeking assistance, \
please indicate that you do not know.
'''

prompt = PromptTemplate(input_variables=[], template=template)

# debug
# print(prompt.format())
# sys.exit()


# Load the tool configs that are needed.
llm_weather_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

tools = [
    Weather,
    Datetime
]

# Construct the react agent type.
agents = initialize_agent(
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
        agents.run(question)
    else:
        print('agent that answers questions using Weather and Datetime custom tools')
        print('usage: py tools_agent.py <question sentence>')
        print('example: py tools_agent.py what time is it?')