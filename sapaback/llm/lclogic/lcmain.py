# 모듈경로 지정


from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from sapaback.core.config import settings
from sapaback.llm.lclogic.tools.def_test_tool import agent
from sapaback.llm.lclogic.agents.weather_agent import agents
from sapaback.llm.lclogic.agents.naver_api_agent import agents_naver
from langchain.llms import OpenAI
from sapaback.llm.lclogic.agents.keyword_agent import llm_keyword_chain,keyword_agents
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.utilities import GoogleSearchAPIWrapper

#전역 변수 오픈 ai 생성
chat = OpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo')
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)


def run():
    print("실행")
    return "실행됬음"

def api_cost():
    # chat("1980년대 메탈 음악 5곡 추천해줘.")
    with get_openai_callback() as cb:
        result = chat("1980년대 메탈 음악 5곡 추천해줘.")

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print(cb)
        print(result)


    return result

def custom_agent():
    str = agents.run("please give me the wind speedi and temperature in Milan, Italy")

    return str
def custom_agent_naver():
    str = agents_naver.run("네이버 카페에서 그리팅 한우 우거지탕을 검색하고 'description' 값을 요약 해줘 . 개수는 10개 찾아줘")

    return

def keyword_agent(query):
    template = '''\
    {word}과 연관된 음식 키워드를 분류명 5개와 분류에 속해 있는 음식으로 묶어서 5개씩 한국어로 다 나열해 줘 술은 제외하고.  \
     result should be JSON in the following format: [<<분류명>:[음식명,음식명,음식명,음식명,음식명]>,....]
    '''

    prompt = PromptTemplate(
        input_variables=["word"],
        template=template,
    )
    chain = LLMChain(llm=chat, prompt=prompt)

    print(chain.run("당뇨"))




   # keyword_agents.run("여름과 관련된 음식 키워드를 한국어로 5개 뽑아줘 형식은 json으로 제공해줘")

def summary_agent():
    # template = '''\
    #      {reviews} 를 기사처럼 요약해. 부정적인 의견이 있다면 포함해서 요약해줘.
    #        가장 긍정적인 리뷰는 맨 마지막에 베스트: <내용> 으로 요약 없이 노출 시켜줘.
    #        \
    #
    #     '''

    template = '''\
            {reviews} 를 기사처럼 요약해. 부정적인 의견이 있다면 포함해서 요약해줘. 가장 긍정적인 문구와 가장 부정적인 문구도 하나씩 선정해 도출 해줘.
             result should be JSON in the following format: [<긍정>:<긍정요약>,<부정>:<긍정요약>,<가장 긍정적인>:<가장 긍정적인 문구>,<가장 부정적인>:<가장 부정적인 문구>....]
              \
    
           '''
    #
    prompt = PromptTemplate(
        input_variables=["reviews"],
        template=template,
    )
    chain = LLMChain(llm=chat, prompt=prompt)

    text =  "1.KT 달달쿠폰을 오늘 알아서 부랴부랴 그리팅 질렀어요\
    2. 그리팅  한우우거지탕  외(절약액 17, 035 원 / 실지출 13, 065원 ) 페이코에\
    그리팅 쿠폰 떴길래 달렸어요 저번처럼 사고픈거 빨리 품절될까봐 교육...\
    3.그리팅 에서 구매했던 한우우거지탕 에계후..저녁은 쭈꾸미삼겹살..쌈이 없어서 아쉬운대로 맨김에 싸먹었는데 나름 괜찮네요\
    4.오늘 아침은 <b>그리팅<\\/b>에서 산 <b>한우우거지탕<\\/b> 끓여줍니다 물만 붓고 15분만 끓이면되니 세상편합니다 이러다 진짜 밥하기 싫어지는거 아닌가 몰라요ㅋㅋ"

    print(chain.run(text))











