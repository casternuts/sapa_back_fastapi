# 모듈경로 지정


from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from sapaback.llm.lclogic.tools.def_test_tool import agent
from sapaback.llm.lclogic.agents.weather_agent import agents
from sapaback.llm.lclogic.agents.naver_api_agent import agents_naver
from langchain.llms import OpenAI
from sapaback.llm.lclogic.agents.keyword_agent import llm_keyword_chain,keyword_agents
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
import json
import urllib.request
from sapaback.core.config import settings
client_id = settings.naver_client_id
client_secret =settings.naver_client_secret
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

def keyword_agent(query:str):
    template = '''\
    {word}과 연관된 음식 키워드를 분류명 3개와 분류에 속해 있는 음식으로 묶어서 5개씩 한국어로 다 나열해 줘.  \
     술, 향신료 ,음료, 양념장은 제외하고 나열 해줘 \
     result should be JSON in the following format: "[<<분류명>:[음식명,음식명,음식명,음식명,음식명]>,....]"
    '''

    template = '''\
       {word}과 연관된 음식 키워드를 분류명 3개와 분류에 속해 있는 음식으로 묶어서 5개씩 한국어로 다 나열해 줘.  \
        술, 향신료 ,음료, 양념장은 제외하고 나열 해줘.  \
        result must be string in the following format: "<분류명>=[<<value>>,<<value>>,<<value>>] || <분류명>=[<<value>>,<<value>>,<<value>>] || ...."
       '''

    prompt = PromptTemplate(
        input_variables=["word"],
        template=template,
    )
    chain = LLMChain(llm=chat, prompt=prompt)
    result :str = chain.run(query)
    # print(result)
    # print(type(result))
    listcate =  result.split("||")
    nestDic = dict()

    nestDic["startDate"] = "2023-05-01"
    nestDic["endDate"] = "2023-05-30"
    nestDic["timeUnit"] = "date" #date week month
    keywordGroupsList = list()
    for idxitem, item in enumerate(listcate):
        print(item)
        listArr = item.split("=")
        itemlist = list()

        keyString = ""
        valueString = ""
        for idx, val in enumerate(listArr):
            keywordGroupsDict = dict()
            print(idx,val)
            if idx == 1:
                valueString = val.replace("[","").replace("]","").replace('"',"").strip().split(",")
                # print("##########################")
                # print(len(valueString))
                # print("##########################")
                #공백 제거
                for idx, val in enumerate(valueString):
                    valueString[idx] = val.strip()
                    print("테스트: "+valueString[idx])
                keywordGroupsDict["groupName"] = keyString
                keywordGroupsDict["keywords"] = valueString
                keywordGroupsList.append(keywordGroupsDict)

            else:
                keyString =listToString(re.compile('[가-힣]+').findall(val.strip()))
                print("key:" + keyString)
                #keywordGroupsDict["groupName"] = keyString
            # 리스트에 삽입


        #dict 자료형에 그룹 리스트 삽입
        nestDic["keywordGroups"] = keywordGroupsList
        nestDic["device"] = "mo" #pc,mo


        agelist = list(); # 1:1~12,2:13~18,3:19~24,4:25~29 5:30~34,6:35~39,7:40~44,8:45~49,9:50~54,10:55~59,11:60세이상
        agelist.append("1")
        agelist.append("2")
        agelist.append("4")
        agelist.append("5")
        agelist.append("6")
        agelist.append("7")
        nestDic["ages"] = agelist
        nestDic["gender"] = "m" #m 남자 f 여자

        # print(nestDic)

        result_dict = naver_trand(nestDic)
        # for idx, val in enumerate(nestDic.keys()):
        #     print(idx,val)

    json_val = json.dumps(result_dict,ensure_ascii=False)

    print("json_val = %s" % json_val)

    print("json_val type = %s" % type(json_val))
#

    # 문자열을 dict 형식으로 변환하기
    # json_object_to_dict = json.loads(result)
    # print(type(json_object_to_dict))
    # print(json_object_to_dict)
    #
    # print(json_object_to_dict.keys())

    # for item in json_object_to_dict.keys():
    #     print(json_object_to_dict[item].replace("[",""))
    #     print(json_object_to_dict[item].replace("]", ""))
    #     print(json_object_to_dict[item].replace("'", ""))


    return result_dict




   # keyword_agents.run("여름과 관련된 음식 키워드를 한국어로 5개 뽑아줘 형식은 json으로 제공해줘")

def naver_trand(nestDic):
    url = "https://openapi.naver.com/v1/datalab/search"  # json 결과
    # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
    # print(nestDic)
    dict_post =nestDic
    post_data = json.dumps(dict_post, ensure_ascii=False)
    details = urllib.parse.urlencode(dict_post)
    details = details.encode('UTF-8')
    # print("Dictionary Type : ", type(dict_post))
    # print("Dictionary : ", dict_post)
    # print("JSON Type : ", type(post_data))
    # print("JSON : ", post_data)
    request = urllib.request.Request(url, details)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    request.add_header("Content-Type", "application/json")
    response = urllib.request.urlopen(request, data=post_data.encode("utf-8"))
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        # dict로 변환
        # 전체 JSON을 dict type으로 가져옴
        dict = json.loads(response_body.decode('utf-8'))

        # print(type(dict))
        # print(dict)

        # items= dict["items"]
        # for item in items:
        #       #print(item)
        #       print(type(item))
        #       print(item["description"])
        return dict
        #return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)
        return rescode
def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

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











