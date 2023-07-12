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
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from glob import glob
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
import requests
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain.agents.agent import BaseSingleActionAgent


#전역 변수 오픈 ai 생성
chat = OpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo')
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

output =[]
def csv_search(query:str):
    FILE_NAME = 'D:\sapaback\sapaback\llm\lclogic\TB_GGG_ansi2.csv'
    loader = CSVLoader(FILE_NAME)
    # Load the documents
    # loader = CSVLoader(file_path=FILE_NAME, csv_args={
    #     'delimiter': ',',
    #     'quotechar': '"',
    #     'fieldnames': ['item_code', 'item_name']
    # },encoding= 'utf8')
    documents = loader.load()

    # text 정제
    for page in documents:
        text = page.page_content
        text = re.sub('\n', ' ', text)  # Replace newline characters with a space
        text = re.sub('\t', ' ', text)  # Replace tab characters with a space
        text = re.sub(' +', ' ', text)  # Reduce multiple spaces to single
        output.append(text)

    print(output)
    #
    doc_chunks = []
    #
    for line in output:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 최대 청크 길이
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # 텍스트를 청크로 분할하는 데 사용되는 문자 목록
            chunk_overlap=0,  # 인접한 청크 간에 중복되는 문자 수
        )
        chunks = text_splitter.split_text(line)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": i, "source": FILE_NAME}
            )
            doc_chunks.append(doc)

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPEN_API_KEY)
    index = Chroma.from_documents(doc_chunks, embeddings)
    #
    system_template = """To answer the question at the end, use the following context. If you don't know the answer, just say you don't know and don't try to make up an answer.    
    for example
    탕은 국,찌개
    면은 파스타,라면,소면,비빔면
        
    you only answer in Korean

    {summaries}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, temperature=0.1),
        chain_type="stuff",
        retriever=index.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        # reduce_k_below_max_tokens=True
    )

    result = bk_chain({"question": "'"+query+"'"+' 과 같은 분류거나 연관된 음식 항목을 code와 함께 나열해줘 itemname 중복된건 빼고'+query+ '중 어떤 항목과 연관된건지 reason도 명시해줘 형식은 [{"code":<<code>>,"itemname":<<itemname>>,"reason":<<reason>>},{"code":<<code>>,"itemname":<<itemname>>,"reason":<<reason>>}....]'})

    print(f"질문 : {result['question']}")
    print()
    print(f"답변 : {result['answer']}")
    # index_creator = VectorstoreIndexCreator()
    # docsearch = index_creator.from_loaders([loader])
    # #
    # chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=settings.OPEN_API_KEY, model_name='gpt-3.5-turbo'), chain_type="stuff",
    #                                     retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    # # data = loader.load()
    # # print(data)
    #
    # # # Pass a query to the chain
    # query = "바베큐와 비슷한 항목을 나열해줘"
    # response = chain({"question": query})
    # print(response['result'])
    dict = json.loads(result['answer'])
    type(dict)
    print(type(dict))
    for item in dict:
        print(item)
        print(item['code'])
        print(item['itemname'])
        print(type(item))
    json_data = json.dumps(dict,ensure_ascii=False)
    return json_data

import csv
output_review =[]
def review_search(query:str):
    FILE_NAME = 'D:\sapaback\sapaback\llm\lclogic\GreatingReview.csv'
    review_list =[]
    review_list_csv = []
    with open(FILE_NAME, "r") as f:
        reader = csv.DictReader(f, delimiter=",", quotechar="\"")
        data_list = [row for row in reader]
        print(type(data_list))
        print(len(data_list))
        for item in data_list:
            if item['itemname'] == query:
                review_list_csv.append(item)
                #print(item)
                #청크 전처리
                contents = item['contents']
                contents = re.sub('\n', ' ', contents)  # Replace newline characters with a space
                contents = re.sub('\t', ' ', contents)  # Replace tab characters with a space
                contents = re.sub(' +', ' ', contents)  # Reduce multiple spaces to single
                item['contents'] =contents
                itemname = item['itemname']
                itemname = re.sub('\n', ' ', itemname)  # Replace newline characters with a space
                itemname = re.sub('\t', ' ', itemname)  # Replace tab characters with a space
                itemname = re.sub(' +', ' ', itemname)  # Reduce multiple spaces to single
                item['itemname'] = itemname
                review_list.append('itemname: '+itemname+' '+'contents: '+contents)

    # 리스트를 CSV 형식으로 변환.
    with open("output.csv", "w", newline="") as f:
        fieldnames=['itemname','contents']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_list_csv)
    FILE_NAME = "output.csv"
    loader = CSVLoader(FILE_NAME)
    #
    documents = loader.load()
    output=[]
    for page in documents:
        text = page.page_content
        #불러온 csv 파일을 document형식으로 바꾼다.
        output.append(text)
    #
    #
    doc_chunks = []
    #
    for line in output:
        print('line: '+str(line))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 최대 청크 길이
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # 텍스트를 청크로 분할하는 데 사용되는 문자 목록
            chunk_overlap=0,  # 인접한 청크 간에 중복되는 문자 수
        )
        chunks = text_splitter.split_text(line)
        #print('chunks: '+chunks)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": i, "source": FILE_NAME}
            )
            doc_chunks.append(doc)

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPEN_API_KEY)
    index = Chroma.from_documents(doc_chunks, embeddings)
    #
    system_template = """To answer the question at the end, use the following context.
    If you don't know the answer, just say you don't know and don't try to make up an answer.
    You should only mention the marketing insight that are similar to or relevant to the provided contents.       
       you only answer in Korean
       

       {summaries}
       """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(openai_api_key=settings.OPEN_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=index.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        # reduce_k_below_max_tokens=True
    )

    result = bk_chain({"question": query + '제품의 contents 내용을 이용해 장점 또는 단점을 분석하고 제품을 어떤 관점으로 판매하면 좋을지 얘기해줘 '})
    #result = bk_chain({"question": query + '제품의 contents 내용을 이용해 contents의 전체적인 요약과 마케팅 인사이트를 제시해 '})

    print(f"질문 : {result['question']}")
    print()
    print(f"답변 : {result['answer']}")





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
       {word}과 연관된 음식 키워드를 '분류명' 3개와 분류에 속해 있는 음식으로 묶어서 5개씩 한국어로 다 나열해 줘.  \
        술, 향신료 ,음료, 양념장은 제외하고 나열 해줘.  \
        for example, 
        해산물=[생선,조개,굴,새우,꽃게]
        찌개=[김치찌개,된장찌개,부대찌개,짜글이,마라탕찌개]
        If you don't know the answer, just say you don't know and don't try to make up an answer. \
        result must be string in the following format: "<분류명>=[<<value>>,<<value>>,<<value>>] || <분류명>=[<<value>>,<<value>>,<<value>>] || ...."
       '''

    template = '''\
           {word}과 연관된 음식 키워드를 '음식 분류명' 3개 이하로 뽑고,  음식 분류에 속해 있는 음식명으로 묶어서 최소 1개 이상씩 나열해 줘.   \
            술, 양념장은 제외하고 나열 해줘.  \
            for example, 
            해산물=[생선,조개,굴,새우,꽃게] || 찌개=[김치찌개,된장찌개,부대찌개,짜글이,마라탕찌개] || ..... 
            If you don't know the answer, just say you don't know and don't try to make up an answer. and result count is lower then 5 \
            result must be string in the following format: "<음식 분류명>=[<<value>>,<<value>>,<<value>>] || <음식 분류명>=[<<value>>,<<value>>,<<value>>] || ...."
           '''

    prompt = PromptTemplate(
        input_variables=["word"],
        template=template,
    )
    chain = LLMChain(llm=chat, prompt=prompt)
    result :str = chain.run(query)
    print('###########')
    print('result1: '+result)
    result = re.sub('\n', '||', result)
    print('result2: '+result)
    print('###########')
    # print(type(result))
    listcate =  result.split("||")
    nestDic = dict()
    print('listcate',listcate)
    print('listcate 크기', len(listcate))
    nestDic["startDate"] = "2023-05-01"
    nestDic["endDate"] = "2023-05-30"
    nestDic["timeUnit"] = "date" #date week month
    keywordGroupsList = list()
    for idxitem, item in enumerate(listcate):
        print('################@@@@@')
        print("idxitem:"+str(idxitem)+'번째, '+ item)
        print('################@@@@@')
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

                keywordGroupsDict["groupName"] = keyString
                keywordGroupsDict["keywords"] = valueString
                print("keyString: " +keyString)
                print("keywords: "  ,valueString)
                keywordGroupsList.append(keywordGroupsDict)

            else:
                 keyString =listToString(re.compile('[가-힣]+').findall(val.strip()))
                 print("key:" + keyString)



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
    print(post_data)
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
    print(url,details)
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











