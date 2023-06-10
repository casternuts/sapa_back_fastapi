from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sapaback.llm.lclogic import lcmain

import json
import urllib.request
from sapaback.core.config import settings
client_id = settings.naver_client_id
client_secret =settings.naver_client_secret

router = APIRouter()

#uvicorn main:app --reload

@router.get("/")
def read_root():
    string =lcmain.custom_agent()
    return {"Hello": string}
@router.get("/2")
def read_root():
    string =lcmain.custom_agent2()
    return {"Hello": string}
@router.get("/navercafe")
def read_root():
    string =lcmain.custom_agent_naver()
    return {"Hello": string}


@router.get("/apitest")
def read_root():
    #string =lcmain.custom_agent_naver()

    encText = urllib.parse.quote("그리팅 한우 우거지탕")  # <- search keyword
    url = "https://openapi.naver.com/v1/search/cafearticle.json?query=" + encText    # json 결과
    # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
    print(url)
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        #print(response_body.decode('utf-8'))
        #jresult = json.dumps(response_body.decode('utf-8'))
        #dict로 변환
        # 전체 JSON을 dict type으로 가져옴
        dict  = json.loads(response_body.decode('utf-8'))

        print(type(dict))
        #print(dict["items"][0]["description"])  # description 정보를 조회
        items= dict["items"]
        for item in items:
              #print(item)
              print(type(item))
              print(item["description"])

        return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)
        return rescode




