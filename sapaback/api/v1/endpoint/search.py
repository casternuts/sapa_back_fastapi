from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sapaback.llm.lclogic import lcmain
from pydantic import BaseModel

import json
import urllib.request
from sapaback.core.config import settings
client_id = settings.naver_client_id
client_secret =settings.naver_client_secret

router = APIRouter()
class WordModel(BaseModel):
    name: str



@router.get("/navercafe")
def naver_cafe():
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


@router.get("/navertrend")
def naver_cafe():
    #string =lcmain.custom_agent_naver()

    encText = urllib.parse.quote("그리팅 한우 우거지탕")  # <- search keyword
    url = "https://openapi.naver.com/v1/datalab/search"    # json 결과
    # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
    print(url)
    dict_post = {
                    "startDate": "2017-01-01",
                    "endDate": "2017-04-30",
                    "timeUnit": "month",
                    "keywordGroups": [
                        {
                            "groupName": "한글",
                            "keywords": [
                                         "한글",
                                         "korean"
                                ]
                        },
                        {
                          "groupName": "영어",
                          "keywords": [
                            "영어",
                            "english"
                          ]
                        }
              ],
              "device": "pc",
              "ages": [
                "1",
                "2"
              ],
              "gender": "f"
    }
    post_data =json.dumps(dict_post,ensure_ascii=False)
    details = urllib.parse.urlencode(dict_post)
    details = details.encode('UTF-8')
    print("Dictionary Type : ", type(dict_post))
    print("Dictionary : ", dict_post)
    print("JSON Type : ", type(post_data))
    print("JSON : ", post_data)
    request = urllib.request.Request(url,details)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    request.add_header("Content-Type", "application/json")
    response = urllib.request.urlopen(request, data=post_data.encode("utf-8"))
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        #dict로 변환
        # 전체 JSON을 dict type으로 가져옴
        dict  = json.loads(response_body.decode('utf-8'))

        print(type(dict))
        print(dict)

        # items= dict["items"]
        # for item in items:
        #       #print(item)
        #       print(type(item))
        #       print(item["description"])

        return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)
        return rescode


@router.get("/keyword/{word}")
def get_keyword(word:str):

    print(word)
    #gpt를 이용해서 키워드를 뽑는다.
    lcmain.keyword_agent(word)

