

#
# weather_tool.py
# builds a langchain tool that incapsules a custom function
# that retrieve weather forecasts data
#
import json
from typing import List
from langchain.agents import Tool
import urllib.request
from sapaback.core.config import settings
client_id = settings.naver_client_id
client_secret =settings.naver_client_secret


#
# weather_data_retriever
# is an example of a custom python function
# that takes a list of custom arguments and returns a text (or in general any data structure)
#
def data_retriever(query: str = None, display: str = None) -> str:


    display = urllib.parse.quote(display)
    encText = urllib.parse.quote(query)  # <- search keyword
    url = "https://openapi.naver.com/v1/search/cafearticle.json?query=" + encText + "&display="+display+"&start=1&sort=sim" # json 결과
    # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
    print(url)
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        print(response_body.decode('utf-8'))
        return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)
        return rescode

        # 전체 JSON을 dict type으로 가져옴
        dict = json.loads(response_body.decode('utf-8'))
    return dict


def naver_trend_search(json_request: str) -> str:


    '''
    Takes a JSON dictionary as input in the form:
        { "when":"<time>", "where":"<location>" }

    Example:
        { "when":"today", "where":"Genova, Italy" }

    Args:
        request (str): The JSON dictionary input string.

    Returns:
        The weather data for the specified location and time.
    '''
    print("naver_trend_search 호출:" + json_request)
    arguments = json.loads(json_request)
    query = arguments["query"]
    display = arguments["display"]
    print("naver_trend_search 호출 query:" + query)
    print("naver_trend_search 호출 displaycount:" + display)
    required_data = arguments["required_data"]
    return data_retriever(query=query, display=display, required_data=required_data)


#
# instantiate the langchain tool.
# The tool description instructs the LLM to pass data using a JSON.
# Note the "{{" and "}}": this double quotation is needed
# to avoid a runt-time error triggered by the agent instatiation.
#
name = "naver_trend_search"
request_format = '{{"startDate":"<startDate>","endDate":"<endDate>","timeUnit":"<timeUnit>","required_data":["variable_name"]}}'
description = f'''
Helps to retrieve naver trend search result.
Input should be JSON in the following format: {request_format}
'''

# create an instance of the custom langchain tool
naver_trend_search_tool = Tool(name=name, func=naver_trend_search, description=description)


if __name__ == '__main__':
    print(naver_trend_search(where='Genova, Italy', when='today'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    print(data_retriever('{ "when":"today", "where":"Genova, Italy" }'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    # print the Weather tool
    print(naver_trend_search_tool)