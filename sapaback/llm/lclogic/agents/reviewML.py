from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import csv
import pandas as pd
from sapaback.llm.lclogic.tools.sentence_ml import tfv_return
import json

def reviewFunction(itemname):

    best_model = joblib.load('C:/Users/Lee/SAPA/sapaback/model_save.joblib')
    greating_review = []
    item_name = []
    good_review = []
    bad_review = []
    ambi_review = []
    all_review = []
    all_review_csv = []

    with open('C:/Users/Lee/SAPA/sapaback/ALL_GREATING_DATA.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        flag = True
        for txt in reader:
            if(flag):
                flag = False
                continue
            item_name.append(txt[6])
            greating_review.append(txt[8])

    the_banchan_review = tfv_return().transform(greating_review) ## 약 20초 소요
    # 후처리
    # 나온 결과를 바탕으로 유의미한 데이터로 추출한다.
    cnt = 0
    for val in the_banchan_review:
        result_zero_or_one = best_model.predict(val).tolist() # 긍정 부정
        probability = np.max(best_model.predict_proba(val))   # 확률
        str_result = "" #csv 파일을 위한 String 결과값

        # 애매한 결과 추출
        if(np.max(probability)*100 >= 50 and np.max(probability)*100 <=79.9):
            str_result = "중립"
            ambi_review.append({"item_name":item_name[cnt],"greating_review":greating_review[cnt],"result":str_result,"probability":probability*100})

        else:
            if(result_zero_or_one[0] == '0'):
                str_result = "부정"
                bad_review.append({"item_name":item_name[cnt],"greating_review":greating_review[cnt],"result":str_result,"probability":probability*100})#부정일때

            else:
                str_result = "긍정"
                good_review.append({"item_name":item_name[cnt],"greating_review":greating_review[cnt],"result":str_result,"probability":probability*100}) #긍정일때

        all_review_csv.append([item_name[cnt],greating_review[cnt], str_result,probability * 100])  # csv 파일 위한 전체값
        all_review.append({"item_name":item_name[cnt],"greating_review":greating_review[cnt],"result":str_result,"probability":probability*100})  #jsonList 위한 전체값
        cnt+=1

    with open('reviewSelect.csv', 'w', newline='', encoding='ANSI') as f:
        makewrite = csv.writer(f)

        for value in all_review_csv:
            makewrite.writerow(value)

    type(all_review)
    print(type(all_review))

    json_data = json.dumps(all_review, ensure_ascii=False)
    print(json_data)
    return json_data