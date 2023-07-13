from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import csv
score = []
text = []

def tfv_return():
    #크롤링으로 뽑아낸 데이터 가져오기
    with open('C:/Users/Lee/SAPA/sapaback/the_banchan_all_review_real_final.csv','rt',encoding='utf-8') as f:
        reader = csv.reader(f)
        flag = True # 1번줄은 제목이므로 스킵

        for txt in reader:
            if (flag):
                flag= False # 1번출일시 flag toggle
                continue
            score.append(txt[0]) # Labeling 된 평점 ( 0 , 1 )
            text.append(txt[2])  # 상품평


    scores = np.array(score) # 혹시 모르니 넘피리스트로 저장
    texts = np.array(text)

    # 227 # train 데이터와 test 데이터 나누기 80 : 20
    train_x, test_x, train_y, test_y = train_test_split(texts, scores, test_size=0.2, random_state=0)  # train 데이터와 test 데이터 나누기 80 : 20
    twitter = Okt()  # 토큰화 객체 생성

    tfv = TfidfVectorizer(tokenizer=twitter.morphs, ngram_range=(1, 2), min_df=1, max_df=0.9)

    # 벡터화 시킨다. 토큰화는 okt의 norphs를 사용한다. KonlPy 태그
    # ngram_range 순서가 있는 진 부분집합 (자기자신제외) 리스트로 뽑음
    # ex) I like Greating -> ngram_range(1,3) 3단어 묶음까지 만들기
    # I, like, Greating, I Like, Like Greating, I Like Greating & 67
    print("토큰화 시작")
    tfv.fit(train_x)
    print("토큰화 끝")

    return tfv

"""
    print(tfv.fit(train_x))
    print(type(tfv.fit(train_x)))
    # 학습시키기
    tfv_train_x = tfv.transform(train_x)
    # 학습데이터 벡터로 만들기

    clf = LogisticRegression(random_state=0)
    # 머신러닝 로지스틱 회귀 사용
    # odds ratio = p/(1-p) -> p는 예측값이 실제 값과 동일하게 나올 확률
    # 활성함수는 sigmoid 사옹 -> Binary 결과로 매핑된다. (퍼셉트론과 비슷)
    #

    params = {'C': [1, 3, 5, 7, 9]}
    grid_cv = GridSearchCV(clf, param_grid=params, cv=4, scoring='accuracy', verbose=1)
    grid_cv.fit(tfv_train_x, train_y)

    best_pars = grid_cv.best_params_
    best_model = grid_cv.best_estimator_

    joblib.dump("model_save2.joblib")
"""
