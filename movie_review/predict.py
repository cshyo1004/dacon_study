import pandas as pd
import os



'''
https://www.koreascience.or.kr/article/CFKO202025036019264.pdf

ETRI KorBERT : 형태소 단위로 문장을 나누어 데이터 학습

https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201909358629867&oCn=JAKO201909358629867&dbt=JAKO&journal=NJOU00400536


'''
# 데이터 불러오기
bpath = 'movie_review'
train = pd.read_csv(os.path.join(bpath, 'train.csv'))
test = pd.read_csv(os.path.join(bpath, 'test.csv'))
submission = pd.read_csv(os.path.join(bpath, 'sample_submission.csv'))

# 데이터 정제
# 결측치
def find_na(df):
    for column in df.columns:
        if not df[df[column].isnull()].empty:
            print(column, len(df[df[column].isnull()]))
find_na(train)            
find_na(test)
# 기호 제거
import re
train['document'] = train['document'].apply(lambda x : re.sub('[^ 가-힣*]', '', x))
test['document'] = test['document'].apply(lambda x : re.sub('[^ 가-힣*]', '', x))

find_na(train)
find_na(test)

# 온점 제거
train['document'] = train['document'].str.replace('.', '')
test['document'] = test['document'].str.replace('.', '')

# 텍스트 백터화
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train['document'])
x_train = vectorizer.transform(train['document'])
x_test = train.label

# 모델 적용
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, x_test)

# 결과
X_pred = vectorizer.transform(["영화 완전 꿀잼"]) 
y_pred = model.predict(X_pred)
submission['label'] = y_pred
submission.to_csv('LR_result.csv', index=False)

# 텍스트 백터화2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer()
vectorizer.fit(train['document'])
x_train = vectorizer.transform(train['document'])
x_test = train.label
y_train = vectorizer.transform(test['document'])

# 모델 적용2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train, x_test)

# 결과2
prediction = model.predict(y_train)
submission['label'] = prediction
submission.to_csv('RF_result.csv', index=False)

# 텍스트 백터화2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
vectorizer = CountVectorizer()
vectorizer.fit(train['document'])
data = vectorizer.transform(train['document'])
x_train, x_test, y_train, y_test = train_test_split(data, train.label, test_size=0.33, random_state=1)

# 모델 적용2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train, x_test)

# 결과2
prediction = model.predict(y_train)
submission['label'] = prediction
submission.to_csv('RF_result.csv', index=False)