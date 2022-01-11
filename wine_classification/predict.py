'''
index 구분자
quality 품질
fixed acidity 산도
volatile acidity 휘발성산
citric acid 시트르산
residual sugar 잔당 : 발효 후 와인 속에 남아있는 당분
chlorides 염화물
free sulfur dioxide 독립 이산화황
total sulfur dioxide 총 이산화황
density 밀도
pH 수소이온농도
sulphates 황산염
alcohol 도수
type 종류

1. target data 파악
2. PCA
3. 결측치 제거

모델 XGBClassifier
선정 이유 : kaggle에서 분류 모델 순위 1위
https://towardsdatascience.com/choosing-the-best-classification-algorithm-f254f68cca39
'''

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# feature / target 나누기
x_data = train.iloc[:,2:]
x_target = train.loc[:,'quality']

# target 파악
x_target.value_counts().plot(kind='bar')
plt.show()

# type 데이터 one-hot encoding
x_data['type'].unique()
x_data['type'] = [0 if x == 'white' else 1 for x in x_data['type']]

# PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_data)

# pc값의 설명력
# 값이 높을수록 설명력이 높음
# 0.1 이하의 feature 제외
pca.explained_variance_

# 결측치 제거
def find_na(df):
    for column in df.columns:
        if not df[df[column].isna()].empty:
            print(column, len(df[df[column].isna()]))
find_na(train)

# train / test
x_train = x_data
y_train = x_target
x_test = test.iloc[:,1:]
x_test['type'] = [0 if x=='white' else 1 for x in x_test['type']]

# 모델
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# train
model = XGBClassifier(n_estimators=300)
model.fit(x_train, y_train)

# test
prediction = model.predict(x_test)

# score
print(f'train score : {model.score(x_train,y_train)}')

# 결과
submission['quality'] = prediction
submission.to_csv('submission.csv', index=False)
