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
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

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

# 모델 검증
x_train, y_train, x_test, y_test = train_test_split(x_data, x_target, test_size=0.33, random_state=1, stratify=x_target)
model = XGBClassifier(n_estimators=300)
evals = [(y_train,y_test)]
params = {'max_depth':[5,7,8], 'min_child_weight':[1,3], 'learning_rate':[0.05,0.1,0.5]}
gridcv = GridSearchCV(model, param_grid=params)
gridcv.fit(x_train, x_test, early_stopping_rounds=30, eval_set=evals)
print(gridcv.best_params_)

model = XGBClassifier(n_estimators=300, max_depth=8, min_child_weight=1, learning_rate=0.5)
model.fit(x_train, x_test, early_stopping_rounds=100, eval_set=evals, verbose=True)
prediction = model.predict(y_train)
print(f'train score : {model.score(x_train,x_test)}')
print(f'model eval : {accuracy_score(y_test, prediction)}')

# 데이터 정제
x_train = x_data
x_test = x_target
y_train = test.iloc[:,1:]
y_train['type'] = [0 if x=='white' else 1 for x in x_test['type']]

# 모델 적용
model = XGBClassifier(n_estimators=300, max_depth=8, min_child_weight=1, learning_rate=0.5)
model.fit(x_train, x_test)
prediction = model.predict(y_train)

# 결과
submission['quality'] = prediction
submission.to_csv('submission.csv', index=False)
