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

1. 결측치 제거
2. 데이터 전처리
3. 모델 학습
4. 결과

모델 XGBClassifier
'''

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# 결측치 제거
def find_na(df):
    for column in df.columns:
        if not df[df[column].isna()].empty:
            print(column, len(df[df[column].isna()]))
find_na(train)

# 데이터 전처리
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS.fit(train)
x_ss = SS.transform(train)

# feature / target 나누기
x_data = train.iloc[:,2:]
x_target = train.loc[:,'quality']

# type 데이터 one-hot encoding
x_data['type'].unique()
x_data['type'] = [0 if x == 'white' else 1 for x in x_data['type']]

# 모델 학습
x_train, y_train, x_test, y_test = train_test_split(x_data, x_target, test_size=0.33, random_state=1, stratify=x_target)
model = XGBClassifier(n_estimators=300)
evals = [(y_train,y_test)]
params = {'max_depth':[4,5,6,7,8], 'min_child_weight':[1,3], 'learning_rate':[0.05,0.1,0.5]}
gridcv = GridSearchCV(model, param_grid=params)
gridcv.fit(x_train, x_test, early_stopping_rounds=30, eval_set=evals)
print(gridcv.best_params_)
print(gridcv.best_score_)

model = XGBClassifier(n_estimators=300, max_depth=8, min_child_weight=1, learning_rate=0.5)
model.fit(x_train, x_test, early_stopping_rounds=100, eval_set=evals, verbose=True)
prediction = model.predict(y_train)
print(f'train score : {model.score(x_train,x_test)}')
print(f'model eval : {accuracy_score(y_test, prediction)}')

# 결과
x_train = x_data
x_test = x_target
y_train = test.iloc[:,1:]
y_train['type'] = [0 if x=='white' else 1 for x in x_test['type']]

prediction = model.predict(y_train)
submission['quality'] = prediction
submission.to_csv('submission.csv', index=False)
