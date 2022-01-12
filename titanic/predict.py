import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

# 데이터 전처리
## 결측치 제거
train.isnull().sum()
test.isnull().sum()
'''
train
Age : 177 / Pclass에 따라서 Age를 짐작할 수 있다고 판단하여, Pclass에 따른 나이의 중간 값으로 대체
Cabin : 687 / Pclass에 따라서 Cabin을 짐작할 수 있다고 판단하여, Pclass에 따른 Cabin의 분포별 값으로 대체
Embarked : 2 / 삭제

test
Age : 86 / Pclass에 따라서 Age를 짐작할 수 있다고 판단하여, Pclass에 따른 나이의 중간 값으로 대체
Cabin : 327 / Pclass에 따라서 Cabin을 짐작할 수 있다고 판단하여, Pclass에 따른 Cabin의 분포별 값으로 대체
Fare : 1 / 0으로 대체(중요도 낮음)
'''
## Pclass별 평균 Age 값으로 대체
age_by_pclass_train = {x : float(int(train[train['Pclass']==x]['Age'].mean())) for x in train['Pclass'].unique()}
age_by_pclass_test = {x : float(int(test[test['Pclass']==x]['Age'].mean())) for x in test['Pclass'].unique()}
train['Age'] = [x if x>=0.0 else age_by_pclass_train[y] for x,y in zip(train['Age'], train['Pclass'])]
train['Age'] = train['Age'].astype(int)
test['Age'] = [x if x>=0.0 else age_by_pclass_test[y] for x,y in zip(test['Age'], test['Pclass'])]
test['Age'] = test['Age'].astype(int)

## Cabin 데이터 대체
## train
## Cabin data에 spacebar가 들어가있는 경우에는 뒤에 가장 뒤에 있는 값으로 대체
train['Cabin'] = [x.split()[-1] if ' ' in x else x for x in train['Cabin'].astype(str)]
## Cabin data의 첫번째 알파벳만 남기고 제거
train['Cabin'] = [x[:1] for x in train['Cabin']]
## Cabin != nan
df_cabin = train[train['Cabin'] != 'n']
p1_per = df_cabin[train['Pclass']==1]['Cabin'].value_counts()/len(df_cabin[train['Pclass']==1])
p2_per = df_cabin[train['Pclass']==2]['Cabin'].value_counts()/len(df_cabin[train['Pclass']==2])
p3_per = df_cabin[train['Pclass']==3]['Cabin'].value_counts()/len(df_cabin[train['Pclass']==3])
## Cabin == nan
df_nan = train[train['Cabin']=='n']
## Cabin == nan, Pclass == 1
p1_values = np.round(np.array(p1_per)*len(df_nan[train['Pclass']==1])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin = {x:y for x,y in zip(df_nan[train['Pclass']==1]['PassengerId'],''.join([x*y for x,y in zip(p1_per.keys(), p1_values)]))}
## Cabin == nan, Pclass == 2
p2_values = np.round(np.array(p2_per)*len(df_nan[train['Pclass']==2])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[train['Pclass']==2]['PassengerId'],''.join([x*y for x,y in zip(p2_per.keys(), p2_values)]))})
## Cabin == nan, Pclass == 3
p3_values = np.round(np.array(p3_per)*len(df_nan[train['Pclass']==3])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[train['Pclass']==3]['PassengerId'],''.join([x*y for x,y in zip(p3_per.keys(), p3_values)]))})
## dataframe의 PassengerId별 Cabin 값
cabin_by_pid = {x:y for x,y in zip(train['PassengerId'], train['Cabin'])}
## nan_cabin update
cabin_by_pid.update(nan_cabin)
## dataframe에 최종 cabin 값 업데이트
train['Cabin'] = list(cabin_by_pid.values())

## test
## Cabin data에 spacebar가 들어가있는 경우에는 뒤에 가장 뒤에 있는 값으로 대체
test['Cabin'] = [x.split()[-1] if ' ' in x else x for x in test['Cabin'].astype(str)]
## Cabin data의 첫번째 알파벳만 남기고 제거
test['Cabin'] = [x[:1] for x in test['Cabin']]
## Cabin != nan
df_cabin = test[test['Cabin'] != 'n']
p1_per = df_cabin[test['Pclass']==1]['Cabin'].value_counts()/len(df_cabin[test['Pclass']==1])
p2_per = df_cabin[test['Pclass']==2]['Cabin'].value_counts()/len(df_cabin[test['Pclass']==2])
p3_per = df_cabin[test['Pclass']==3]['Cabin'].value_counts()/len(df_cabin[test['Pclass']==3])
## Cabin == nan
df_nan = test[test['Cabin']=='n']
## Cabin == nan, Pclass == 1
p1_values = np.round(np.array(p1_per)*len(df_nan[test['Pclass']==1])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin = {x:y for x,y in zip(df_nan[test['Pclass']==1]['PassengerId'],''.join([x*y for x,y in zip(p1_per.keys(), p1_values)]))}
## Cabin == nan, Pclass == 2
p2_values = np.round(np.array(p2_per)*len(df_nan[test['Pclass']==2])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[test['Pclass']==2]['PassengerId'],''.join([x*y for x,y in zip(p2_per.keys(), p2_values)]))})
## Cabin == nan, Pclass == 3
p3_values = np.round(np.array(p3_per)*len(df_nan[test['Pclass']==3])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[test['Pclass']==3]['PassengerId'],''.join([x*y for x,y in zip(p3_per.keys(), p3_values)]))})
## dataframe의 PassengerId별 Cabin 값
cabin_by_pid = {x:y for x,y in zip(test['PassengerId'], test['Cabin'])}
## nan_cabin update
cabin_by_pid.update(nan_cabin)
## dataframe에 최종 cabin 값 업데이트
test['Cabin'] = list(cabin_by_pid.values())

## Embarked == nan, 제거
train.dropna(subset=['Embarked'], inplace=True)
train.isnull().sum()
## Fare 데이터 0으로 대체
test[test['Fare'].isnull()] = 0
# 이상값 대체
test.loc[test[test['Sex'] == 0].index[0],'Sex'] = 'female'
# 성별 데이터 one-hot encoding
train['Sex'] = train['Sex'].map({'female':0,'male':1})
test['Sex'] = test['Sex'].map({'female':0,'male':1})
# 데이터 확인
train.info()
test.info()

#--------------------------------------------
# # 모델 테스트


# train['Name'] = train['Name'].astype('category')
# train['Ticket'] = train['Ticket'].astype('category')
# train['Cabin'] = train['Cabin'].astype('category')
# train['Embarked'] = train['Embarked'].astype('category')
# x_train, x_test, y_train, y_test = train_test_split(train, train['Survived'], test_size=0.3, random_state=1, stratify=train['Survived'])

# clf = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3, tree_method='gpu_hist', enable_categorical=True)
# evals = [(x_test,y_test)]
# clf.fit(x_train, y_train,early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)
# # test
# prediction = clf.predict(x_test)
# # score
# print(f'train score : {clf.score(x_train,y_train)}')
# print(f'test accuracy : {accuracy_score(y_test, prediction)}')

#----------------------------------------
# 결과
from xgboost import XGBClassifier
x_train = train.drop(columns=['Survived'])
y_train = train.loc[:, 'Survived']
x_test = test

x_train['Name'] = x_train['Name'].astype('category')
x_train['Ticket'] = x_train['Ticket'].astype('category')
x_train['Cabin'] = x_train['Cabin'].astype('category')
x_train['Embarked'] = x_train['Embarked'].astype('category')
x_test['Name'] = x_test['Name'].astype('category')
x_test['Ticket'] = x_test['Ticket'].astype('category')
x_test['Cabin'] = x_test['Cabin'].astype('category')
x_test['Embarked'] = x_test['Embarked'].astype('category')

model = XGBClassifier(tree_method='gpu_hist' ,n_estimators=400, enable_categorical=True)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
submission['Survived'] = prediction
submission.to_csv('submission_XGB.csv', index=False)

## Dacon accuracy_score 0.72


