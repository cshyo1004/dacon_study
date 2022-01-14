import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

## 결측치 확인
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

# 데이터 전처리
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

## Embarked 데이터 삭제
train.dropna(subset=['Embarked'], inplace=True)
train.isnull().sum()

## Fare 데이터 0으로 대체
test['Fare'].fillna(0, inplace=True)

## Fare 데이터 dtype int형으로 변경
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

## PassengerId 삭제
train.drop(columns=['PassengerId'], inplace=True)
test.drop(columns=['PassengerId'], inplace=True)

## 데이터 확인
train.info()
test.info()
train['Embarked'].value_counts()
# 범주형 데이터 전처리
'''
Name : 삭제
Sex : encoding
Ticket : 삭제
Cabin : encoding
Embarked : encoding
'''
train.drop(columns=['Name','Ticket'], inplace=True)
test.drop(columns=['Name','Ticket'], inplace=True)
# encoding
## Sex
concat_sex_train = pd.get_dummies(train['Sex'], prefix='Sex')
concat_sex_test = pd.get_dummies(test['Sex'], prefix='Sex')
## Cabin
concat_cabin_train = pd.get_dummies(train['Cabin'], prefix='Cabin')
concat_cabin_test = pd.get_dummies(test['Cabin'], prefix='Cabin')
## Embarked
concat_embarked_train = pd.get_dummies(train['Embarked'], prefix='Embarked')
concat_embarked_test = pd.get_dummies(test['Embarked'], prefix='Embarked')
# concat
train = pd.concat([train, concat_sex_train, concat_cabin_train, concat_embarked_train], axis=1)
train.drop(columns=['Sex','Cabin','Embarked'], inplace=True)
test = pd.concat([test, concat_sex_test, concat_cabin_test, concat_embarked_test], axis=1)
test.drop(columns=['Sex','Cabin','Embarked'], inplace=True)

# 모델 학습
xtrain, ytrain, xtest, ytest = train_test_split(train.drop(columns=['Survived']), train.loc[:, 'Survived'], test_size=0.3, random_state=1, stratify=train.loc[:, 'Survived'])
train_pool = Pool(xtrain, xtest, cat_features=np.where(xtrain.columns)[0])
eval_pool = Pool(ytrain, ytest, cat_features=np.where(ytrain.columns)[0])
model = CatBoostClassifier(eval_metric='AUC', use_best_model=True)
model.fit(train_pool, eval_set=eval_pool)
prediction = model.predict(ytrain)
print(f'train score : {model.score(xtrain, xtest)}')
print(f'model eval : {accuracy_score(ytest, prediction)}')

# 결과
x_train = train.drop(columns=['Survived'])
x_test = train.loc[:, 'Survived']
y_train = test
y_train['Cabin_T'] = 0

prediction = model.predict(y_train)
submission['Survived'] = prediction
submission.to_csv('submission_CAT.csv', index=False)