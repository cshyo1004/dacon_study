# 환경 설정
'''
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# 데이터 불러오기
df = pd.read_csv('train.csv')
# 상위 5개
df.head()
# shape
df.shape
# 통계량
df.describe(include='all')
# feature dtype 파악
df.dtypes
# features
'''
PassengerID : 탑승객 고유 아이디
Survival : 탑승객 생존 유무 (0: 사망, 1: 생존)
Pclass : 객실 등급
Name : 이름
Sex : 성별
Age : 나이
Sibsp : 함께 탐승한 형제자매, 아내, 남편의 수
Parch : 함께 탐승한 부모, 자식의 수
Ticket :티켓 번호
Fare : 티켓의 요금
Cabin : 객실번호
Embarked : 배에 탑승한 항구 이름 ( C = Cherbourn, Q = Queenstown, S = Southampton)
'''
# 결측치 제거
df.isnull().sum().to_frame('nan_count')
df.isnull().sum()/len(df)
'''
Age : 177 / Pclass에 따라서 Age를 짐작할 수 있다고 판단하여, Pclass에 따른 나이의 중간 값으로 대체
Cabin : 687 / Pclass에 따라서 Cabin을 짐작할 수 있다고 판단하여, Pclass에 따른 Cabin의 최대분포 값으로 대체
Embarked : 2 / 삭제
'''
## Pclass별 Age 분포도
sns.boxplot(x='Pclass', y='Age', data=df[['Pclass','Age']])
plt.show()
## Pclass == 1, Age 분포도
plt.hist(df[df['Pclass']==1]['Age'])
plt.show()
## Pclass별 평균 Age 값
age_by_pclass = {x : float(int(df[df['Pclass']==x]['Age'].mean())) for x in df['Pclass'].unique()}
# Age 값 업데이트
df['Age'] = [x if x>=0.0 else age_by_pclass[y] for x,y in zip(df['Age'], df['Pclass'])]
df['Age'] = df['Age'].astype(int)

## Cabin data에 spacebar가 들어가있는 경우에는 뒤에 가장 뒤에 있는 값으로 대체
df['Cabin'] = [x.split()[-1] if ' ' in x else x for x in df['Cabin'].astype(str)]
## Cabin data의 첫번째 알파벳만 남기고 제거
df['Cabin'] = [x[:1] for x in df['Cabin']]
## Cabin 분포도
df['Cabin'].value_counts().plot(kind='bar')
plt.show()
## Cabin != nan
df_cabin = df[df['Cabin'] != 'n']
## Pclass == 1, Cabin 분포도
plt.hist(df_cabin[df['Pclass']==1]['Cabin'])
plt.show()
p1_per = df_cabin[df['Pclass']==1]['Cabin'].value_counts()/len(df_cabin[df['Pclass']==1])
## Pclass == 2, Cabin 분포도
plt.hist(df_cabin[df['Pclass']==2]['Cabin'])
plt.show()
p2_per = df_cabin[df['Pclass']==2]['Cabin'].value_counts()/len(df_cabin[df['Pclass']==2])
## Pclass == 3, Cabin 분포도
plt.hist(df_cabin[df['Pclass']==3]['Cabin'])
plt.show()
p3_per = df_cabin[df['Pclass']==3]['Cabin'].value_counts()/len(df_cabin[df['Pclass']==3])
## Cabin == nan
df_nan = df[df['Cabin']=='n']
## Cabin == nan, Pclass 분포
df_nan['Pclass'].value_counts().plot(kind='bar')
plt.show()
## Cabin == nan, Pclass == 1
p1_values = np.round(np.array(p1_per)*len(df_nan[df['Pclass']==1])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin = {x:y for x,y in zip(df_nan[df['Pclass']==1]['PassengerId'],''.join([x*y for x,y in zip(p1_per.keys(), p1_values)]))}
## Cabin == nan, Pclass == 2
p2_values = np.round(np.array(p2_per)*len(df_nan[df['Pclass']==2])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[df['Pclass']==2]['PassengerId'],''.join([x*y for x,y in zip(p2_per.keys(), p2_values)]))})
## Cabin == nan, Pclass == 3
p3_values = np.round(np.array(p3_per)*len(df_nan[df['Pclass']==3])).astype(int)
## PassengerId별 Cabin 값 저장
nan_cabin.update({x:y for x,y in zip(df_nan[df['Pclass']==3]['PassengerId'],''.join([x*y for x,y in zip(p3_per.keys(), p3_values)]))})
## dataframe의 PassengerId별 Cabin 값
cabin_by_pid = {x:y for x,y in zip(df['PassengerId'], df['Cabin'])}
## nan_cabin update
cabin_by_pid.update(nan_cabin)
## dataframe에 최종 cabin 값 업데이트
df['Cabin'] = list(cabin_by_pid.values())

## Embarked == nan, 제거
df.dropna(subset=['Embarked'], inplace=True)
df.isnull().sum()

# 데이터 확인
df.info()
# 성별 데이터 one-hot encoding
df['Sex'] = df['Sex'].map({'female':'0','male':'1'})