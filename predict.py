import pandas as pd
import os
#ARMIA
from statsmodels.tsa.arima_model import ARIMA
#LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

bpath = 'public_bicycle_usage'
train = pd.read_csv(os.path.join(bpath, 'train.csv'))
test = pd.read_csv(os.path.join(bpath, 'test.csv'))
submission = pd.read_csv(os.path.join(bpath, 'submission.csv'))

'''
- temperature 시간에 따른 기온값 확인해서 데이터 추가 / 사유: 특정 시간에 따라서 기온이 평균적으로 형성되어있다고 생각되어짐
- precipitation 기온값과 비교하여 임의의 데이터 추가 / 사유: 기온에 따라서 비가 왔는지 안 왔는지 대략적으로 짐작 가능
- windspeed 해당 데이터 삭제 / 사유: 예측 안됨
- humidity 해당 데이터 삭제 / 사유: 예측 안됨
- visibility 해당 데이터 삭제 / 사유: 예측 안됨
- ozone 해당 데이터 삭제 / 사유: 예측 안됨
- pm10 해당 데이터 삭제 / 사유: 예측 안됨
- pm2.5 해당 데이터 삭제 / 사유: 예측 안됨
'''

len(train)
train.dropna(subset=['hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'], inplace=True)

for column in train.columns:
    if not train[train[column].isna()].empty:
        print(column, train[train[column].isna()])
        
len(test)
test.dropna(subset=['hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'], inplace=True)

for column in test.columns:
    if not test[test[column].isna()].empty:
        print(column, test[test[column].isna()])
        
ARIMA        
        
