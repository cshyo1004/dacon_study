import pandas as pd
import os

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

'''
데이터의 정상성 : 시계열의 평균과 분산이 일정하고, 특정한 트렌드 (추세)가 존재하지 않는 성질
'''

'''
비정상 시계열 데이터를 정상 시계열로 전환하는 방법
'''
# import matplotlib.pyplot as plt
# #로그변환
# import numpy as np
# log_count = np.log(train['count'])
# plt.plot(log_count)
# plt.show()

# # scale
# from sklearn import preprocessing
# log_count = preprocessing.scale(log_count)
# plt.plot(log_count)
# plt.show()

# #1차 차분
# diff_count = np.diff(train['count'])
# plt.plot(diff_count)
# plt.show()

# #2차 차분
# diff2_count = np.diff(train['count'],2)
# plt.plot(diff2_count)
# plt.show()

# #로그변환 + 차분
# fcount = np.log(train['count'])
# plt.plot(fcount)
# plt.show()

# fcount = np.diff(fcount)
# plt.plot(fcount)
# plt.show()

'''
ARIMA model
p : AR(자기 회귀 모델)
q : MA(이동 평균 모델)
d : I(차분)
'''
from statsmodels.tsa.stattools import adfuller
result = adfuller(train['count'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(train['count']); axes[0, 0].set_title('Original Series')
plot_acf(train['count'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(train['count'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(train['count'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(train['count'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(train['count'].diff().diff().dropna(), ax=axes[2, 1])

plt.show()

from pmdarima.arima.utils import ndiffs
y = train['count']
## Adf Test
ndiffs(y, test='adf')

# KPSS test
ndiffs(y, test='kpss')

# PP test:
ndiffs(y, test='pp')

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['count'], order=(0,0,0))
model_fit = model.fit()
print(model_fit.summary())
fore = model_fit.forecast(steps=50)
print(fore)

