import pandas as pd
import re
import numpy as np
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt


# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# 데이터 정제
# 결측치
def find_na(df):
    for column in df.columns:
        if not df[df[column].isna()].empty:
            print(column, len(df[df[column].isna()]))
find_na(train)            
find_na(test)
# 기호 제거
train['document'] = train['document'].apply(lambda x : re.sub('[^ 가-힣*]', '', x))
test['document'] = test['document'].apply(lambda x : re.sub('[^ 가-힣*]', '', x))
find_na(train)
find_na(test)

# 온점 제거
train['document'] = train['document'].str.replace('.', '')
test['document'] = test['document'].str.replace('.', '')

# 빈칸 제거
train['document'] = train['document'].apply(lambda x : ' '.join(x.split()))
test['document'] = test['document'].apply(lambda x : ' '.join(x.split()))
find_na(train)
find_na(test)

# 명사만 추출했을 때 길이
okt = Okt()
len_nouns = len(set([y for x in train['document'] for y in okt.nouns(x)])) # 5207
# 전체 단어 길이
len_words = len(set([y for x in train['document'] for y in okt.morphs(x)])) # 10409
# 명사만 추출
train_words = [' '.join(okt.nouns(x)) for x in train['document']]
test_words = [' '.join(okt.nouns(x)) for x in test['document']]

# #--------------------------------------------------------------------
# # 텍스트 벡터화
# Tvectorizer = TfidfVectorizer()
# Tvectorizer.fit(train_words)
# data = Tvectorizer.transform(train_words)
# x_train, x_test, y_train, y_test = train_test_split(data, train.label, test_size=0.33, stratify=train.label)

# # LSTM 학습을 위해 3차원 배열로 X값을 변형
# x_train = x_train.toarray()[:, :, np.newaxis]
# x_test = x_test.toarray()[:, :, np.newaxis]

# # build model
# model = keras.Sequential()
# model.add(keras.layers.LSTM(units=128,input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(keras.layers.Dropout(rate=0.5))
# model.add(keras.layers.Dense(units=1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# # train
# model.fit(x_train, y_train, validation_split=0.2, epochs=1)

# # test
# result = model.predict(x_test)
# result
# # score
# score = model.evaluate(y_test, result)
# print(f'score: {score:.3}')
# #--------------------------------------------------------------------

# 텍스트 벡터화
Tvectorizer = TfidfVectorizer()
Tvectorizer.fit(train_words)
x_train = Tvectorizer.transform(train_words)
y_train = train.label
x_test = Tvectorizer.transform(test_words)

# LSTM 학습을 위해 3차원 배열로 X값을 변형
x_train = x_train.toarray()[:, :, np.newaxis]
x_test = x_test.toarray()[:, :, np.newaxis]

# build model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=128,input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# train
model.fit(x_train, y_train, validation_split=0.2, epochs=1)

# 결과
prediction = model.predict(x_test)
submission['label'] = prediction
submission.to_csv('LSTM_result.csv', index=False)


