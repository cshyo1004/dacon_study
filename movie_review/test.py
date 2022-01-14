from transformers import ElectraModel, ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
tokenizer.encode_plus(text="[CLS] 한국어 ELECTRA를 공유합니다. [SEP]",padding=True)
tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'EL', '##EC', '##TRA', '##를', '공유', '##합니다', '.', '[SEP]'])

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy
import numpy as np
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings('ignore')


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
train['document'] = train['document'].apply(lambda x : re.sub('[^ a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣*]', '', x))
test['document'] = test['document'].apply(lambda x : re.sub('[^ a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣*]', '', x))
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 텍스트 데이터 전처리
train['document_tokenized'] = train['document'].apply(lambda x : tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
train_words = scipy.sparse.csr_matrix(pad_sequences(train['document_tokenized'], maxlen=128, dtype=float, padding='post'))
x_train, y_train, x_test, y_test = train_test_split(train_words, train.label, test_size=0.2, random_state=1, stratify=train.label)

# 모델 테스트
model = LogisticRegression()
model.fit(x_train, x_test)
prediction = model.predict(y_train)

# 스코어
accuracy = accuracy_score(y_test, prediction)
print(f'accuracy score: {accuracy:.3}')

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train, x_test)
prediction = model.predict(y_train)

# 스코어
accuracy = accuracy_score(y_test, prediction)
print(f'Mean accuracy score: {accuracy:.3}')



# 텍스트 데이터 전처리
train['document_tokenized'] = train['document'].apply(lambda x : tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
test['document_tokenized'] = test['document'].apply(lambda x : tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
train_words = scipy.sparse.csr_matrix(pad_sequences(train['document_tokenized'], maxlen=128, dtype=float, padding='post'))
test_words = scipy.sparse.csr_matrix(pad_sequences(test['document_tokenized'], maxlen=128, dtype=float, padding='post'))

x_train = train_words
x_test = train.label
y_train = test_words








# 모델 
model = LogisticRegression()
model.fit(x_train, x_test)

# 결과
prediction = model.predict(y_train)
submission['label'] = prediction
submission.to_csv('submission.csv', index=False)






