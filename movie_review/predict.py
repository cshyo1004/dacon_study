import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# 데이터 전처리
# 결측치 확인
train.isnull().sum()
test.isnull().sum()

# 기호 제거
train['document'] = train['document'].apply(lambda x : re.sub('[^ a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣*]', '', x))
test['document'] = test['document'].apply(lambda x : re.sub('[^ a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣*]', '', x))

# 온점 제거
train['document'] = train['document'].str.replace('.', '')
test['document'] = test['document'].str.replace('.', '')

# 빈칸 제거
train['document'] = train['document'].apply(lambda x : ' '.join(x.split()))
test['document'] = test['document'].apply(lambda x : ' '.join(x.split()))

# 텍스트 데이터 정제
okt = Okt()
train_words = [' '.join(okt.morphs(x, stem=True)) for x in train['document']]
test_words = [' '.join(okt.morphs(x, stem=True)) for x in test['document']]

# 모델 학습
for m_feature in range(1000,10000,1000):
    # TfidfVectorizer
    Tvectorizer = TfidfVectorizer(analyzer="word", sublinear_tf=True, ngram_range=(1, 2), max_features=m_feature)
    Tvectorizer.fit(train_words)
    data = Tvectorizer.transform(train_words)
    x_train, y_train, x_test, y_test = train_test_split(data, train.label, test_size=0.33, random_state=1, stratify=train.label)

    # 모델 테스트
    model = LogisticRegression()
    model.fit(x_train, x_test)
    prediction = model.predict(y_train)

    # 스코어
    accuracy = accuracy_score(y_test, prediction)
    print(f'train score : {model.score(x_train,x_test)}')
    print(f'accuracy score: {accuracy:.3}')

# 결과
eval_data = Tvectorizer.transform(test_words)
prediction = model.predict(eval_data)
submission['label'] = prediction
submission.to_csv('submission_2.csv', index=False)