
# 탐색적 데이터 분석
# 리뷰 정보와 리뷰 정보의 긍정/부정 평가
'''
document : 리뷰 정보
label : 긍정/부정 평가
'''

# 데이터 불러오기
import pandas as pd
import os
bpath = 'movie_review'
train = pd.read_csv(os.path.join(bpath, 'train.csv'))
test = pd.read_csv(os.path.join(bpath, 'test.csv'))
submission = pd.read_csv(os.path.join(bpath, 'sample_submission.csv'))

# 결측치 확인
def find_na(df):
    for column in df.columns:
        if not df[df[column].isnull()].empty:
            print(column, len(df[df[column].isnull()]))

find_na(train)
find_na(test)

# 기초 통계 분석
train.info()

# 긍정/부정 리뷰 비율 확인
train_values = train['label'].value_counts()
train_values[0]
train_values[1]

# 데이터 시각화
import matplotlib.pyplot as plt
plt.title('label', fontsize=10)
plt.bar(train_values.keys(), train_values.values)
plt.xticks(train_values.keys())
plt.show()

# 리뷰 길이 확인
import numpy as np
review_avg_len = np.mean(train['document'].str.len())

# 전체리뷰 / 긍정리뷰 / 부정리뷰 비교
pos_review = train[train['label']==1].document
neg_review = train[train['label']==0].document
compare = [train.document, pos_review, neg_review]

# 리뷰 길이 히스토그램
# plt.figure(figsize=(20,7.5))
plt.suptitle("Histogram: review length", fontsize=20)
name = ['total dataset', 'positive reviews', 'negative reviews']

for i in range(len(compare)):
    document = compare[i]
    string_len = [len(x) for x in document]    
    plt.subplot(1,3,i+1) # 행 개수/ 열 개수/ 해당 그래프 표시 순서
    plt.title(name[i], fontsize=10)
    plt.axis([16, 42, 0, 800])  #x축 시작, 끝 / y축 시작, 끝
    plt.hist(string_len, alpha=0.5, color='orange')
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

word_split = train['document'].str.split()
# plt.figure(figsize=(20,7.5))
plt.suptitle("Histogram: word count", fontsize=20)
name = ['total dataset', 'positive reviews', 'negative reviews']

for i in range(len(compare)):
    document = compare[i]
    split = document.str.split()
    split_len = [len(x) for x in split] 
    plt.subplot(1,3,i+1) # 행 개수/ 열 개수/ 해당 그래프 표시 순서
    plt.title(name[i], fontsize=10)
    plt.axis([1, 15, 0, 1750])  #x축 시작, 끝 / y축 시작, 끝
    plt.hist(split_len, alpha=0.5, color='purple')
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 워드 클라우드
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud

def df2str(df):
    s = [s for s in df]
    document = ""
    for i in range(len(s)):
        document += s[i]
    return document

def get_noun(text):
    okt = Okt()
    noun = okt.nouns(text) # 텍스트를 단어 단위로 분류
    for i,v in enumerate(noun):
        if len(v) < 2:
            noun.pop(i)
    count = Counter(noun)
    noun_list = count.most_common(100) # 100번째 큰 순서까지 데이터 추출
    return noun_list

def visualize(noun_list, title):
    wc = WordCloud(
        font_path = 'movie_review/NanumBarunGothic.ttf',
        background_color = 'white',
        colormap = 'Dark2',
        width = 800,
        height = 800
    )    .generate_from_frequencies(dict(noun_list))
    
    plt.figure(figsize=(10,10))
    plt.suptitle("Word Cloud", fontsize=20)
    plt.title(title, fontsize=10)
    plt.imshow(wc, interpolation='lanczos')
    plt.axis('off')
    plt.show()
    return wc

# view 전체 데이터
document = df2str(train['document'])
noun_list = get_noun(document)
noun_list[:5]

plt.rc('font', family='NanumBarunGothic')
top_10 = dict(noun_list[:10])
plt.figure(figsize=(10,7.5))
plt.suptitle("bar plot", fontsize=20)
plt.title('total reviews', fontisze=10)
plt.bar(top_10.keys(), top_10.values())
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

total_reviews = visualize(noun_list, 'total_reviews')

# view 긍정 리뷰
document_p = df2str(pos_review)
noun_list_p = get_noun(document_p)
noun_list_p[:5]

plt.rc('font', family='NanumBarunGothic')
top_10_p = dict(noun_list_p[:10])
plt.figure(figsize=(10,7.5))
plt.suptitle('bar plot', fontsize=20)
plt.title('positive reviews', fontsize=10)
plt.bar(top_10_p.keys(), top_10_p.values())
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

positive_reviews = visualize(noun_list_p, 'positive_reviews')

# view 부정 리뷰
document_n = df2str(neg_review)
noun_list_n = get_noun(document_n)

plt.rc('font', family='NanumBarunGothic')
top_10_n = dict(noun_list_n[:10])
plt.figure(figsize=(10,7.5))
plt.suptitle('bar plot', fontsize=20)
plt.title('negative reviews', fontsize=10)
plt.bar(top_10_n.keys(), top_10_n.values())
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

negative_reviews = visualize(noun_list_n, 'negative_reviews')

neg = list(dict(noun_list_n).keys())
pos = list(dict(noun_list_p).keys())

drop_words = [x for x in neg if x in pos]
drop_words[:10]
