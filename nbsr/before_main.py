# -*- coding: utf-8 -*- 
import enum
import re
import csv
import shutil
import pickle
import gensim
import pathlib
import os.path
import operator
import matplotlib

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from tqdm import tqdm
from konlpy.tag import Kkma
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from hanspell import spell_checker
from soynlp.word import WordExtractor
from sklearn.linear_model import Ridge
from soynlp.tokenizer import LTokenizer
from matplotlib import font_manager, rc
from soynlp.noun import LRNounExtractor_v2
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Dropout, SimpleRNN, LSTM, GRU

DATA_DIR = './nbsr';
FONT_DIR = './nbsr/font';
DATA_IN_DIR = './nbsr/data/in';
DATA_OUT_DIR = './nbsr/data/out';
WORDCLOUD_FILE_PATH = 'wordcloud.png';
FONT_FILE_PATH = 'font/KOTRA_HOPE.ttf';
SCORE_CORPUS_FILE_PATH = 'before_score_corpus.xlsx';
COMBINED_SCORE_FILE_PATH = 'combined_score.json';
CRAWKLING_DATA_FILE_PATH = 'news_data_collection.xlsx';
PREPROCESSING_CORPUS_FILE_PATH = 'preprocessing_corpus.xlsx';
STOCK_DATA_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';
TF_TOP100_VISUALIZATION_FILE_PATH = 'tf_top100_visualization.png';
CHECKED_PREPROCESSING_CORPUS_FILE_PATH = 'checked_preprocessing_corpus.xlsx'
SEED = 346672;

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

def korean_spell_check(corpus):
    kkma = Kkma()
    sentence_list = kkma.sentences(corpus)

    checked_sentence = []
    for sentence in sentence_list:       
        if len(sentence) < 500:
            spelled_sent = spell_checker.check(sentence)
            checked_sentence.append(spelled_sent.checked)
        else:
            word_list = sentence.split(' ')
            word_list_size = len(word_list)
            
            checked_word_list = []
            
            total_size = 0
            pre_index = 0
            cur_index = 0
            while cur_index < word_list_size:
                if total_size + len(word_list[cur_index]):
                    mini_sent = ' '.join(word_list[pre_index : cur_index])
                    spelled_sent = spell_checker.check(mini_sent)
                    checked_word_list.append(spelled_sent.checked)
                    
                    pre_index = cur_index
                    total_size = 0
                
                total_size += len(word_list[cur_index])    
                cur_index += 1

            if total_size:
                mini_sent = ' '.join(word_list[pre_index : cur_index])
                spelled_sent = spell_checker.check(mini_sent)
                checked_word_list.append(spelled_sent.checked)
            
            checked_sentence.append(' '.join(checked_word_list))
            
    return ' '.join(checked_sentence)

def save_dict_to_file(dict, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(dict, fp)

def load_file_to_dict(file_name):
    data = None
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data

def make_combined_scores(corpus, file_name):
    """## 전에 만들었던 종합 점수 통계치가 존재하면 불러오고 없으면 새로 만듬"""
    if os.path.isfile(file_name):
        return load_file_to_dict(file_name)
    else:
        "## 통계 기반으로 단어(의 경계)를 학습하는 모델 생성"
        word_extractor = WordExtractor(
            min_frequency=100,
            min_cohesion_forward=0.05,
            min_right_branching_entropy=0.0
        )

        """## 모델 훈련 및 통계치 저장"""
        word_extractor.train(corpus)
        words = word_extractor.extract()

        """## 통계치를 이용하여 {단어: 단어 점수} 형태로 변환"""
        cohesion_score = {word:score.cohesion_forward for word, score in words.items()}

        """## WordExtractor 모델에서 명사 추출의 정확성과 합성명사 인식 능력 높인 모델"""
        noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
        nouns = noun_extractor.train_extract(corpus)
        
        # top100 = sorted(nouns.items(), 
        #     key=lambda x:-x[1].frequency)[:100]

        # for i, (word, score) in enumerate(top100):
        #     if i % 5 == 0:
        #         print()
        #     print('%6s (%.2f)' % (word, score.score), end='')

        noun_scores = {noun:score.score for noun, score in nouns.items()}

        """## WordExtractor과 LRNounExtractor_v2의 통계치 점수를 합산"""
        combined_scores = {noun:score + cohesion_score.get(noun, 0)
            for noun, score in noun_scores.items()}

        combined_scores.update(
            {subword:cohesion for subword, cohesion in cohesion_score.items()
            if not (subword in combined_scores)}
        )

        """## 만든 종합 점수 통계치를 저장"""
        save_dict_to_file(combined_scores, file_name)
        
        return combined_scores
    
def make_preprocessing_corpus():
    preprocessing_corpus = []
    file_name = '{}/{}'.format(DATA_OUT_DIR, PREPROCESSING_CORPUS_FILE_PATH)
    
    if os.path.isfile(file_name):
        return  pd.read_excel(file_name).dropna()
    else:
        print('preprocessing_corpus 파일이 없어 새로 만드는 작업을 시작합니다.\n')
        raw_data = pd.DataFrame(columns=['date', 'section'])
        
        print('크롤링한 데이터 불러오는 중...\n')
        df = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, CRAWKLING_DATA_FILE_PATH)).dropna();

        corpus = ['{} {}'.format(t, s) for t, s in zip(df['title'].to_list(), df['section'].to_list())]
        print('크롤링한 데이터를 불러왔습니다.\n')
        
        """## 종합 점수 통계치 생성"""
        combined_scores = make_combined_scores(corpus, '{}/{}'.format(DATA_OUT_DIR, COMBINED_SCORE_FILE_PATH))

        """## L parts(명사/동사/형용사/부사) 나머지 부분이 R parts로 구분하는 Tokenizer"""
        tokenizer = LTokenizer(scores=combined_scores)

        print('\n' + '-' * 100 + '\n토큰 처리를 시작합니다.\n')
        pbar = tqdm(total=len(corpus))
        for cp in corpus:
            lr_list = tokenizer.tokenize(cp, flatten=False)
            
            l_list = []
            for l, r in lr_list:
                l_list.append(l)
            preprocessing_corpus.append(' '.join(l_list))
            
            pbar.update(1)
            
        pbar.close()
        print('\n토큰 처리를 완료하였습니다.\n' + '-' * 100)

        """## 특수 문자 제거"""
        print('-' * 100 + '\n특수 문자 제거를 시작합니다.\n')
        pbar = tqdm(total=len(preprocessing_corpus))
        checked_corpus = []
        for cp in preprocessing_corpus:
            new_cp = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z. ]",'', cp)
            # new_cp = korean_spell_check(new_cp)
            checked_corpus.append(new_cp)
            pbar.update(1)
            
        pbar.close()
        preprocessing_corpus = checked_corpus
        print('\n특수 문자 제거를 완료하였습니다.\n' + '-' * 100)
        
        for date, corpus in zip(df['date'].to_list(), preprocessing_corpus):
            raw_data = raw_data.append({'date': date, 'section': corpus}, ignore_index=True)

        """## 전처리된 파일을 엑셀 형식으로 저장"""
        print('\n데이터프레임 형식으로 저장 중')
        raw_data.to_excel('{}/{}'.format(DATA_OUT_DIR, PREPROCESSING_CORPUS_FILE_PATH), index=False, encoding='utf-8')
        print('\n저장을 완료하였습니다.\n')
            
        return raw_data.dropna()
    
    
def make_checked_preprocessing_corpus(tfidf, preprocessing_corpus):
    checked_preprocessing_corpus = []
    file_name = '{}/{}'.format(DATA_OUT_DIR, CHECKED_PREPROCESSING_CORPUS_FILE_PATH)
    
    """## 목표 파일이 존재하면 불러오고 없다면 새로 만듬"""
    if os.path.isfile(file_name):
        return  pd.read_excel(file_name).dropna()
    else:
        print('checked_preprocessing_corpus 파일이 없어 새로 만드는 작업을 시작합니다.\n')
        pre_raw_data = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, PREPROCESSING_CORPUS_FILE_PATH))
        
        raw_data = pd.DataFrame(columns=['date', 'section'])
        
        word_list = tfidf.get_feature_names_out()
        
        """## TF-IDF 단어를 이용해 전처리 완료"""
        checked_preprocessing_corpus = []
        print('-' * 100 + "\n최종 전처리를 수행 중 입니다.")
        pbar = tqdm(total=len(preprocessing_corpus))
        for cp in preprocessing_corpus:
            result = []
            for c in cp.split(' '):
                check = False
                for w in word_list:
                    if c == w:
                        check = True
                        break
                if check:
                    result.append(c)            
            checked_preprocessing_corpus.append(' '.join(result))
            pbar.update(1)
            
        pbar.close()
        print("\n최종 전처리를 완료하였습니다.\n" + '-' * 100)

        for date, corpus in zip(pre_raw_data['date'].to_list(), checked_preprocessing_corpus):
            raw_data = raw_data.append({'date': date, 'section': corpus}, ignore_index=True)

        """## 최종 전처리된 파일을 엑셀 형식으로 저장"""
        print('\n데이터프레임 형식으로 저장 중')
        raw_data.to_excel('{}/{}'.format(DATA_OUT_DIR, CHECKED_PREPROCESSING_CORPUS_FILE_PATH), index=False, encoding='utf-8')
        print('\n저장을 완료하였습니다.\n')
        
        return raw_data.dropna()
    
def cal_mae(model, test_input, test_target):
    test_prediction = model.predict(test_input);

    mae = mean_absolute_error(test_target, test_prediction);
    print(mae);

def fit_test_regression(model, train_input, train_target, valid_input, valid_target, test_input, test_target):
    model.compile(loss="mae", optimizer="adam")

    history = model.fit(train_input, train_target, epochs=1000, validation_data=(valid_input, valid_target),
    callbacks=[early_stopping_cb])

    train_score = model.evaluate(train_input, train_target)
    test_score = model.evaluate(test_input, test_target)
    print(train_score)
    print(test_score)

    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.show()

"""## 맷 플롯 한글 깨짐 방지 설정"""
font_path = '{}/{}'.format(DATA_DIR, FONT_FILE_PATH)
font = fm.FontProperties(fname=font_path);

"""## 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처"""
matplotlib.rcParams['axes.unicode_minus'] = False

# check = False
# ttf_path = os.path.join(pathlib.Path(matplotlib.matplotlib_fname()).parent, 'fonts\\ttf')
# ttf_list = os.listdir(ttf_path)
# for ttf in ttf_list:
#     if ttf.find(font) != -1:
#         check = True
#         break

# if not check:
#     shutil.copy(font_path, ttf_path)
#     fm.FontManager().addfont(font_path)

"""---------- ★★★ 전처리 ★★★ ----------"""
# # Todo: 기본 전처리, 형태소 분석, TF-IDF

# raw_data = make_preprocessing_corpus()
# preprocessing_corpus = raw_data['section'].to_list()

"""## 전처리된 corpus를 워드 클라우드로 시각화"""
# wordcloud = WordCloud(font_path).generate(' '.join(preprocessing_corpus))
        
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.savefig('{}/{}'.format(DATA_OUT_DIR, WORDCLOUD_FILE_PATH))


"""## IDF 분포도 확인"""
# tfidf = TfidfVectorizer(max_df = 0.2, min_df = 0.011, sublinear_tf=True).fit(preprocessing_corpus)
# tfidf_dict = {w: idf for w, idf in zip(tfidf.get_feature_names_out(), tfidf.idf_)}
# tfidf_dict = sorted(tfidf_dict.items(), key=operator.itemgetter(1))

# plt.figure(figsize=(12, 8))
# plt.hist(tfidf.idf_)
# plt.yscale('log')
# plt.xlabel('Number of scores')
# plt.ylabel('Number of Words')
# plt.show()

# print(len(tfidf_dict))
# print(tfidf_dict[:100])
# print()
# print(tfidf_dict[-100:])
# print()

"""## TF-IDF score Top 100 단어 시각화"""
# tfidf = TfidfVectorizer(max_features=100, max_df = 0.2, min_df = 0.011, sublinear_tf=True).fit(preprocessing_corpus)
    
# tfidf_dict = tfidf.get_feature_names_out()
# A_tfidf_sp = tfidf.transform(preprocessing_corpus) 
# data_array = A_tfidf_sp.toarray()

# tsne = TSNE(n_components=2, n_iter=10000, verbose=1)
# Z = tsne.fit_transform(data_array.T)

# plt.scatter(Z[:,0], Z[:,1])
# for i in range(len(tfidf_dict)):
#     plt.annotate(text = tfidf_dict[i].encode("utf8").decode("utf8"), xy = (Z[i,0], Z[i,1]), fontproperties=font)

# plt.draw()
# plt.savefig('{}/{}'.format(DATA_OUT_DIR, TF_TOP100_VISUALIZATION_FILE_PATH))

"""## TF-IDF를 이용하여 최종 전처리된 데이터 생성"""
# tfidf = TfidfVectorizer(max_df = 0.2, min_df = 0.011, sublinear_tf=True).fit(preprocessing_corpus)
# raw_data = make_checked_preprocessing_corpus(tfidf, preprocessing_corpus)
# checked_preprocessing_corpus = raw_data['section'].to_list()

"""## 최종 전처리된 corpus를 워드 클라우드로 시각화"""
# wordcloud = WordCloud(font_path).generate(' '.join(checked_preprocessing_corpus))
        
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.savefig('{}/{}'.format(DATA_OUT_DIR, 'checked_' + WORDCLOUD_FILE_PATH))

"""---------- ★★★ Word2Vec 모델 학습 및 생성 ★★★ ----------"""
# Todo: 전처리 과정을 통해 얻은 여러 문장을 Word2Vec 모델을 만들어 학습

# sentences = [cp.split(' ') for cp in checked_preprocessing_corpus]

# model = Word2Vec.load('{}/{}'.format(DATA_DIR, 'model/ko.bin'))  
# model.wv.save_word2vec_format('{}/{}'.format(DATA_DIR, 'model/ko.bin.gz', binary=False))

# model_2 = Word2Vec(size=200, min_count=1)
# model_2.build_vocab(sentences)
# total_examples = model_2.corpus_count

# model_2.build_vocab([list(model.wv.vocab.keys())], update=True)
# model_2.intersect_word2vec_format('{}/{}'.format(DATA_DIR, 'model/ko.bin.gz', binary=False))

# model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)
# model_2.save('{}/{}'.format(DATA_DIR, 'model/ko_new.bin'))

# w2v = Word2Vec.load('{}/{}'.format(DATA_DIR, 'model/ko_new.bin'))  

"""---------- ★★★ Clustering(K means) 모델 학습 및 생성 ★★★ ----------"""
# Todo: Word2Vec을 통해 문장을 하나로 압축하고 압축된 것을 통해 K mean을 이용하여 감정을 분류하고 그 결과로 감성사전을 구축

# label_list = ['증가', '유지', '감소']

# input_data = []
# print('-' * 100 + "\n ['증가', '유지', '감소'] 기준으로 유사도 계산 중...")
# pbar = tqdm(total=len(sentences))
# for sentence in sentences:
#     result = [0,0,0]
#     for word in sentence:
#         for i,l in enumerate(label_list):
#             result[i] += w2v.wv.similarity(l, word)

#     word_size = len(sentence)
#     result = [r/word_size for r in result]
    
#     input_data.append(result)
#     pbar.update(1)
# pbar.close()
# print('\n계산을 완료하였습니다.\n' + '-' * 100)

# scaler = StandardScaler()
# input_data = np.array(input_data)
# input_scaler =  scaler.fit_transform(input_data)

# Kmean = KMeans(n_clusters=3)
# Kmean.fit(input_scaler)

# classification_predict = Kmean.predict(input_scaler)

"""---------- ★★★ '여론 점수' 특성 생성 ★★★ ----------"""
# Todo: 감성사전 구축한 것을 통해 네이버 뉴스 기사를 바탕으로 여론 점수 구하기

# new_data = pd.DataFrame(columns=['일자', '여론점수'])
# date_list = set(raw_data['date'].to_list())

# score_list = np.zeros(shape = len(date_list))
# score_count = np.zeros(shape = len(date_list))

# print('-' * 100 + "\n 여론 점수 계산 중...")
# pbar = tqdm(total=len(classification_predict))
# for date, score in zip(raw_data['date'].to_list(), classification_predict):
#     for i, d in enumerate(date_list):
#         if d == date:
#             score_list[i]+= score
#             score_count[i]+=1
#             break
#     pbar.update(1)
# pbar.close()
# print("\n여론 점수 계산을 완료하였습니다.\n" + '-' * 100)

# score_list /= score_count
# score_list = list(score_list)

# for date, score in zip(date_list, score_list):
#     new_data = new_data.append({'일자': date.replace('.', '/', 2).replace('.', ''), '여론점수': score}, ignore_index=True)
    
# """## 여론 점수를 엑셀 형식으로 저장"""
# print('\n데이터프레임 형식으로 저장 중')
# new_data.to_excel('{}/{}'.format(DATA_OUT_DIR, SCORE_CORPUS_FILE_PATH), index=False, encoding='utf-8')
# print('\n저장을 완료하였습니다.\n')


"""---------- ★★★ 최종 모델 학습 및 예측 ★★★ ----------"""

df_stock = pd.read_csv('{}/{}'.format(DATA_IN_DIR, STOCK_DATA_FILE_PATH));
df_score = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, SCORE_CORPUS_FILE_PATH))

df = pd.merge(df_stock, df_score, how='left',on='일자')
# print(df.info());

sample_df = df[df['여론점수'].isnull()].drop(['여론점수'], axis=1, inplace=False)

score_input = df.dropna().drop(['여론점수'], axis=1, inplace=False)
score_target = df.dropna()['여론점수']

score_input.reset_index(drop=True, inplace=True)
score_target.reset_index(drop=True, inplace=True)

score_input['일자'] =  score_input['일자'].map(format_date_to_int);
sample_df['일자'] =  sample_df['일자'].map(format_date_to_int);

score_input = np.array(score_input)
score_target = np.array(score_target)
sample_input = np.array(sample_df)

"""## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
train_input, test_input, train_target, test_target = train_test_split(score_input, score_target, random_state=SEED);

"""## 특성 스케일링"""
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)
sample_scaled = scaler.transform(sample_input)

"""## 선형 회귀 모델"""
lr = LinearRegression();
lr.fit(train_scaled, train_target);

sample_df['여론점수'] = lr.predict(sample_scaled)

"""## 최종 모델 생성"""
stock_input = df.drop(['시가총액'], axis=1 ,inplace=False);
stock_target = df['시가총액'] / 1000000000000;

stock_input.reset_index(drop=True, inplace=True)
stock_target.reset_index(drop=True, inplace=True)

"""## str 형태인 날짜 값을 숫자형태로 변환"""
stock_input['일자'] =  stock_input['일자'].map(format_date_to_int);

"""## 예측된 여론 점수를 통합"""
stock_input['여론점수'].fillna(sample_df['여론점수'], inplace=True)

"""## 예측을 위해서 input 값의 첫번째 데이터 삭제"""
stock_input.drop(0, axis=0, inplace=True);

"""## 마찬가지로 예측을 위해서 target 값 인덱스를 한칸씩 이동"""
stock_target = [stock_target[i] for i in range(len(stock_target)-1)];

stock_input = np.array(stock_input)
stock_target = np.array(stock_target)

"""## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
train_input, test_input, train_target, test_target = train_test_split(stock_input, stock_target, random_state=SEED);

"""## 훈련 데이터와 검증 데이터를 20% 비율로 나눔"""
train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, random_state=SEED, test_size=0.2)

"""## 다음날 예측 값 설정"""
sample_input = df.iloc[0].drop('시가총액', axis= 0, inplace=False)
sample_input['일자'] = 20210927;

sample_input = np.array(sample_input, dtype='int64')
sample_input = sample_input.reshape(1,-1)

"""## 특성 스케일링"""
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
valid_scaled = scaler.transform(valid_input)
test_scaled = scaler.transform(test_input)
sample_scaled = scaler.transform(sample_input)

"""## 입력 데이터를 2차원 형식으로 변형"""
train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
valid_scaled = valid_scaled.reshape(valid_scaled.shape[0], valid_scaled.shape[1], 1)
test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)
sample_scaled = sample_scaled.reshape(1, -1, 1)

"""## 딥러닝 학습 전 기본적인 셋팅"""
tf.random.set_seed(SEED)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""## RNN 모델 part 1"""
rnn = Sequential([
    Conv1D(filters=10, kernel_size=2, kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=[11,1]),
    GRU(units=50,  activation='relu', return_sequences=True),
    GRU(units=100,  activation='relu', return_sequences=False),
    Dense(100),
    Dense(10),
    Dense(1)
])

fit_test_regression(rnn, train_scaled, train_target, valid_scaled, valid_target, test_scaled, test_target)

print(rnn.predict(sample_scaled))