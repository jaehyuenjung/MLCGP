# -*- coding: utf-8 -*- 
import re
import time
import pickle
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from konlpy.tag import Kkma
from hanspell import spell_checker
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from matplotlib import font_manager, rc
from soynlp.noun import LRNounExtractor_v2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = './nbsr';
DATA_IN_DIR = './nbsr/data/in';
DATA_OUT_DIR = './nbsr/data/out';
COMBINED_SCORE_FILE_PATH = 'combined_score.json';
STOP_WORD_DATA_FILE_PATH= 'stop_words/korean_stop_words.csv';
STOCK_DATA_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';
CRAWKLING_DATA_FILE_PATH = 'news_data_collection.xlsx';
TF_DATA_FILE_PATH = 'tf_data.bin'
SEED = 346672;

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

def korean_spell_check(corpus):
    kkma = Kkma()
    sentence_list = kkma.sentences(corpus)

    checked_sentence = []
    
    r = re.compile(r"[!\"#$%&'()*+,/;<=>?@[\]^`{|}~.]")
    for sentence in sentence_list:
        sentence = r.sub('', sentence)
        
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
    
def make_tf_data(file_name):
    tfidfv = None
    if os.path.isfile(file_name):
        with open(file_name, "rb") as fr:
            tfidfv = pickle.load(fr)
        return tfidfv
    else:
        df = pd.read_excel('{}/{}'.format(DATA_OUT_DIR, CRAWKLING_DATA_FILE_PATH));

        corpus = df.dropna()['section'].to_list()

        # """## 문장 맞춤법 체크"""
        # pbar = tqdm(total=len(corpus))
        # for cp in corpus:
        #     cp = korean_spell_check(cp)
        #     pbar.update(1)
        #     time.sleep(0.3)
            
        # pbar.close()

        """## 종합 점수 통계치 생성"""
        combined_scores = make_combined_scores(corpus, '{}/{}'.format(DATA_OUT_DIR, COMBINED_SCORE_FILE_PATH))

        """## L parts(명사/동사/형용사/부사) 나머지 부분이 R parts로 구분하는 Tokenizer"""
        tokenizer = LTokenizer(scores=combined_scores)

        pretreatmented_corpus = []
        for cp in corpus:
            lr_list = tokenizer.tokenize(cp, flatten=False)
            
            l_list = []
            for l, r in lr_list:
                l_list.append(l)
            pretreatmented_corpus.append(' '.join(l_list))

        df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, STOP_WORD_DATA_FILE_PATH));

        tfidfv = TfidfVectorizer(stop_words=df['용어'].to_list()).fit(pretreatmented_corpus)

        with open(file_name, "wb") as fw:
                pickle.dump(tfidfv, fw)
                
        return tfidfv

# def fit_test_regression(model, train_input, train_target, valid_input, valid_target, test_input, test_target):
#     model.compile(loss="mae", optimizer="adam")

#     history = model.fit(train_input, train_target, epochs=1000, validation_data=(valid_input, valid_target),
#     callbacks=[early_stopping_cb])

#     train_score = model.evaluate(train_input, train_target)
#     test_score = model.evaluate(test_input, test_target)
#     print(train_score)
#     print(test_score)

#     pd.DataFrame(history.history).plot()
#     plt.grid(True)
#     plt.show()

"""## 맷 플롯 한글 깨짐 방지 설정"""
font_path = "C:/Windows/Fonts/malgun.ttf";
font = font_manager.FontProperties(fname= font_path).get_name();
rc('font', family=font);

# """---------- ★★★ 전처리 ★★★ ----------"""
# # Todo: 기본 전처리, 형태소 분석, TF-IDF

make_tf_data('{}/{}'.format(DATA_OUT_DIR, TF_DATA_FILE_PATH))

"""---------- ★★★ Word2Vec 모델 학습 및 생성 ★★★ ----------"""
# Todo: 전처리 과정을 통해 얻은 여러 문장을 Word2Vec 모델을 만들어 학습

"""---------- ★★★ Clustering(K means) 모델 학습 및 생성 ★★★ ----------"""
# Todo: Word2Vec을 통해 문장을 하나로 압축하고 압축된 것을 통해 K mean을 이용하여 감정을 분류하고 그 결과로 감성사전을 구축

"""---------- ★★★ '여론 점수' 특성 생성 ★★★ ----------"""
# Todo: 감성사전 구축한 것을 통해 네이버 뉴스 기사를 바탕으로 여론 점수 구하기

"""---------- ★★★ 최종 모델 학습 및 예측 ★★★ ----------"""

# df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, STOCK_DATA_FILE_PATH));
# # print(df.info());

# stock_input = df.drop(['시가총액'], axis=1 ,inplace=False);
# stock_target = df['시가총액'] / 1000000000000;

# """## 예측을 위해서 input 값의 첫번째 데이터 삭제"""
# stock_input.drop(0, axis=0, inplace=True);

# """## 마찬가지로 예측을 위해서 target 값 인덱스를 한칸씩 이동"""
# stock_target = [stock_target[i] for i in range(len(stock_target)-1)];

# """## str 형태인 날짜 값을 숫자형태로 변환"""
# stock_input['일자'] =  stock_input['일자'].map(format_date_to_int);

# stock_input = np.array(stock_input)
# stock_target = np.array(stock_target)

# """## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
# train_input, test_input, train_target, test_target = train_test_split(stock_input, stock_target, random_state=SEED);

# """## 훈련 데이터와 검증 데이터를 20% 비율로 나눔"""
# train_input, valid_input, train_target, valid_target = train_test_split(
#     train_input, train_target, random_state=SEED, test_size=0.2)

# """## 다음날 예측 값 설정"""
# sample_input = df.iloc[0].drop('시가총액', axis= 0, inplace=False)
# sample_input['일자'] = 20210927;

# sample_input = np.array(sample_input, dtype='int64')
# sample_input = sample_input.reshape(1,-1)

# """## 특성 스케일링"""
# scaler = StandardScaler()
# train_scaled = scaler.fit_transform(train_input)
# valid_scaled = scaler.transform(valid_input)
# test_scaled = scaler.transform(test_input)
# sample_scaled = scaler.transform(sample_input)

# """## 딥러닝 학습 전 기본적인 셋팅"""
# tf.random.set_seed(SEED)
# early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

# """## 입력 데이터를 2차원 형식으로 변형"""
# train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
# valid_scaled = valid_scaled.reshape(valid_scaled.shape[0], valid_scaled.shape[1], 1)
# test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)