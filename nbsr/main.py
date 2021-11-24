import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

DATA_DIR = './nbsr';
DATA_IN_DIR = './nbsr/data/in';
DATA_OUT_DIR = './nbsr/data/out';
STOCK_DATA_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';
CRAWKLING_DATA_FILE_PATH = '';
SEED = 346672;

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

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
font_path = "C:/Windows/Fonts/malgun.ttf";
font = font_manager.FontProperties(fname= font_path).get_name();
rc('font', family=font);

"""---------- ★★★ 전처리 ★★★ ----------"""
# Todo: 기본 전처리, 형태소 분석, TF-IDF

"""---------- ★★★ Word2Vec 모델 학습 및 생성 ★★★ ----------"""
# Todo: 전처리 과정을 통해 얻은 여러 문장을 Word2Vec 모델을 만들어 학습

"""---------- ★★★ Clustering(K means) 모델 학습 및 생성 ★★★ ----------"""
# Todo: Word2Vec을 통해 문장을 하나로 압축하고 압축된 것을 통해 K mean을 이용하여 감정을 분류하고 그 결과로 감성사전을 구축

"""---------- ★★★ '여론 점수' 특성 생성 ★★★ ----------"""
# Todo: 감성사전 구축한 것을 통해 네이버 뉴스 기사를 바탕으로 여론 점수 구하기

"""---------- ★★★ 최종 모델 학습 및 예측 ★★★ ----------"""

df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, STOCK_DATA_FILE_PATH));
# print(df.info());

stock_input = df.drop(['시가총액'], axis=1 ,inplace=False);
stock_target = df['시가총액'] / 1000000000000;

"""## 예측을 위해서 input 값의 첫번째 데이터 삭제"""
stock_input.drop(0, axis=0, inplace=True);

"""## 마찬가지로 예측을 위해서 target 값 인덱스를 한칸씩 이동"""
stock_target = [stock_target[i] for i in range(len(stock_target)-1)];

"""## str 형태인 날짜 값을 숫자형태로 변환"""
stock_input['일자'] =  stock_input['일자'].map(format_date_to_int);

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

"""## 딥러닝 학습 전 기본적인 셋팅"""
tf.random.set_seed(SEED)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""## 입력 데이터를 2차원 형식으로 변형"""
train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
valid_scaled = valid_scaled.reshape(valid_scaled.shape[0], valid_scaled.shape[1], 1)
test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)