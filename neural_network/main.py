import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from matplotlib import font_manager, rc
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Dropout, SimpleRNN, LSTM, GRU

DATA_DIR = './neural_network';
DATA_IN_DIR = './neural_network/data/in';
REGRESSION_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';
CLASSIFICATION_FILE_PATH = 'fish/Fish.csv';

SEED = 346672;

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

def format_data_to_scaling(train_input, valid_input, test_input):
    mean = np.mean(train_input, axis=0)
    std = np.std(train_input, axis=0)

    train_scaled = (train_input - mean) / std
    valid_scaled = (valid_input - mean) / std
    test_scaled = (test_input - mean) / std

    return train_scaled, valid_scaled, test_scaled

def fit_test_classification(model, train_input, train_target, valid_input, valid_target, test_input, test_target):
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    history = model.fit(train_input, train_target, epochs=1000, validation_data=(valid_input, valid_target),
    callbacks=[early_stopping_cb])

    train_score = model.evaluate(train_input, train_target)
    test_score = model.evaluate(test_input, test_target)
    print(train_score)
    print(test_score)

    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

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

"""-------- ★★ 회귀 ★★ --------"""

df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, REGRESSION_FILE_PATH));
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

"""DNN 모델 part 1"""
# dnn = Sequential([
#     Dense(100, activation="relu", input_shape=[10]),
#     Dense(10, activation="relu"),
#     Dense(10, activation="relu"),
#     Dense(1)
# ])

"""DNN 모델 part 2"""
# dnn = Sequential([
#     Dense(100, activation="relu", input_shape=[10]),
#     Dense(10, activation="relu"),
#     Dense(50, activation="relu"),
#     Dense(10, activation="relu"),
#     Dense(50, activation="relu"),
#     Dense(10, activation="relu"),
#     Dense(1)
# ])

# fit_test_regression(dnn, train_scaled, train_target, valid_scaled, valid_target, test_scaled, test_target)

# print(dnn.predict(sample_scaled))

"""## 입력 데이터를 2차원 형식으로 변형"""
train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
valid_scaled = valid_scaled.reshape(valid_scaled.shape[0], valid_scaled.shape[1], 1)
test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)

"""RNN 모델 part 1"""
# rnn = Sequential([
#     Conv1D(filters=10, kernel_size=2, kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=[10,1]),
#     LSTM(units=50,  activation='relu', return_sequences=True),
#     LSTM(units=100,  activation='relu', return_sequences=False),
#     Dense(100),
#     Dense(10),
#     Dense(1)
# ])

# fit_test_regression(rnn, train_scaled, train_target, valid_scaled, valid_target, test_scaled, test_target)

# sample_scaled = sample_scaled.reshape(1, 10, 1)
# print(rnn.predict(sample_scaled))

"""RNN 모델 part 2"""
# rnn = Sequential([
#     LSTM(units=50,  activation='relu', return_sequences=True, input_shape=[10,1]),
#     LSTM(units=100,  activation='relu', return_sequences=False),
#     Dense(100),
#     Dense(10),
#     Dense(1)
# ])

# fit_test_regression(rnn, train_scaled, train_target, valid_scaled, valid_target, test_scaled, test_target)

# sample_scaled = sample_scaled.reshape(1, 10, 1)
# print(rnn.predict(sample_scaled))

"""RNN 모델 part 3"""
# rnn = Sequential([
    # Conv1D(filters=10, kernel_size=2, kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=[10,1]),
    # GRU(units=50,  activation='relu', return_sequences=True),
    # GRU(units=100,  activation='relu', return_sequences=False),
    # Dense(100),
    # Dense(10),
    # Dense(1)
# ])

# fit_test_regression(rnn, train_scaled, train_target, valid_scaled, valid_target, test_scaled, test_target)

# sample_scaled = sample_scaled.reshape(1, 10, 1)
# print(rnn.predict(sample_scaled))

"""-------- ★★ 분류 ★★ --------"""

data_frame = pd.read_csv('{}/{}'.format(DATA_IN_DIR, CLASSIFICATION_FILE_PATH), header=0, sep=',')
# print(data_frame.info());

fish_total_count = len(data_frame)
# print('데이터 셋 크기: {}'.format(fish_total_count))

species_list = data_frame['Species'].unique()
# print('생선 종류: {}'.format(species_list))

"""## 다중 분류를 하기 위해 라벨 값 분류"""
fish_input = data_frame.drop('Species', axis=1, inplace=False);
# print(fish_input);

fish_target = data_frame['Species'];
# print(fish_target);

"""## 라벨에 대한 사전 생성"""
fish_dict = {sp:i for sp,i in zip(fish_target.unique(),range(len(fish_target.unique())))};
# print(fish_dict);

"""## 사전을 이용해 해당 라벨에 대한 고유 인덱스 설정"""
fish_target = fish_target.map(fish_dict);
# print(fish_target);

"""## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, stratify=fish_target, random_state=SEED)

"""## 훈련 데이터와 검증 데이터를 20% 비율로 나눔"""
train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, stratify=train_target, random_state=SEED, test_size=0.2)

"""## 데이터 프레임을 넘파이로 배열로 변경"""
train_numpy = train_input.to_numpy()
valid_numpy = valid_input.to_numpy()
test_numpy = test_input.to_numpy()

"""## one-hot 인코딩으로 변환"""
train_target = to_categorical(train_target)
valid_target = to_categorical(valid_target)
test_target = to_categorical(test_target)

"""## 딥러닝 학습 전 기본적인 셋팅"""
tf.random.set_seed(SEED)
DefaultConv1D = partial(Conv1D, kernel_size=2, activation='relu', kernel_initializer='he_uniform', padding='same')
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""## DNN 모델 part 1"""
# dnn = Sequential([
#     Dense(300, activation="tanh", input_shape=[6]),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN 모델 part 2"""
# dnn = Sequential([
#     Dense(120, activation="tanh", input_shape=[6]),
#     Dense(60, activation="tanh"),
#     Dense(60, activation="tanh"),
#     Dense(60, activation="tanh"),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN 모델 part 3"""
# dnn = Sequential([
#     Dense(256, activation="tanh", input_shape=[6]),
#     Dense(256, activation="tanh"),
#     Dropout(0.2),
#     Dense(128, activation="tanh"),
#     Dropout(0.2),
#     Dense(64, activation="tanh"),
#     Dropout(0.2),
#     Dense(64, activation="tanh"),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN 모델 part 4"""
# dnn = Sequential([
#     Dense(512, activation="tanh", input_shape=[6]),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN 모델 part 5"""
# dnn = Sequential([
#     Dense(60, activation="tanh", input_shape=[6]),
#     Dropout(0.2),
#     Dense(30, activation="tanh"),
#     Dropout(0.3),
#     Dense(30, activation="tanh"),
#     Dense(30, activation="tanh"),
#     Dense(30, activation="tanh"),
#     Dense(30, activation="tanh"),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## 입력 데이터에 맞게 형식 변형"""
train_input = train_numpy.reshape(train_numpy.shape[0], train_numpy.shape[1], 1)
valid_input = valid_numpy.reshape(valid_numpy.shape[0], valid_numpy.shape[1], 1)
test_input = test_numpy.reshape(test_numpy.shape[0], test_numpy.shape[1], 1)

""" ## CNN 모델 part 1"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN 모델 part 2"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN 모델 part 3"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     DefaultConv1D(filters=64),
#     MaxPooling1D(pool_size=2),
#     DefaultConv1D(filters=16),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN 모델 part 4"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     DefaultConv1D(filters=64),
#     MaxPooling1D(pool_size=2),
#     DefaultConv1D(filters=16),
#     DefaultConv1D(filters=8),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN 모델 part 5"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     DefaultConv1D(filters=64),
#     AveragePooling1D(pool_size=2),
#     DefaultConv1D(filters=16),
#     DefaultConv1D(filters=8),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN 분류 part1"""
# rnn=Sequential([
    # layers.SimpleRNN(units=48,return_sequences=True,input_shape=[6,1]),
    # layers.SimpleRNN(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN 분류 part2"""
# rnn=Sequential([
    # layers.SimpleRNN(units=48,return_sequences=True,input_shape=[None,1]),
    # layers.SimpleRNN(units=24),
    # layers.Dense(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN 분류 part3"""
# rnn=Sequential([
    # layers.LSTM(units=48,return_sequences=True,input_shape=[6,1]),
    # layers.LSTM(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN 분류 part4"""
# rnn=Sequential([
    # DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
    # layers.GRU(units=48,return_sequences=True),
    # layers.GRU(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)