import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from matplotlib import font_manager, rc
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Dropout

DATA_DIR = './neural_network';
DATA_IN_DIR = './neural_network/data/in';
REGRESSION_FILE_PATH = 'stock_ko/samsung_20000101-20210926.csv';
CLASSIFICATION_FILE_PATH = 'fish/Fish.csv';

SEED = 346672;

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

def format_data_to_scaling(train_input, test_input):
    mean = np.mean(train_input, axis=0)
    std = np.std(train_input, axis=0)

    train_scaled = (train_input - mean) / std
    test_scaled = (test_input - mean) / std

    return train_scaled, test_scaled

def fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target):
    cnn.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    history = cnn.fit(train_input, train_target, epochs=1000, validation_data=(valid_input, valid_target),
    callbacks=[early_stopping_cb])

    train_score = cnn.evaluate(train_input, train_target)
    test_score = cnn.evaluate(test_input, test_target)
    print(train_score)
    print(test_score)

    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

"""## 맷 플롯 한글 깨짐 방지 설정"""
font_path = "C:/Windows/Fonts/malgun.ttf";
font = font_manager.FontProperties(fname= font_path).get_name();
rc('font', family=font);

"""-------- ★★ 회귀 ★★ --------"""

df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, REGRESSION_FILE_PATH));
# print(df.info());

stock_input = df.drop('시가총액', axis=1 ,inplace=False);
stock_target = df['시가총액'];

"""## 예측을 위해서 input 값의 첫번째 데이터 삭제"""
stock_input.drop(0, axis=0, inplace=True);

"""## 마찬가지로 예측을 위해서 target 값 인덱스를 한칸씩 이동"""
stock_target = [stock_target[i] for i in range(len(stock_target)-1)];

"""## str 형태인 날짜 값을 숫자형태로 변환"""
stock_input['일자'] =  stock_input['일자'].map(format_date_to_int);

"""## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
train_input, test_input, train_target, test_target = train_test_split(stock_input, stock_target, random_state=SEED);

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

"""## 입력 데이터를 2차원 형식으로 변형"""
train_input = train_numpy.reshape(train_numpy.shape[0], train_numpy.shape[1], 1)
valid_input = valid_numpy.reshape(valid_numpy.shape[0], valid_numpy.shape[1], 1)
test_input = test_numpy.reshape(test_numpy.shape[0], test_numpy.shape[1], 1)

"""## one-hot 인코딩으로 변환"""
train_target = to_categorical(train_target)
valid_target = to_categorical(valid_target)
test_target = to_categorical(test_target)

tf.random.set_seed(SEED)
DefaultConv1D = partial(Conv1D, kernel_size=2, activation='relu', kernel_initializer='he_uniform', padding='same')
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""
    ## CNN 모델 part 1
    [1.0799344778060913, 0.5789473652839661]
    [1.048640251159668, 0.550000011920929]
"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""
    ## CNN 모델 part 2
    [0.5580806732177734, 0.821052610874176]
    [0.6378886103630066, 0.7749999761581421]
"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""
    ## CNN 모델 part 3
    [0.060844190418720245, 0.9789473414421082]
    [0.05038032680749893, 1.0]
"""
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

# fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""
    ## CNN 모델 part 4
    [0.20359115302562714, 0.9578947424888611]
    [0.18772734701633453, 0.949999988079071]
"""
cnn = Sequential([
    DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
    DefaultConv1D(filters=64),
    MaxPooling1D(pool_size=2),
    DefaultConv1D(filters=16),
    DefaultConv1D(filters=8),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=14, activation='relu'),
    Dense(units=7, activation='softmax'),
])

fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""
    ## CNN 모델 part 5
    [0.10862832516431808, 0.9789473414421082]
    [0.11227522045373917, 0.9750000238418579]
"""
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

# fit_test_cnn(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)








