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

"""## ??? ?????? ?????? ?????? ?????? ??????"""
font_path = "C:/Windows/Fonts/malgun.ttf";
font = font_manager.FontProperties(fname= font_path).get_name();
rc('font', family=font);

"""-------- ?????? ?????? ?????? --------"""

df = pd.read_csv('{}/{}'.format(DATA_IN_DIR, REGRESSION_FILE_PATH));
# print(df.info());

stock_input = df.drop(['????????????'], axis=1 ,inplace=False);
stock_target = df['????????????'] / 1000000000000;

"""## ????????? ????????? input ?????? ????????? ????????? ??????"""
stock_input.drop(0, axis=0, inplace=True);

"""## ??????????????? ????????? ????????? target ??? ???????????? ????????? ??????"""
stock_target = [stock_target[i] for i in range(len(stock_target)-1)];

"""## str ????????? ?????? ?????? ??????????????? ??????"""
stock_input['??????'] =  stock_input['??????'].map(format_date_to_int);

stock_input = np.array(stock_input)
stock_target = np.array(stock_target)

"""## ?????? ???????????? ????????? ???????????? 25% ????????? ??????"""
train_input, test_input, train_target, test_target = train_test_split(stock_input, stock_target, random_state=SEED);

"""## ?????? ???????????? ?????? ???????????? 20% ????????? ??????"""
train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, random_state=SEED, test_size=0.2)

"""## ????????? ?????? ??? ??????"""
sample_input = df.iloc[0].drop('????????????', axis= 0, inplace=False)
sample_input['??????'] = 20210927;

sample_input = np.array(sample_input, dtype='int64')
sample_input = sample_input.reshape(1,-1)

"""## ?????? ????????????"""
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
valid_scaled = scaler.transform(valid_input)
test_scaled = scaler.transform(test_input)
sample_scaled = scaler.transform(sample_input)

"""## ????????? ?????? ??? ???????????? ??????"""
tf.random.set_seed(SEED)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""DNN ?????? part 1"""
# dnn = Sequential([
#     Dense(100, activation="relu", input_shape=[10]),
#     Dense(10, activation="relu"),
#     Dense(10, activation="relu"),
#     Dense(1)
# ])

"""DNN ?????? part 2"""
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

"""## ?????? ???????????? 2?????? ???????????? ??????"""
train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], 1)
valid_scaled = valid_scaled.reshape(valid_scaled.shape[0], valid_scaled.shape[1], 1)
test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], 1)

"""RNN ?????? part 1"""
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

"""RNN ?????? part 2"""
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

"""RNN ?????? part 3"""
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

"""-------- ?????? ?????? ?????? --------"""

data_frame = pd.read_csv('{}/{}'.format(DATA_IN_DIR, CLASSIFICATION_FILE_PATH), header=0, sep=',')
# print(data_frame.info());

fish_total_count = len(data_frame)
# print('????????? ??? ??????: {}'.format(fish_total_count))

species_list = data_frame['Species'].unique()
# print('?????? ??????: {}'.format(species_list))

"""## ?????? ????????? ?????? ?????? ?????? ??? ??????"""
fish_input = data_frame.drop('Species', axis=1, inplace=False);
# print(fish_input);

fish_target = data_frame['Species'];
# print(fish_target);

"""## ????????? ?????? ?????? ??????"""
fish_dict = {sp:i for sp,i in zip(fish_target.unique(),range(len(fish_target.unique())))};
# print(fish_dict);

"""## ????????? ????????? ?????? ????????? ?????? ?????? ????????? ??????"""
fish_target = fish_target.map(fish_dict);
# print(fish_target);

"""## ?????? ???????????? ????????? ???????????? 25% ????????? ??????"""
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, stratify=fish_target, random_state=SEED)

"""## ?????? ???????????? ?????? ???????????? 20% ????????? ??????"""
train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, stratify=train_target, random_state=SEED, test_size=0.2)

"""## ????????? ???????????? ???????????? ????????? ??????"""
train_numpy = train_input.to_numpy()
valid_numpy = valid_input.to_numpy()
test_numpy = test_input.to_numpy()

"""## one-hot ??????????????? ??????"""
train_target = to_categorical(train_target)
valid_target = to_categorical(valid_target)
test_target = to_categorical(test_target)

"""## ????????? ?????? ??? ???????????? ??????"""
tf.random.set_seed(SEED)
DefaultConv1D = partial(Conv1D, kernel_size=2, activation='relu', kernel_initializer='he_uniform', padding='same')
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

"""## DNN ?????? part 1"""
# dnn = Sequential([
#     Dense(300, activation="tanh", input_shape=[6]),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN ?????? part 2"""
# dnn = Sequential([
#     Dense(120, activation="tanh", input_shape=[6]),
#     Dense(60, activation="tanh"),
#     Dense(60, activation="tanh"),
#     Dense(60, activation="tanh"),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN ?????? part 3"""
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

"""## DNN ?????? part 4"""
# dnn = Sequential([
#     Dense(512, activation="tanh", input_shape=[6]),
#     Dense(7, activation="softmax")
# ])

# fit_test_classification(dnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## DNN ?????? part 5"""
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

"""## ?????? ???????????? ?????? ?????? ??????"""
train_input = train_numpy.reshape(train_numpy.shape[0], train_numpy.shape[1], 1)
valid_input = valid_numpy.reshape(valid_numpy.shape[0], valid_numpy.shape[1], 1)
test_input = test_numpy.reshape(test_numpy.shape[0], test_numpy.shape[1], 1)

""" ## CNN ?????? part 1"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN ?????? part 2"""
# cnn = Sequential([
#     DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(units=14, activation='relu'),
#     Dense(units=7, activation='softmax'),
# ])

# fit_test_classification(cnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## CNN ?????? part 3"""
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

"""## CNN ?????? part 4"""
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

"""## CNN ?????? part 5"""
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

"""## RNN ?????? part1"""
# rnn=Sequential([
    # layers.SimpleRNN(units=48,return_sequences=True,input_shape=[6,1]),
    # layers.SimpleRNN(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN ?????? part2"""
# rnn=Sequential([
    # layers.SimpleRNN(units=48,return_sequences=True,input_shape=[None,1]),
    # layers.SimpleRNN(units=24),
    # layers.Dense(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN ?????? part3"""
# rnn=Sequential([
    # layers.LSTM(units=48,return_sequences=True,input_shape=[6,1]),
    # layers.LSTM(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)

"""## RNN ?????? part4"""
# rnn=Sequential([
    # DefaultConv1D(filters=32, kernel_size=3, input_shape=(6,1)),
    # layers.GRU(units=48,return_sequences=True),
    # layers.GRU(units=7,activation='softmax')
# ])

# fit_test_classification(rnn, train_input, train_target, valid_input, valid_target, test_input, test_target)