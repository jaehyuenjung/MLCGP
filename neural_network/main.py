import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split

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
train_input, test_input, train_target, test_target = train_test_split(stock_input, stock_target, random_state=346672);

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

