import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_absolute_error;

DATA_DIR = './data/stock_ko';
FILE_PATH = 'samsung_20000101-20210926.csv';

def format_date_to_int(date):
    year, month, day = date.split('/');
    return int('{}{}{}'.format(year, month, day));

def draw_linear_regression(model, train_input, df, limit = 0):
    x = df['일자'].drop(0, axis=0, inplace=False);
    y = df['시가총액'].drop(0, axis=0, inplace=False);

    x_r = np.flip(x, axis=0);
    y_r = np.flip(y, axis=0);

    y_predict = [];

    for inputs in train_input:
        ci = inputs.copy().reshape(1,-1);
        y_predict.append(model.predict(ci));

    y_predict = np.flip(np.array(y_predict), axis=0);

    if limit:
        x_r = x_r[-(limit):];
        y_r = y_r[-(limit):];
        y_predict = y_predict[-(limit):, :];

    plt.plot(x_r,y_r, label='실제 값');
    plt.plot(x_r,y_predict, label='예측 값');
    plt.xlabel('일자');
    plt.ylabel('시가총액(단위:원)');
    plt.legend();
    plt.show();

def fit_test_poly(model, train_input, train_target, test_input, test_target, start = 3, stop = 4, step = 1):
    train_score = [];
    test_score = [];

    for i in range(start,stop, step):
        poly = PolynomialFeatures(degree = i, include_bias=False);
        poly.fit(train_input);
        train_poly = poly.transform(train_input);
        test_poly = poly.transform(test_input);

        model.fit(train_poly, train_target);

        train_score.append(model.score(train_poly, train_target));
        test_score.append(model.score(test_poly, test_target));

    plt.plot(np.arange(start, stop, step), train_score);
    plt.plot(np.arange(start, stop, step), test_score);
    plt.xlabel('degree');
    plt.ylabel('R^2');
    plt.show();

   
df = pd.read_csv('{}/{}'.format(DATA_DIR, FILE_PATH));

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

"""## 맷 플롯 한글 깨짐 방지 설정"""
font_path = "C:/Windows/Fonts/malgun.ttf";
font = font_manager.FontProperties(fname= font_path).get_name();
rc('font', family=font);

"""## 다중 회귀 모델 part 1"""
lr = LinearRegression();
lr.fit(train_input, train_target);
# print(lr.score(train_input, train_target));
# print(lr.score(test_input, test_target));

# test_prediction = lr.predict(test_input);

# mae = mean_absolute_error(test_target, test_prediction);
# print(mae);

"## 과대 적합을 해결하기 위해 특성값을 추가"
poly = PolynomialFeatures(include_bias=False);
poly.fit(train_input);
train_poly = poly.transform(train_input);
test_poly = poly.transform(test_input);
# print(train_poly.shape);

# poly.get_feature_names();

"""## 다중 회귀 모델 part2"""
lr = LinearRegression();
lr.fit(train_poly, train_target);
# print(lr.score(train_poly, train_target));
# print(lr.score(test_poly, test_target));

# test_prediction = lr.predict(test_poly);

# mae = mean_absolute_error(test_target, test_prediction);
# print(mae);

poly.fit(stock_input);
stock_poly = poly.transform(stock_input);
draw_linear_regression(lr, stock_poly, df, 5);

"""## 과대 적합을 해결할 적절한 degree 값을 찾기 위해 테스트"""
# fit_test_poly(lr, train_input, train_target, test_input, test_target, 2, 4);


"""## 다음날 시가총액 예측"""
# sample_input = df.iloc[0].drop('시가총액', axis= 0, inplace=False);
# sample_input['일자'] = 20210927;

# sample_input = sample_input.to_numpy();
# sample_input = sample_input.reshape(1,-1);

# poly = PolynomialFeatures(include_bias=False);
# poly.fit(sample_input);
# sample_poly = poly.transform(sample_input);

# print(lr.predict(sample_poly));

