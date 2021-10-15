import pydot
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

DATA_DIR = './classification';
DATA_IN_DIR = './classification/data/in';
FILE_PATH = 'fish/Fish.csv';

data_colors = ['blue', 'red']

SEED = 346672;

def format_data_to_scaling(train_input, test_input):
    mean = np.mean(train_input, axis=0)
    std = np.std(train_input, axis=0)

    train_scaled = (train_input - mean) / std
    test_scaled = (test_input - mean) / std

    return train_scaled, test_scaled

def get_colors(y):
    return [data_colors[label] for label in y]

def plot_decision_function(X_train, y_train, X_test, y_test, clf):
    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(121)
    plt.title("Training data")
    plot_decision_function_helper(X_train, y_train, clf)
    plt.subplot(122)
    plt.title("Test data")
    plot_decision_function_helper(X_test, y_test, clf, True)
    plt.show()

def plot_decision_function_helper(X, y, clf, show_only_decision_function = False):
    colors = get_colors(y)
    plt.axis('equal')
    plt.tight_layout()

    plt.scatter(X[:, 0], X[:, 1], c= colors, s=20, edgecolors=colors)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    if  show_only_decision_function:
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
                 linestyles='-')
    else :
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                 linestyles=['--','-','--'])

def fit_test_regulation(train_input, train_target, test_input, test_target, start = 1, stop = 4, step = 1):
    
    """## i: max_depth, j: min_samples_leaf"""
    x_values = [(i,j)for i in range(start, stop, step) for j in range(start, stop, step)];

    train_score = [];
    test_score = [];

    for i,j in x_values:
        classifier = DecisionTreeClassifier(max_depth=i, min_samples_leaf=j, random_state=SEED)
        classifier.fit(train_input, train_target);

        train_score.append(classifier.score(train_input, train_target));
        test_score.append(classifier.score(test_input, test_target));

    x_labels = ["({},{})".format(i,j) for i,j in x_values];

    plt.figure(figsize=(12,8));
    plt.plot(x_labels, train_score);
    plt.plot(x_labels, test_score);
    plt.xlabel('(max_depth, min_samples_leaf)');
    plt.ylabel('accuracy');
    plt.xticks(rotation=45)
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

data_frame = pd.read_csv('{}/{}'.format(DATA_IN_DIR, FILE_PATH), header=0, sep=',')
# print(data_frame.info());

"""-------- ★★ 이진 분류 ★★ --------"""

fish_total_count = len(data_frame)
# print('데이터 셋 크기: {}'.format(fish_total_count))

species_list = data_frame['Species'].unique()
# print('생선 종류: {}'.format(species_list))

"""## 전체 비중에서 각 물고기 종류가 몇 퍼센트인지 확인"""
fish_species_ratio = [round((data_frame['Species'].value_counts()[sp]/fish_total_count*100),1) for sp in species_list]

# plt.title('fish_species_ratio')
# plt.pie(fish_species_ratio, labels=species_list, autopct='%.1f%%')
# plt.show()

"""## 생선 종류별 무게와 길이 평균 그래프 작성"""
fish_weight_avg = [np.average(data_frame[data_frame['Species']==sp][['Weight']].values) for sp in species_list]
fish_length_avg = [np.average(data_frame[data_frame['Species']==sp][['Length2']].values) for sp in species_list]

# plt.title('fish_weight_avg')
# plt.bar(np.arange(len(species_list)), fish_weight_avg)
# plt.xticks(np.arange(len(species_list)), species_list)
# plt.show()

# plt.title('fish_length_avg')
# plt.bar(np.arange(len(species_list)), fish_length_avg)
# plt.xticks(np.arange(len(species_list)), species_list)
# plt.show()

"""## 2진 분류를 하기 위해 pike와 smelt 데이터를 각각 만듬"""
pike_weight = data_frame[data_frame['Species']=='Pike'][['Weight']].values
pike_length = data_frame[data_frame['Species']=='Pike'][['Length2']].values
pike_size = len(pike_weight)

smelt_weight = data_frame[data_frame['Species']=='Smelt'][['Weight']].values
smelt_length = data_frame[data_frame['Species']=='Smelt'][['Length2']].values
smelt_size = len(smelt_weight)

"""## pike와 smelt 데이터를 하나로 합치고 라벨 생성"""
fish_input = np.column_stack((np.concatenate((pike_length, smelt_length))
                           , np.concatenate((pike_weight, smelt_weight))))
fish_target = np.concatenate((np.ones(pike_size, dtype=np.int64), np.zeros(smelt_size, dtype=np.int64)))
# print(fish_input[:5])
# print(fish_target)

"""## 훈련 데이터와 테스트 데이터를 25% 비율로 나눔"""
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target,stratify=fish_target, random_state=SEED)
# print(train_input[:5])
# print(train_target)

"""## 무게와 길이의 스케일링 차이로 인해 잘못된 결과를 방지"""
pike_index = [i for i,v in enumerate(train_target) if v==1]
smelt_index = [i for i,v in enumerate(train_target) if v==0]

# plt.scatter(train_input[pike_index,0], train_input[pike_index,1], color="red")
# plt.scatter(train_input[smelt_index,0], train_input[smelt_index,1], color="blue")
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

train_scaled, test_scaled = format_data_to_scaling(train_input, test_input)

# plt.scatter(train_scaled[pike_index,0], train_scaled[pike_index,1], color="red")
# plt.scatter(train_scaled[smelt_index,0], train_scaled[smelt_index,1], color="blue")
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

"""## 이진 분류 SVC 모델"""
classifier = SVC(kernel='linear')
classifier.fit(train_scaled, train_target)
# print(classifier.score(train_scaled, train_target))
# print(classifier.score(test_scaled, test_target))

# plot_decision_function(train_scaled, train_target, test_scaled, test_target, classifier)

"""-------- ★★ 다중 분류 ★★ --------"""

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

"""## 결정 트리 모델 part 1"""
classifier = DecisionTreeClassifier(random_state=SEED);
classifier.fit(train_input, train_target);
# print(classifier.score(train_input, train_target));
# print(classifier.score(test_input, test_target));

# export_graphviz(classifier, out_file="{}/dicisionTree1.dot".format(DATA_OUT_DIR), class_names=list(fish_dict.keys()),
#                 feature_names=fish_input.columns, filled=True)

# (graph,) = pydot.graph_from_dot_file("{}/dicisionTree1.dot".format(DATA_OUT_DIR), encoding='utf8')

# graph.write_png('{}/dicisionTree1.png'.format(DATA_OUT_DIR))

"""## 적절한 규제값인 max_depth와 min_samples_leaf을 찾기 위해 테스트"""
# fit_test_regulation(train_input, train_target, test_input, test_target, 1, 10, 3);

"""## 결정 트리 모델 part 2"""
classifier = DecisionTreeClassifier(random_state=SEED, max_depth=7, min_samples_leaf=6);
classifier.fit(train_input, train_target);
# print(classifier.score(train_input, train_target));
# print(classifier.score(test_input, test_target));

# export_graphviz(classifier, out_file="{}/dicisionTree2.dot".format(DATA_OUT_DIR), class_names=list(fish_dict.keys()),
#                 feature_names=fish_input.columns, filled=True)

# (graph,) = pydot.graph_from_dot_file("{}/dicisionTree2.dot".format(DATA_OUT_DIR), encoding='utf8')

# graph.write_png('{}/dicisionTree2.png'.format(DATA_OUT_DIR))

"""## SVC 다중 모델 part 1"""
train_scaled, test_scaled = format_data_to_scaling(train_input, test_input)

svc = SVC(kernel='linear');
svc.fit(train_scaled, train_target);

# print(svc.score(train_scaled, train_target));
# print(svc.score(test_scaled, test_target));

"""## 적절한 degree 값을 찾기 위해 테스트"""
# fit_test_poly(svc, train_input, train_target, test_input, test_target, 1, 4);

"""## SVC 다중 모델 part 2"""
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_scaled)

train_poly = poly.transform(train_scaled)
test_poly = poly.transform(test_scaled)

svc = SVC(kernel='linear');
svc.fit(train_poly, train_target);

# print(svc.score(train_poly, train_target));
# print(svc.score(test_poly, test_target));

"""## 소프트맥스 회귀 모델"""
# softmax= LogisticRegression(multi_class='multinomial',  solver='lbfgs', max_iter=100000, random_state=SEED)

# softmax.fit(train_input,train_target)
# print(softmax.score(train_input, train_target))
# print(softmax.score(test_input, test_target))
