import pandas as pd
import numpy as np
import pydot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

DATA_DIR = './classification';
DATA_IN_DIR = './classification/data/in';
DATA_OUT_DIR = './classification/data/out';
FILE_PATH = 'fish/Fish.csv';

data_frame = pd.read_csv('{}/{}'.format(DATA_IN_DIR, FILE_PATH), header=0, sep=',')

pike_weight = data_frame[data_frame['Species']=='Pike'][['Weight']].values
pike_length = data_frame[data_frame['Species']=='Pike'][['Length2']].values
pike_size = len(pike_weight)

smelt_weight = data_frame[data_frame['Species']=='Smelt'][['Weight']].values
smelt_length = data_frame[data_frame['Species']=='Smelt'][['Length2']].values
smelt_size = len(smelt_weight)

fish_data = np.column_stack((np.concatenate((pike_length, smelt_length))
                           , np.concatenate((pike_weight, smelt_weight))))
fish_target = np.concatenate((np.ones(pike_size, dtype=np.int64), np.zeros(smelt_size, dtype=np.int64)))

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target,stratify=fish_target, random_state=123)

classifier = DecisionTreeClassifier()
classifier.fit(train_input, train_target)

print("Accuracy: {}%".format(classifier.score(test_input, test_target)*100))
export_graphviz(classifier, out_file="{}/dicisionTree.dot".format(DATA_OUT_DIR), class_names=["Smelt", "Pike"],
                feature_names=["length", "weight"], filled=True)

(graph,) = pydot.graph_from_dot_file("{}/dicisionTree.dot".format(DATA_OUT_DIR), encoding='utf8')

graph.write_png('{}/dicisionTree.png'.format(DATA_OUT_DIR))