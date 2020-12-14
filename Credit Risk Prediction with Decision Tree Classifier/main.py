from graphviz import Source
import os
import numpy as np
import pandas as pd
import pydotplus

from sklearn import tree
from IPython.display import Image

os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# path = 'C:/Users/bandi/Downloads/CS584/HW2/src_code/'

# train_df = pd.DataFrame()  # init null dataframe
# with open('train_data.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row[0])

train_df = pd.read_csv("train_data.csv")
# print(train_df.head())

# one hot encoding
print("printing one hot encoding\n")
encode_train = pd.get_dummies(train_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11']])
print(encode_train)

#classifier Initialization
var_classifier = tree.DecisionTreeClassifier(max_depth= 12, min_samples_split = 50, class_weight = {0:0.24720 , 1:0.7841})
train_classifier = var_classifier.fit(encode_train, train_df['credit'])

print(tree.export_graphviz(train_classifier, None))


#dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(encode.columns.values), class_names=['Bad Credit', 'Good Credit'], rounded=True, filled=True)
# print("Calculating graph from dot, takes time")
# graph = pydotplus.graph_from_dot_data(dot_data)

#converting to Graph
# Image(graph.create_png())
# graph.write_png("visualize.png")
# print("After Visual Graph Fisnihed")

# predicting code
# prediction = clf_train.predict([[0,0,1,0,1,0,0,1,1,0]])


#testing code
test_df = pd.read_csv("test_data.csv")
encode_test = pd.get_dummies(test_df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11']])
# print(encode1.head())

var_prediction = train_classifier.predict(encode_test)
var_prediction = var_prediction.reshape(var_prediction.size, 1)
print(var_prediction)



out_file_ptr = open("format.txt", "w")
for row in var_prediction:
    out_file_ptr.write(str(row[0]) + "\n")
out_file_ptr.close()

# Test Set Trail
# 7	40	3	7	0	0	4	4	1	 Black	 Male