import pickle
import pandas as pd
import numpy as np
import math
import re
from collections import Counter
import time


# class Score:
#     def __init__(self, cos, pos_neg, review_string):
#         self.cos = cos
#         self.pos_neg = pos_neg
#         self.review_string = review_string


#   https://stackoverflow.com/a/15174569
def cos_sim(review_train, review_test):
    vec1 = review_train
    vec2 = review_test
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


start_time = time.time()

path = 'C:/Users/bandi/Downloads/CS584/HW1/'

WORD = re.compile(r"\w+")

data_train = pd.read_pickle(path + "train_pickle")
data_test = pd.read_pickle(path + "test_pickle")

data_train_list = data_train.values.tolist()

data_test_list = []
for index, row in data_test.iterrows():
    data_test_list.append(row[0])

i_counter = 0
final_frame_list = []
for i in data_test_list:
    # defining Data Structure for K
    frame = [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]]

    vec_test = Counter(WORD.findall(i))

    for j in (data_train_list):

        score = cos_sim(Counter(WORD.findall(j[1])), vec_test)

        # print(score)
        score_val = j[0]
        # score_string = train_row[1]

        for x in frame:
            if (x[0] < score):
                x[0] = score
                x[1] = score_val
                break

    final_frame_list.append(frame)

    print(str(i_counter) + "  " + str(frame))

with open(path + 'vector1', 'wb') as f:
    pickle.dump(final_frame_list, f)
print("--- %s seconds ---" % (time.time() - start_time))
