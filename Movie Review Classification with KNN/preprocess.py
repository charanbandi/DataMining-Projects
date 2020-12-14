import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

porter_stemmer  = PorterStemmer()
lemmatizer = WordNetLemmatizer()

path = 'C:/Users/bandi/Downloads/CS584/HW1/'

def clean(list_strings):
    list_strings_clean = []

    # cleaning strings
    for i in list_strings:

        i = i.replace('<br /><br />',' ')# remove all break tags, and add a space
        i = i.lower()  # lowercase
        i = re.sub('\[.*?\]', '', i)  # square brackets
        i = re.sub('[%s]' % re.escape(string.punctuation), '', i)  # remove punctuation
        i = re.sub('\w*\d\w*', '', i)  # remove words containing numbers
        i = re.sub('[‘’“”…]', '', i)  # remove more punctuation
        i = re.sub('\n', '', i)  # remove Non-sensical text
        list_strings_clean.append(i)


    # removing stop words
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    for i in list_strings_clean:
        word_tokens = word_tokenize(i)
        sentence = []
        #   Removing stop words with tokenize and also achieveing extra spaces removes as a bonus
        for w in word_tokens:
            if w not in stop_words:
                # choose and check between stemming or lematization, leaving a code snippet for both anyways
                temp =(porter_stemmer.stem(w)) # stemming each word to its base
                sentence.append(temp)
                # sentence.append(lemmatizer.lemmatize(w))    # lemmatize each word to its lemma
        filtered_sentence.append(' '.join(sentence))

    # for i in range(len(list_strings)):
    #     print("Before Clean: " + list_strings[i])
    #     print("After Clean: " + list_strings_clean[i])
    #     print("After Stop & Lemma: "+ filtered_sentence[i])
    return filtered_sentence

def pickle_file(file_name,file_data):
    file_data.to_pickle(file_name)




train_data_unclean = pd.read_csv(path + "train_data.txt", sep="\t", header=None)
# print(len(train_data))
# print(train_data.head())
# print(type(train_data))

temp_list_train = []
temp_list_train_vals = []
for index, row in train_data_unclean.iterrows():
    temp_list_train.append(row[1])
    temp_list_train_vals.append(row[0])
temp_list_train = clean(temp_list_train)

dict_train = {'Value':temp_list_train_vals,'Review':temp_list_train}
train_data = pd.DataFrame(dict_train) # for pickle
print(train_data.head(),train_data.shape)


temp_list = []
count = 0
with open(path + "test_data.txt", encoding='utf-8') as fp:
    for line in fp:
        count += 1
        temp_list.append(line.strip())

test_data = pd.DataFrame(clean(temp_list)) # for pickle
print(test_data.head(),test_data.shape)

file_train = "train_pickle"
file_test = "test_pickle"

pickle_file(file_train,train_data)
pickle_file(file_test,test_data)


