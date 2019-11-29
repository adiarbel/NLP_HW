import numpy as np
import scipy
from feature_statistics_class import feature_statistics_class
from feature2id_class import feature2id_class
import train_helper
import operator
import pickle

path = r"..\Data\train2.wtag"
threshold = 30

'''
find_all_features = True
if find_all_features:
    sum_of_reals, all_features = train_helper.get_all_features(path, statistics.unigram_tags_count_dict.keys(), ids)
    features_path = './all_features.pkl'
    sum_of_reals_path = './sum_of_reals.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(all_features, f)
    with open(sum_of_reals_path, 'wb') as f:
        pickle.dump(sum_of_reals, f)
else:
    with open('./all_features.pkl', 'rb') as f:
        all_features = pickle.load(f)
    with open('./sum_of_reals.pkl', 'rb') as f:
        sum_of_reals = pickle.load(f)
'''

train_helper.train(path, threshold)
#
# print(train_helper.represent_input_with_features(("dog", "NN", "DT", "NN", "The", "eats"), ids))

