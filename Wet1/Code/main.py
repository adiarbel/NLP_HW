
import train_helper
import pickle
import test_helper
import feature_statistics_class
import feature2id_class
import numpy as np

path = r"..\Data\train1.wtag"
#test_path = r"..\Data\train1.wtag"
test_path = r"..\Data\miniTest.wtag"
threshold = 70

feature_statistics = feature_statistics_class.feature_statistics_class()
feature_ids = feature2id_class.feature2id_class(feature_statistics, threshold)

feature_statistics.get_all_counts(path)
all_tags = list(feature_statistics.unigram_tags_count_dict.keys())
all_tags.insert(0, '*')
feature_ids.get_all_ids(path)

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
weights_path_load = "train1_weights_extra_features/trained_weights_th10_ep22.pkl"
weights_path_pretrain = "train1_weights_extra_features/trained_weights_th10_ep12.pkl"
num_iterations = 10
#train_helper.train(path, threshold, num_iterations, weights_path_load,weights_path_pretrain)
with open(weights_path_load, 'rb') as f:  #
    optimal_params = pickle.load(f)
    pre_trained_weights = optimal_params[0]
print(all_tags)
final_tags, confusion_matrix = test_helper.memm_viterbi(all_tags, test_path, pre_trained_weights, feature_ids)
con_mat = confusion_matrix
con_mat_offdiag = con_mat - np.diag(np.diag(con_mat))
print(np.unravel_index(np.argmax(con_mat_offdiag),con_mat.shape))

save_con_path = "con_mat_22.pkl"
with open(save_con_path, 'wb') as f:
    pickle.dump(confusion_matrix, f)