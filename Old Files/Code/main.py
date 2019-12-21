
import train_helper
import pickle
import test_helper
import feature_statistics_class
import feature2id_class
import numpy as np
import multiprocessing
import argparse
from itertools import product
import pandas as pd
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    path = r"..\Data\train1.wtag"
    test_path = r"..\Data\test1.wtag"
    #test_path = r"..\Data\miniTest.wtag"
    #test_path = r"..\Data\compTest.words"

    threshold = 1
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
    #20 is best ->92.2% -> because we started at 14 the best is 34

    parser.add_argument("-m", "--milestone", help="milestone",default=85)

    args = parser.parse_args()
    milestone = args.milestone
    print(all_tags)
    #for milestone in milestones:
    print("######### at milestone: {} ##########".format(milestone))
    num_iterations = 100
    weights_path_load = "train1_weights_extra_features/newlam_trained_weights_th{}_ep{}_milestone{}.pkl".format(threshold, num_iterations, milestone)

    #train_helper.train(path, threshold, num_iterations, weights_path_load,weights_path_pretrain)
    with open(weights_path_load, 'rb') as f:  #
        optimal_params = pickle.load(f)
        pre_trained_weights = optimal_params

    t_start_test = time.time()
    final_tags, confusion_matrix, final_acc = test_helper.memm_viterbi(all_tags, test_path, pre_trained_weights, feature_ids)
    t_elapsed_test = time.time() - t_start_test

    con_mat_offdiag = confusion_matrix - np.diag(np.diag(confusion_matrix))
    con_mat_offdiag_sum = np.sum(con_mat_offdiag, axis=1)
    max10inds = list(np.argsort(con_mat_offdiag_sum))
    max10inds.reverse()
    max10inds = np.array(max10inds[0:10])
    con_mat_10 = np.zeros((len(max10inds),len(max10inds)))
    for i,j in product(max10inds, max10inds):
        con_mat_10[np.where(max10inds==i),np.where(max10inds==j)] = con_mat_offdiag[i, j]
    columns = []
    for ind in max10inds:
        columns.append(all_tags[ind])
    final_con_mat = pd.DataFrame(con_mat_10, columns=columns)
    final_con_mat.index = columns

    con_mat = (confusion_matrix, final_con_mat, final_tags, final_acc, t_elapsed_test)

    save_con_path = "con_mat_test1_th{}_ep{}_mile{}.pkl".format(threshold, num_iterations, milestone)
    with open(save_con_path, 'wb') as f:
        pickle.dump(con_mat, f)

    # test_helper.memm_viterbi_untagged(all_tags, test_path, pre_trained_weights, feature_ids)

