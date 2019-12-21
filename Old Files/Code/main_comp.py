
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
    #test_path = r"..\Data\test1.wtag"
    #test_path = r"..\Data\miniTest.wtag"
    test_path = r"..\Data\comp1.words"

    threshold = 1
    feature_statistics = feature_statistics_class.feature_statistics_class()
    feature_ids = feature2id_class.feature2id_class(feature_statistics, threshold)

    feature_statistics.get_all_counts(path)
    all_tags = list(feature_statistics.unigram_tags_count_dict.keys())
    all_tags.insert(0, '*')
    feature_ids.get_all_ids(path)



    parser.add_argument("-m", "--milestone", help="milestone",default=85)

    args = parser.parse_args()
    milestone = args.milestone

    num_iterations = 100
    weights_path_load = "train1_weights_extra_features/newlam_trained_weights_th{}_ep{}_milestone{}.pkl".format(threshold, num_iterations, milestone)

    #train_helper.train(path, threshold, num_iterations, weights_path_load,weights_path_pretrain)
    with open(weights_path_load, 'rb') as f:  #
        optimal_params = pickle.load(f)
        pre_trained_weights = optimal_params

    t_start_test = time.time()
    test_helper.memm_viterbi_untagged(all_tags, test_path, pre_trained_weights, feature_ids,1)
    t_elapsed_test = time.time() - t_start_test

    save_con_path = "time_comp1_th{}_ep{}_mile{}.pkl".format(threshold, num_iterations, milestone)
    with open(save_con_path, 'wb') as f:
        pickle.dump(t_elapsed_test, f)
