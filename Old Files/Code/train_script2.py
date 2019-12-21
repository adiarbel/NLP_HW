
import train_helper
import pickle
import test_helper
import feature_statistics_class
import feature2id_class
import time
import argparse
import scipy



path = r"../Data/train2.wtag"
#test_path = r"../Data/test1.wtag"
test_path = r"../Data/test2.wtag"

#thresholds = [5, 10, 25, 50, 70, 100, 200]
#thresholds = [10, 25, 50, 70]

thresholds = [1]
num_iterations = 100
for threshold in thresholds:

    t0_statistics = time.time()
    feature_statistics = feature_statistics_class.feature_statistics_class()
    feature_ids = feature2id_class.feature2id_class(feature_statistics, threshold)

    feature_statistics.get_all_counts(path)
    all_tags = list(feature_statistics.unigram_tags_count_dict.keys())
    all_tags.insert(0, '*')
    feature_ids.get_all_ids(path)

    elapsed_statistics = time.time() - t0_statistics

    #t0_train = time.time()
    #weights_path_load = "train1_weights_extra_features/trained_weights_th{}_ep{}_milestone{}.pkl".format(threshold, num_iterations, milestone)
    weights_path_load = "train2_weights_extra_features/newlam101_trained_weights_th{}_ep{}.pkl".format(threshold, num_iterations)
    #weights_path_pretrain = "train1_weights_extra_features/trained_weights_th1_ep26_milestone14.pkl"

    #weights_path_load = r"C:\Users\user\Documents\NLP_HW\Wet1\Code\train1_weights_extra_features\trained_weights_th3_ep150_milestone30.pkl"
    #eights_path_load = r"C:\Users\user\Technion\Adi Arbel - 097215\Repository\train1_weights_extra_features\trained_weights_th10_ep22.pkl"


    #weights_path_load = r"C:\Users\user\Documents\NLP_HW\Wet1\Code\train1_weights_extra_features\trained_weights_th3_ep150.pkl"

    train_helper.train(path, threshold, num_iterations, weights_path_load, "times/train2101_newlam1")
    '''
    #elapsed_train = time.time() - t0_train

    with open(weights_path_load, 'rb') as f:  #
        optimal_params = pickle.load(f)
        pre_trained_weights = optimal_params

    #t0_predict = time.time()
    final_tags, confusion_matrix,final_acc = test_helper.memm_viterbi(all_tags, test_path, pre_trained_weights, feature_ids)
    #elapsed_predict = time.time() - t0_predict

    save_con_path = "con_mat_miniTest_th{}_ep{}_mile{}.pkl".format(threshold, num_iterations, milestone)
    con_mat = (confusion_matrix, final_tags , final_acc)
    time_path = "time_miniTest_th{}_ep{}_mile{}.pkl".format(threshold, num_iterations, milestone)
    with open(save_con_path, 'wb') as f:
        pickle.dump(con_mat, f)
    with open(time_path, 'wb') as f:
        pickle.dump((elapsed_statistics, elapsed_train, elapsed_predict), f)
    '''
