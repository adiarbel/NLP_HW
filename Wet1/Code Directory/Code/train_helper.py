import numpy as np
import re
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
from feature_statistics_class import feature_statistics_class
from feature2id_class import feature2id_class
import pickle
import time
import test_helper
from sklearn.model_selection import KFold

def get_suf(word):
    suf_slice_size = min(len(word), 4)
    cur_part = word[-suf_slice_size:]
    return cur_part


def get_pre(word):
    pre_slice_size = min(len(word), 4)
    cur_part = word[:pre_slice_size]
    return cur_part


def represent_input_with_features(history, features_id):
    """
        Extract feature vector in per a given history
        :param history: touple{word, pptag, ptag, ctag, nword, pword}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    word = history[0]
    pptag = history[1]
    ptag = history[2]
    ctag = history[3]
    nword = history[4]
    pword = history[5]
    nnword = history[6]
    ppword = history[7]
    features = []

    if (word, ctag) in features_id.words_tags_dict:
        features.append(features_id.words_tags_dict[(word, ctag)])

    suf = get_suf(word)
    for i in reversed(range(1, len(suf) + 1)):
        if (suf[-i:], ctag) in features_id.words_suffix_tags_dict:
            features.append(features_id.words_suffix_tags_dict[(suf[-i:], ctag)])

    pre = get_pre(word)
    for i in reversed(range(1, len(pre) + 1)):
        if (pre[:i], ctag) in features_id.words_prefix_tags_dict:
            features.append(features_id.words_prefix_tags_dict[(pre[:i], ctag)])

    if (pptag, ptag, ctag) in features_id.trigram_tags_dict:
        features.append(features_id.trigram_tags_dict[(pptag, ptag, ctag)])

    if (ptag, ctag) in features_id.bigram_tags_dict:
        features.append(features_id.bigram_tags_dict[(ptag, ctag)])

    if ctag in features_id.unigram_tags_dict:
        features.append(features_id.unigram_tags_dict[ctag])

    flag_all_caps = True
    for ch in word:
        if 'A' > ch or 'Z' < ch:
            flag_all_caps = False
    if ctag in features_id.capitalized_tags_dict and 'A' <= word[0] <= 'Z' and not flag_all_caps:
        features.append(features_id.capitalized_tags_dict[ctag])

    flag_contain = False
    for ch in word:
        if 'A' <= ch <= 'Z':
            flag_contain = True
    if ctag in features_id.contain_capital_tags_dict and not  'A' <= word[0] <= 'Z' and not flag_all_caps and flag_contain:
        features.append(features_id.contain_capital_tags_dict[ctag])

    if (nword, ctag) in features_id.nwords_tags_dict:
        features.append(features_id.nwords_tags_dict[(nword, ctag)])

    if (pword, ctag) in features_id.pwords_tags_dict:
        features.append(features_id.pwords_tags_dict[(pword, ctag)])

    if (ppword, ctag) in features_id.ppwords_tags_dict:
        features.append(features_id.ppwords_tags_dict[(ppword, ctag)])

    if (nnword, ctag) in features_id.nnwords_tags_dict:
        features.append(features_id.nnwords_tags_dict[(nnword, ctag)])

    if ctag in features_id.Allcapitalized_tags_dict and flag_all_caps:
        features.append(features_id.Allcapitalized_tags_dict[ctag])

    if ctag in features_id.dot_tags_dict and '.' in word:
        features.append(features_id.dot_tags_dict[ctag])

    if ctag in features_id.apos_tags_dict and '\'' in word:
        features.append(features_id.apos_tags_dict[ctag])

    if ctag in features_id.hyphen_tags_dict and '-' in word:
        features.append(features_id.hyphen_tags_dict[ctag])


    flag_num = False
    for ch in word:
        if '0' <= ch <= '9':
            flag_num = True
    if ctag in features_id.contain_number_tags_dict and flag_num:
        features.append(features_id.contain_number_tags_dict[ctag])

    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #

    return features


def get_all_features(file_path, all_tags, features_id):
    all_hist, real_hist = get_all_history(file_path,all_tags)

    sum_of_reals = np.zeros(features_id.n_total_features)
    for history in real_hist:
        curr_feature = represent_input_with_features(history,features_id)
        for idx in curr_feature:
            sum_of_reals[idx] += 1

    all_features = []
    for curr_opt_histories in all_hist:
        curr_opt_features = []
        for history in curr_opt_histories:
            curr_feature = represent_input_with_features(history, features_id)
            curr_opt_features.append(curr_feature)
        all_features.append(curr_opt_features)

    return [sum_of_reals, all_features]


def create_kfold_files(file_path,k):
    kf = KFold(n_splits=k, shuffle=True)
    lines = np.array([])
    with open(file_path) as f:
        for line in f:
            lines = np.append(lines, line)
    i = 0
    for train_index, test_index in kf.split(lines):
        lines_train = lines[train_index]
        lines_test = lines[test_index]
        train_path = '../Data/train_'+str(i)+'fold.wtag'
        test_path = '../Data/test_'+str(i)+'fold.wtag'
        with open(train_path,'w') as train_f:
            for line in lines_train:
                #print(type((str(line) + '\n')))
                train_f.write(line)
            train_f.close()
        with open(test_path,'w') as test_f:
            for line in lines_test:
                #print(type((str(line) + '\n')))
                test_f.write(line)
            test_f.close()
        i += 1

def train2_kfold(threshold,num_iterations,milestone):
    file_path = '../Data/train2.wtag'
    k = 10

    create_kfold_files(file_path, k)
    for i in range(k):
        train_path = '../Data/train_'+str(i)+'fold.wtag'

        weights_path_load = "train2_weights_extra_features_kfold/kfold{}_trained_weights_th{}_ep{}.pkl".format(i,threshold, num_iterations)
        train(train_path, threshold, num_iterations, weights_path_load, 'times/kfold{}_train2'.format(i))
    accs = []
    for i in range(2,k):
        train_path = '../Data/train_' + str(i) + 'fold.wtag'
        test_path = '../Data/test_' + str(i) + 'fold.wtag'
        feature_statistics = feature_statistics_class()
        feature_ids = feature2id_class(feature_statistics, threshold)

        feature_statistics.get_all_counts(train_path)
        all_tags = list(feature_statistics.unigram_tags_count_dict.keys())
        all_tags.insert(0, '*')
        feature_ids.get_all_ids(train_path)

        weights_path_load = "train2_weights_extra_features_kfold/kfold{}_trained_weights_th{}_ep{}_milestone{}.pkl".format(i, threshold, num_iterations, milestone)
        with open(weights_path_load, 'rb') as f:  #
            optimal_params = pickle.load(f)
            pre_trained_weights = optimal_params
            _, _, final_acc = test_helper.memm_viterbi(all_tags, test_path, pre_trained_weights,
                                                       feature_ids)
            accs.append(final_acc)

    return accs


def get_all_history(file_path,all_tags):
    all_hist = []
    real_hist = []
    with open(file_path) as f:
        for line in f:
            splited_words = re.split(' |\n', line)
            splited_words[-1] = "*_*"  # add '*' at the end
            splited_words.append("*_*")  # add '*' at the beginning
            splited_words.insert(0, "*_*")  # add '*' at the beginning
            splited_words.insert(0, "*_*")  # add '*' at the beginning

            for word_idx in range(len(splited_words) - 4):
                ppword, pptag = splited_words[word_idx].split('_')
                pword, ptag = splited_words[word_idx + 1].split('_')
                word, ctag = splited_words[word_idx + 2].split('_')
                nword, _ = splited_words[word_idx + 3].split('_')
                nnword, _ = splited_words[word_idx + 4].split('_')

                curr_base_hist = [word, pptag, ptag, ctag, nword, pword,nnword,ppword]
                ctag_idx = 3
                real_hist.append(tuple(curr_base_hist))
                curr_opt_hist = []
                for tag in all_tags:
                    opt_hist = curr_base_hist
                    opt_hist[ctag_idx] = tag
                    curr_opt_hist.append(tuple(opt_hist))
                all_hist.append(curr_opt_hist)
    return [all_hist, real_hist]

def calc_objective_per_iter(w_i, lam, all_features, sum_of_real_features, all_tags, ids, iter, weights_path_write,t0,time_path):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """
    #print("###################started iteration!###################")
    ## Calculate the terms required for the likelihood and gradient calculations
    ## Try implementing it as efficient as possible, as this is repeated for each iteration of optimization.
    exp_terms = []
    linear_term = sum_of_real_features.dot(w_i)
    normalization_term = 0
    for curr_opt_features in all_features:
        sum_curr_opt_features = 0
        exp_terms_Xi = []
        for feature in curr_opt_features:
            sum_one_feature = 0
            for idx in feature:
                sum_one_feature += w_i[idx]
            sum_curr_opt_features += np.exp(sum_one_feature)
            exp_terms_Xi.append([feature, np.exp(sum_one_feature)])
        normalization_term += np.log(sum_curr_opt_features)
        exp_terms.append(exp_terms_Xi)
    regularization = 0.5*lam*w_i.T.dot(w_i)
    #print("###################calc likelihood !###################")
    likelihood = linear_term - normalization_term - regularization

    regularization_grad = lam*w_i
    empirical_counts = sum_of_real_features
    expected_counts = np.zeros(ids.n_total_features)

    for exp_term_Xi in exp_terms:
        exp_term_Xi_np = np.array(exp_term_Xi)
        sum_curr_opt_features = np.sum(exp_term_Xi_np[:, 1])
        for feature, exp_term in exp_term_Xi:
            temp_feature = np.zeros(ids.n_total_features)
            for idx in feature:
                expected_counts[idx] += exp_term/sum_curr_opt_features

    grad = empirical_counts - expected_counts - regularization_grad
    #print("###################Finished iter!###################")
    # weights_path_write = weights_path_write[:-4]+'_milestone{}.pkl'.format(iter[0])
    # if iter[0] % 1 == 0:
    #     with open(weights_path_write, 'wb') as f:
    #         pickle.dump(tuple(w_i), f)
    iter[0] += 1
    t_till_now = time.time() - t0
    # curr_time_path = time_path+'_milestone{}.pkl'.format(iter[0])
    # with open(curr_time_path, 'wb') as f:
    #     pickle.dump(t_till_now, f)

    return (-1) * likelihood, (-1) * grad


def train(path,threshold, num_iterations, weights_path_write, time_path, weights_path_pretrain=""):
    # Statistics
    statistics = feature_statistics_class()
    statistics.get_all_counts(path)

    # feature2id
    feature2id = feature2id_class(statistics, threshold)
    feature2id.get_all_ids(path)

    # define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter'
    ##lam = 0.001
    lam = 0.1
    #EPSILON = 10 ** (-5)
    #m = 10
    all_tags = statistics.unigram_tags_count_dict.keys()
    sum_of_real_features, all_features = get_all_features(path, all_tags, feature2id)
    iter = [0]
    t0 = time.time()
    args = (lam, all_features, sum_of_real_features, all_tags, feature2id, iter, weights_path_write,t0,time_path)

    if weights_path_pretrain != "":
        with open(weights_path_pretrain, 'rb') as f:
            optimal_params = pickle.load(f)
            pre_trained_weights = optimal_params
            w_0 = pre_trained_weights
    else:
        w_0 = np.random.random(feature2id.n_total_features)
        #w_0 = np.zeros(feature2id.n_total_features, dtype=np.float32)



    #optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=num_iterations, iprint=1)
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=num_iterations, iprint=1)
    weights = optimal_params[0]
    with open(weights_path_write, 'wb') as f:
        pickle.dump(tuple(weights), f)




    #### In order to load pre-trained weights, just use the next code: ####
    #                                                                     #
    # with open(weights_path, 'rb') as f:                                 #
    #   optimal_params = pickle.load(f)                                   #
    # pre_trained_weights = optimal_params[0]                             #
    #                                                                     #
    #######################################################################
