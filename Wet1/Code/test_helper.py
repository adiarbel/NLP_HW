import numpy as np
import train_helper
import re
import time

import calc_one_hist
from itertools import product
from multiprocessing import Pool
from psutil import cpu_count

from itertools import product

'''
def get_all_history(file_path, all_tags):
    num_tags = len(all_tags)
    all_hist = []
    with open(file_path) as f:

        for line in f:
            splited_words = re.split(' |\n', line)
            splited_words[-1] = "*"  # add '*' at the end
            splited_words.insert(0, "*")  # add '*' at the beginning
            splited_words.insert(0, "*")  # add '*' at the beginning

            for word_idx in range(len(splited_words) - 3):
                curr_hist  = np.zeros((num_tags, num_tags, num_tags)) # this will be all the possible tags for pp,p,c
                pword = splited_words[word_idx + 1]
                cword = splited_words[word_idx + 2]
                nword= splited_words[word_idx + 3]
                curr_opt_hists = []
                for pptag in all_tags:
                    for ptag in all_tags:
                        for ctag in all_tags:
                            curr_hist = [cword, pptag, ptag, ctag, nword, pword]
                            curr_opt_hists.append(curr_hist)

                all_hist.append(curr_opt_hists)
    return all_hist
'''
def get_curr_options_q(five_words, all_tags, features_id, w):
    curr_opt_hist = get_curr_features_exps(all_tags, features_id, w, five_words)
    #norm:
    norm_term = np.sum(curr_opt_hist, 2)
    norm_term[norm_term == 0] = 1
    norm_term_3d = np.tile(norm_term, (len(all_tags), 1, 1))
    norm_term_3d = np.transpose(norm_term_3d, (1, 2, 0))
    curr_opt_hist_norm = curr_opt_hist/norm_term_3d
    return curr_opt_hist_norm



def get_curr_features_exps_threads(all_tags, features_id, w, five_words):
    num_tags = len(all_tags)
    curr_opt_hist = np.zeros((num_tags, num_tags, num_tags))  # this will be all the possible tags for pp,p,c
    ppword = five_words[0]
    pword = five_words[1]
    cword = five_words[2]
    nword = five_words[3]
    nnword = five_words[4]

    poolnum = cpu_count(logical=False)
    print(poolnum)
    with Pool(poolnum) as pool:
        if pword == '*':
            pptag = '*'
            ptag = '*'
            for v, ctag in enumerate(all_tags[1:]):
                curr_hist = [cword, pptag, ptag, ctag, nword, pword, nnword, ppword]
                curr_feature = train_helper.represent_input_with_features(curr_hist, features_id)
                sum_one_feature = 0
                for idx in curr_feature:
                    sum_one_feature += w[idx]
                curr_exp = np.exp(sum_one_feature)
                curr_opt_hist[0, 0, v + 1] = curr_exp
        else:
            '''
            t = list(range(num_tags))
            u = list(range(1,num_tags))
            v = list(range(1,num_tags))
            pool.starmap(calc_one_hist.calc_one_hist, product(t, u, v, [all_tags], [five_words], [features_id], [w], [curr_opt_hist]))
            '''
            for t, pptag in enumerate(all_tags):
                for u, ptag in enumerate(all_tags[1:]):
                    for v, ctag in enumerate(all_tags[1:]):
                        curr_hist = [cword, pptag, ptag, ctag, nword, pword, nnword, ppword]
                        curr_feature = train_helper.represent_input_with_features(curr_hist, features_id)
                        sum_one_feature = 0
                        for idx in curr_feature:
                            sum_one_feature += w[idx]
                        curr_exp = np.exp(sum_one_feature)
                        curr_opt_hist[t, u + 1, v + 1] = curr_exp

    return curr_opt_hist
def get_curr_features_exps(all_tags, features_id, w,five_words):
    num_tags = len(all_tags)
    curr_opt_hist = np.zeros((num_tags, num_tags, num_tags))  # this will be all the possible tags for pp,p,c
    ppword = five_words[0]
    pword = five_words[1]
    cword = five_words[2]
    nword = five_words[3]
    nnword = five_words[4]

    if pword == '*':
        pptag = '*'
        ptag = '*'
        for v, ctag in enumerate(all_tags[1:]):
            curr_hist = [cword, pptag, ptag, ctag, nword, pword, nnword, ppword]
            curr_feature = train_helper.represent_input_with_features(curr_hist, features_id)
            sum_one_feature = 0
            for idx in curr_feature:
                sum_one_feature += w[idx]
            curr_exp = np.exp(sum_one_feature)
            curr_opt_hist[0, 0, v+1] = curr_exp
    else:
        for t, pptag in enumerate(all_tags):
            for u, ptag in enumerate(all_tags[1:]):
                for v, ctag in enumerate(all_tags[1:]):
                    curr_hist = [cword, pptag, ptag, ctag, nword, pword, nnword, ppword]
                    curr_feature = train_helper.represent_input_with_features(curr_hist, features_id)
                    sum_one_feature = 0
                    for idx in curr_feature:
                        sum_one_feature += w[idx]
                    curr_exp = np.exp(sum_one_feature)
                    curr_opt_hist[t, u+1, v+1] = curr_exp

    return curr_opt_hist

def memm_viterbi_untagged(all_tags, file_path, weights,feature_ids):
    """
       Write your MEMM Vitebi imlementation below
       You can implement Beam Search to improve runtime
       Implement q efficiently (refer to conditional probability definition in MEMM slides)
       """
    final_tags = []
    prediction_file_path = file_path[:-6]+'_predict.wtag'
    prediction_file = open(prediction_file_path, "w")
    with open(file_path) as f:
        for idx, line in enumerate(f):
            # print(line)
            splited_words = re.split(' |\n', line)
            del splited_words[-1]
            PI_per_line = np.zeros((len(all_tags), len(all_tags), len(splited_words) + 1))
            PIbp_per_line = np.zeros((len(all_tags), len(all_tags), len(splited_words) + 1))

            splited_words.insert(0, "*")
            splited_words.insert(0, "*")
            splited_words.append("*")
            splited_words.append("*")
            PI_per_line[0, 0, 0] = 1  # Starting condition
            PI_per_line, PIbp_per_line = memm_viterbi_iter_untagged(PI_per_line, PIbp_per_line, all_tags, 1, splited_words,
                                                           feature_ids, weights)

            N = np.size(PIbp_per_line, 2)
            line_tags = np.zeros(N - 1, dtype=int)

            line_tags[N - 3], line_tags[N - 2] = np.unravel_index(np.argmax(PI_per_line[:, :, N - 1]),
                                                                  PI_per_line[:, :, N - 1].shape)
            for i in reversed(range(N - 3)):
                line_tags[i] = PIbp_per_line[line_tags[i + 2], line_tags[i + 1], i + 3]
            line_tags_strings = []
            for tag in line_tags:
                # if tag != 0:
                line_tags_strings.append(all_tags[tag])
            underscore_array = ['_'] * len(splited_words)
            line_txt_file = np.core.defchararray.add(np.core.defchararray.add(splited_words, underscore_array), line_tags_strings)
            prediction_file.write(line_txt_file+'\n')
    prediction_file.close()

def memm_viterbi(all_tags, file_path, weights,feature_ids):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    final_tags = []
    test_tags = []
    confusion_matrix = np.zeros((len(all_tags), len(all_tags)))
    num_correct = 0
    num_all = 0

    with open(file_path) as f:
        for idx, line in enumerate(f):
            #print(line)
            line_test_tag = []
            splited_words = re.split(' |\n', line)
            del splited_words[-1]
            PI_per_line = np.zeros((len(all_tags), len(all_tags), len(splited_words)+1))
            PIbp_per_line = np.zeros((len(all_tags), len(all_tags), len(splited_words)+1))

            for word_idx in range(len(splited_words)):
                _, temp_tag = splited_words[word_idx].split('_')
                line_test_tag.append(temp_tag)
            splited_words.insert(0, "*_*")
            splited_words.insert(0, "*_*")
            splited_words.append("*_*")
            splited_words.append("*_*")
            PI_per_line[0, 0, 0] = 1  # Starting condition
            PI_per_line, PIbp_per_line = memm_viterbi_iter(PI_per_line, PIbp_per_line, all_tags, 1, splited_words,feature_ids, weights)

            N = np.size(PIbp_per_line, 2)
            line_tags = np.zeros(N-1, dtype=int)

            line_tags[N-3], line_tags[N-2] = np.unravel_index(np.argmax(PI_per_line[:, :, N-1]), PI_per_line[:, :, N-1].shape)
            for i in reversed(range(N-3)):
                line_tags[i] = PIbp_per_line[line_tags[i+2], line_tags[i+1], i+3]
            line_tags_strings = []
            for tag in line_tags:
                #if tag != 0:
                line_tags_strings.append(all_tags[tag])
            final_tags.append(line_tags)
            test_tags.append(line_test_tag)

            for i, real_tag in enumerate(line_test_tag):
                num_all += 1
                if real_tag == line_tags_strings[i]:
                    num_correct += 1
                confusion_matrix[all_tags.index(real_tag), all_tags.index(line_tags_strings[i])] += 1

            print("Curr accuracy {}".format(num_correct * 100 / num_all))
        print("Final accuracy {}".format(num_correct * 100 / num_all))
        final_acc = num_correct * 100 / num_all
    return final_tags, confusion_matrix , final_acc


def memm_viterbi_iter_untagged(PI, PIbp, all_tags, k, all_words, feature_ids, weights):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    if k == PI.shape[2]:
        return PI, PIbp

    five_words = []
    for i in range(k-1, k+4):
        five_words.append(all_words[i])

    q = get_curr_features_exps_threads(all_tags, feature_ids, weights, five_words)
    PI_km = PI[:, :, k-1]
    piq = np.zeros(q.shape)
    for i in range(np.size(q, 0)):
        piq[:, :, i] = q[:, :, i]*PI_km

    PIbp_k = np.argmax(piq, 0)
    PI_k = piq[PIbp_k]
    PI[:, :, k] = PI_k
    PIbp[:, :, k] = PIbp_k

    PI, PIbp = memm_viterbi_iter(PI, PIbp, all_tags, k+1, all_words, feature_ids, weights)
    return PI, PIbp

def memm_viterbi_iter(PI, PIbp, all_tags, k, all_words, feature_ids, weights):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    t0 = time.time()
    if k == PI.shape[2]:
        return PI, PIbp

    five_words = []
    for i in range(k-1, k+4):
        curr_word, _ = all_words[i].split('_')
        five_words.append(curr_word)


    q = get_curr_options_q(five_words, all_tags, feature_ids, weights)
    t1 = time.time()
    #print("time to calc q:{}".format(t1-t0))
    #q = get_curr_features_exps(all_tags, feature_ids, weights, three_words)
    PI_km = PI[:, :, k-1]
    piq = np.zeros(q.shape)
    for i in range(np.size(q, 0)):
        piq[:, :, i] = q[:, :, i]*PI_km
    t2 = time.time()
    PIbp_k = np.argmax(piq, 0)
    PI_k = np.zeros(PIbp_k.shape)
    for i in range(np.size(PIbp_k,0)):
        for j in range(np.size(PIbp_k,1)):
            PI_k[i,j] = piq[PIbp_k[i,j],i,j]
    t3 = time.time()
    PI[:, :, k] = PI_k
    PIbp[:, :, k] = PIbp_k

    PI, PIbp = memm_viterbi_iter(PI, PIbp, all_tags, k+1, all_words, feature_ids, weights)
    return PI, PIbp


