import numpy as np
import train_helper
import math

def calc_one_hist(t, u, v,all_tags,five_words,features_id,w,curr_opt_hist):
    num_tags = len(all_tags)
    ppword = five_words[0]
    pword = five_words[1]
    cword = five_words[2]
    nword = five_words[3]
    nnword = five_words[4]

    pptag = all_tags[t]
    ptag = all_tags[u]
    ctag = all_tags[v]

    curr_hist = [cword, pptag, ptag, ctag, nword, pword, nnword, ppword]
    curr_feature = train_helper.represent_input_with_features(curr_hist, features_id)
    sum_one_feature = 0
    for idx in curr_feature:
        sum_one_feature += w[idx]
    curr_exp = math.exp(sum_one_feature)
    #curr_exp = np.exp(sum_one_feature)
    curr_opt_hist[t, u, v] = curr_exp
