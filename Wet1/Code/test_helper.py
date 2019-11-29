import numpy as np


def get_all_history(file_path, all_tags):
    all_hist = []
    with open(file_path) as f:
        all_hists = []
        for line in f:
            splited_words = re.split(' |\n ', line)
            splited_words[-1] = "*"  # add '*' at the end
            splited_words.insert(0, "*")  # add '*' at the beginning
            splited_words.insert(0, "*")  # add '*' at the beginning

            for word_idx in range(len(splited_words) - 3):
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


def memm_viterbi(all_tags, file_path, weights):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    all_histories = get_all_history(file_path, all_tags)

    features  =
    exps_terms =
    
    num_words = 0
    with open(file_path) as f:
        for line in f:
            splited_words = re.split(' |\n ', line)
            del splited_words[-1]
            num_words += len(splited_words)

    all_tags = all_tags.append("*")
    PI = np.zeros((len(all_tags), len(all_tags), num_words))
    PI[0, 0, 0] = 1  # Starting condition
    PI = memm_viterbi_iter(PI, all_tags, 0)
    return PI


def memm_viterbi_iter(PI, all_tags, k):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    if k == PI.shape[2]:
        return PI
    PI = memm_viterbi_iter(PI, all_tags, k+1)
    return PI
