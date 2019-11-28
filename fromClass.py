from collections import OrderedDict


class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        # ---Add more count dictionaries here---

    def get_word_tag_pair_count(self, file_path):
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if (cur_word, cur_tag) not in self.words_tags_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #


class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = collections.OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #

########################################################################################################################


class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = collections.OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = split(splited_words[word_idx], '_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #

########################################################################################################################

def represent_input_with_features(history, word_tags_dict):
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
    features = []

    if (word, ctag) in word_tags_dict:
        features.append(word_tags_dict[(word, ctag)])

    # --- CHECK APEARANCE OF MORE FEATURES BELOW --- #

    return features


########################################################################################################################


def calc_objective_per_iter(w_i, arg_1, arg_2, ...):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """

    ## Calculate the terms required for the likelihood and gradient calculations
    ## Try implementing it as efficient as possible, as this is repeated for each iteration of optimization.

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad



########################################################################################################################


from scipy.optimize import fmin_l_bfgs_b

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pairs(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)
feature2id.get_word_tag_pairs(train_path)

# define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter'
args = (arg_1, arg_2, ...)
w_0 = np.zeros(n_total_features, dtype=np.float32)
optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]

# Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl' # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

#### In order to load pre-trained weights, just use the next code: ####
#                                                                     #
# with open(weights_path, 'rb') as f:                                 #
#   optimal_params = pickle.load(f)                                   #
# pre_trained_weights = optimal_params[0]                             #
#                                                                     #
#######################################################################



########################################################################################################################


def memm_viterbi():
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    return tags_infer

########################################################################################################################

from IPython.display import HTML
from base64 import b64encode
! git clone https://github.com/eyalbd2/097215_Natural-Language-Processing_Workshop-Notebooks.git


########################################################################################################################

mp4 = open('/content/097215_Natural-Language-Processing_Workshop-Notebooks/ViterbiSimulation.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


########################################################################################################################


HTML("""
<video width=1000 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


########################################################################################################################


