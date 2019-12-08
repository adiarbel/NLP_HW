from collections import OrderedDict
import re

class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        self.n_tag_pairs = 0
        self.n_tag_pairs_p = 0
        self.n_tag_pairs_n = 0
        self.n_suffix_tags = 0
        self.n_prefix_tags = 0
        self.n_trigram_tags = 0
        self.n_bigram_tags = 0
        self.n_unigram_tags = 0
        self.n_capitalized_tags = 0
        self.n_Allcapitalized_tags = 0
        self.n_hyphen_tags = 0
        self.n_contains_number_tags = 0

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.pwords_tags_dict = OrderedDict()
        self.nwords_tags_dict = OrderedDict()
        self.words_suffix_tags_dict = OrderedDict()
        self.words_prefix_tags_dict = OrderedDict()
        self.trigram_tags_dict = OrderedDict()
        self.bigram_tags_dict = OrderedDict()
        self.unigram_tags_dict = OrderedDict()
        self.capitalized_tags_dict = OrderedDict()

        self.Allcapitalized_tags_dict = OrderedDict()

        self.hyphen_tags_dict = OrderedDict()

        self.contain_number_tags_dict = OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")
                splited_words.append("*_*")

                for word_idx in range(1, len(splited_words)-1):

                    pword, _ = splited_words[word_idx - 1].split('_')
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    nword, _ = splited_words[word_idx + 1].split('_')

                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.statistics.words_tags_count_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_total_features
                        self.n_total_features += 1
                        self.n_tag_pairs += 1

                    if ((pword, cur_tag) not in self.pwords_tags_dict) \
                            and (self.statistics.pwords_tags_count_dict[(pword, cur_tag)] >= self.threshold):
                        self.pwords_tags_dict[(pword, cur_tag)] = self.n_total_features
                        self.n_total_features += 1
                        self.n_tag_pairs += 1

                    if ((nword, cur_tag) not in self.nwords_tags_dict) \
                            and (self.statistics.nwords_tags_count_dict[(nword, cur_tag)] >= self.threshold):
                        self.nwords_tags_dict[(nword, cur_tag)] = self.n_total_features
                        self.n_total_features += 1
                        self.n_tag_pairs += 1

    def get_pre_suf_tag_pairs(self, file_path, pre_suf_flag):
        """
            Extract out of text all pre-suf/tag pairs
            :param file_path: full path of the file to read
            :param pre_suf_flag: flag to choose between suffix (True) and prefix (False)
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                split_words = re.split(' | \n', line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    cur_part = ""
                    cur_word_len = len(cur_word)
                    pre_suf_slice_size = min(cur_word_len, 4)
                    if pre_suf_flag:
                        cur_part = cur_word[-pre_suf_slice_size:]
                    else:
                        cur_part = cur_word[:pre_suf_slice_size]
                    for i in reversed(range(1,pre_suf_slice_size+1)):
                        if pre_suf_flag:
                            if (cur_part[-i:], cur_tag) not in self.words_suffix_tags_dict  \
                                    and (self.statistics.words_suffix_tags_count_dict[(cur_part[-i:], cur_tag)] >= self.threshold):
                                self.words_suffix_tags_dict[(cur_part[-i:], cur_tag)] = self.n_total_features
                                self.n_total_features += 1
                                self.n_suffix_tags += 1
                        else:
                            if (cur_part[:i], cur_tag) not in self.words_prefix_tags_dict \
                                    and (self.statistics.words_prefix_tags_count_dict[
                                             (cur_part[:i], cur_tag)] >= self.threshold):
                                self.words_prefix_tags_dict[(cur_part[:i], cur_tag)] = self.n_total_features
                                self.n_total_features += 1
                                self.n_prefix_tags += 1

    def get_trigram_tags(self, file_path):
        """
            Extract out of text all trigram counts
            We padded with one '*' at the beginning and end of sentence
            This is because the information captured by "* * tag" is also captured by the bigram
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                for word_idx in range(len(splited_words) - 2):
                    _, prev_prev_tag = splited_words[word_idx].split('_')
                    _, prev_tag = splited_words[word_idx + 1].split('_')
                    _, cur_tag = splited_words[word_idx + 2].split('_')
                    curr_key = (prev_prev_tag, prev_tag, cur_tag)
                    if curr_key not in self.trigram_tags_dict \
                            and (self.statistics.trigram_tags_count_dict[curr_key] >= self.threshold):
                        self.trigram_tags_dict[curr_key] = self.n_total_features
                        self.n_total_features += 1
                        self.n_trigram_tags += 1

    def get_bigram_tags(self, file_path):
        """
            Extract out of text all bigram counts
            We padded with one '*' at the beginning and end of sentence
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                for word_idx in range(len(splited_words) - 1):
                    _, prev_tag = splited_words[word_idx].split('_')
                    _, cur_tag = splited_words[word_idx + 1].split('_')
                    curr_key = (prev_tag, cur_tag)
                    if curr_key not in self.bigram_tags_dict \
                            and (self.statistics.bigram_tags_count_dict[curr_key] >= self.threshold):
                        self.bigram_tags_dict[curr_key] = self.n_total_features
                        self.n_total_features += 1
                        self.n_bigram_tags += 1

    def get_unigram_tags(self, file_path):
        """
            Extract out of text all uniram counts
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    _, cur_tag = splited_words[word_idx].split('_')
                    curr_key = cur_tag
                    if curr_key not in self.unigram_tags_dict \
                            and (self.statistics.unigram_tags_count_dict[curr_key] >= self.threshold):
                        self.unigram_tags_dict[curr_key] = self.n_total_features
                        self.n_total_features += 1
                        self.n_unigram_tags += 1

    def get_capitalized_tags(self, file_path):
        """
            Extract out of text all the counts for capitalized tags
            :param file_path: full path of the file to read
                return all capitalized tags counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    flag = True
                    for ch in cur_word:
                        if 'A' > ch or 'Z' < ch:
                            flag = False
                    if 'A' <= cur_word[0] <= 'Z' and not flag:
                        curr_key = cur_tag
                        if curr_key not in self.capitalized_tags_dict \
                                and (self.statistics.capitalized_tags_count_dict[curr_key] >= self.threshold):
                            self.capitalized_tags_dict[curr_key] = self.n_total_features
                            self.n_total_features += 1
                            self.n_capitalized_tags += 1

    def get_Allcapitalized_tags(self, file_path):
        """
            Extract out of text all the counts for capitalized tags
            :param file_path: full path of the file to read
                return all capitalized tags counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    flag = True
                    for ch in cur_word:
                        if 'A' > ch or 'Z' < ch:
                            flag = False
                    if flag:
                        curr_key = cur_tag
                        if curr_key not in self.Allcapitalized_tags_dict \
                                and (self.statistics.Allcapitalized_tags_count_dict[curr_key] >= self.threshold):
                            self.Allcapitalized_tags_dict[curr_key] = self.n_total_features
                            self.n_total_features += 1
                            self.n_capitalized_tags += 1

    def get_hyphen_tags(self, file_path):
        """
            Extract out of text all the counts for capitalized tags
            :param file_path: full path of the file to read
                return all capitalized tags counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if '-' in cur_word:
                        curr_key = cur_tag
                        if curr_key not in self.hyphen_tags_dict \
                                and (self.statistics.hyphen_tags_count_dict[curr_key] >= self.threshold):
                            self.hyphen_tags_dict[curr_key] = self.n_total_features
                            self.n_total_features += 1
                            self.n_capitalized_tags += 1

    def get_contains_number_tags(self, file_path):
        """
            Extract out of text all the counts for capitalized tags
            :param file_path: full path of the file to read
                return all capitalized tags counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    flag = False
                    for ch in cur_word:
                        if '0' <= ch <= '9':
                            flag = True
                    if flag:
                        curr_key = cur_tag
                        if curr_key not in self.contain_number_tags_dict \
                                and (self.statistics.contain_number_tags_count_dict[curr_key] >= self.threshold):
                            self.contain_number_tags_dict[curr_key] = self.n_total_features
                            self.n_total_features += 1
                            self.n_capitalized_tags += 1

    def get_all_ids(self, file_path):
        self.get_word_tag_pairs(file_path)
        self.get_pre_suf_tag_pairs(file_path, True)
        self.get_pre_suf_tag_pairs(file_path, False)
        self.get_capitalized_tags(file_path)
        self.get_trigram_tags(file_path)
        self.get_trigram_tags(file_path)
        self.get_bigram_tags(file_path)
        self.get_unigram_tags(file_path)
        self.get_Allcapitalized_tags(file_path)
        self.get_hyphen_tags(file_path)
        self.get_contains_number_tags(file_path)