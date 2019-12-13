from collections import OrderedDict
import re

class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.nwords_tags_count_dict = OrderedDict()
        self.pwords_tags_count_dict = OrderedDict()

        self.words_suffix_tags_count_dict = OrderedDict()
        self.words_prefix_tags_count_dict = OrderedDict()

        self.trigram_tags_count_dict = OrderedDict()
        self.bigram_tags_count_dict = OrderedDict()
        self.unigram_tags_count_dict = OrderedDict()

        self.capitalized_tags_count_dict = OrderedDict()
        self.Allcapitalized_tags_count_dict = OrderedDict()
        self.contain_capital_tags_count_dict = OrderedDict()

        self.hyphen_tags_count_dict = OrderedDict()
        self.dot_tags_count_dict = OrderedDict()
        self.apos_tags_count_dict = OrderedDict()

        self.contain_number_tags_count_dict = OrderedDict()

        self.nnwords_tags_count_dict = OrderedDict()
        self.ppwords_tags_count_dict = OrderedDict()



    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                    relative_w_t : relative position of word and tag
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")
                splited_words.append("*_*")
                for word_idx in range(1,len(splited_words)-1):

                    pword, _ = splited_words[word_idx-1].split('_')
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    nword, _ = splited_words[word_idx+1].split('_')

                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

                    if (pword, cur_tag) not in self.pwords_tags_count_dict:
                        self.pwords_tags_count_dict[(pword, cur_tag)] = 1
                    else:
                        self.pwords_tags_count_dict[(pword, cur_tag)] += 1

                    if (nword, cur_tag) not in self.nwords_tags_count_dict:
                        self.nwords_tags_count_dict[(nword, cur_tag)] = 1
                    else:
                        self.nwords_tags_count_dict[(nword, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #

    def get_pre_suf_tag_pair_count(self, file_path, pre_suf_flag):
        """
            Extract out of text all pre-suf/tag pairs
            :param file_path: full path of the file to read
            :param pre_suf_flag: flag to choose between suffix (True) and prefix (False)
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                split_words = re.split(' |\n',line)
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
                            if (cur_part[-i:], cur_tag) not in self.words_suffix_tags_count_dict:
                                self.words_suffix_tags_count_dict[(cur_part[-i:], cur_tag)] = 1
                            else:
                                self.words_suffix_tags_count_dict[(cur_part[-i:], cur_tag)] += 1
                        else:
                            if (cur_part[:i], cur_tag) not in self.words_prefix_tags_count_dict:
                                self.words_prefix_tags_count_dict[(cur_part[:i], cur_tag)] = 1
                            else:
                                self.words_prefix_tags_count_dict[(cur_part[:i], cur_tag)] += 1

    def get_trigram_tags_count(self, file_path):
        """
            Extract out of text all trigram counts
            We padded with two '*' at the beginning and end of sentence
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n',line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                for word_idx in range(len(splited_words)-2):
                    _, prev_prev_tag = splited_words[word_idx].split('_')
                    _, prev_tag = splited_words[word_idx+1].split('_')
                    _, cur_tag = splited_words[word_idx+2].split('_')
                    curr_key = (prev_prev_tag, prev_tag, cur_tag)
                    if curr_key not in self.trigram_tags_count_dict:
                        self.trigram_tags_count_dict[curr_key] = 1
                    else:
                        self.trigram_tags_count_dict[curr_key] += 1


    def get_bigram_tags_count(self, file_path):
        """
            Extract out of text all bigram counts
            We padded with one '*' at the beginning and end of sentence
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n',line)
                del splited_words[-1]
                splited_words.insert(0, "*_*")  # add '*' at the beginning
                for word_idx in range(len(splited_words) - 1):
                    _, prev_tag = splited_words[word_idx].split('_')
                    _, cur_tag = splited_words[word_idx+1].split('_')
                    curr_key = (prev_tag, cur_tag)
                    if curr_key not in self.bigram_tags_count_dict:
                        self.bigram_tags_count_dict[curr_key] = 1
                    else:
                        self.bigram_tags_count_dict[curr_key] += 1

    def get_unigram_tags_count(self, file_path):
        """
            Extract out of text all uniram counts
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n',line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    _, cur_tag = splited_words[word_idx].split('_')
                    curr_key = cur_tag
                    if curr_key not in self.unigram_tags_count_dict:
                        self.unigram_tags_count_dict[curr_key] = 1
                    else:
                        self.unigram_tags_count_dict[curr_key] += 1

    def get_capitalized_tags_count(self, file_path):
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
                        if curr_key not in self.capitalized_tags_count_dict:
                            self.capitalized_tags_count_dict[curr_key] = 1
                        else:
                            self.capitalized_tags_count_dict[curr_key] += 1

    def get_contain_capital_tags_count(self, file_path):
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
                    flag_all = True
                    for ch in cur_word:
                        if 'A' > ch or 'Z' < ch:
                            flag_all = False

                    flag_first = 'A' <= cur_word[0] <= 'Z'

                    flag_contain = False
                    for ch in cur_word:
                        if 'A' <= ch <= 'Z':
                            flag_contain = True

                    if not flag_first and not flag_all and flag_contain:
                        curr_key = cur_tag
                        if curr_key not in self.contain_capital_tags_count_dict:
                            self.contain_capital_tags_count_dict[curr_key] = 1
                        else:
                            self.contain_capital_tags_count_dict[curr_key] += 1

    def get_Allcapitalized_tags_count(self, file_path):
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
                        if curr_key not in self.Allcapitalized_tags_count_dict:
                            self.Allcapitalized_tags_count_dict[curr_key] = 1
                        else:
                            self.Allcapitalized_tags_count_dict[curr_key] += 1

    def get_hyphen_tags_count(self, file_path):
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
                        if curr_key not in self.hyphen_tags_count_dict:
                            self.hyphen_tags_count_dict[curr_key] = 1
                        else:
                            self.hyphen_tags_count_dict[curr_key] += 1

    def get_dot_tags_count(self, file_path):
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
                    if '.' in cur_word:
                        curr_key = cur_tag
                        if curr_key not in self.dot_tags_count_dict:
                            self.dot_tags_count_dict[curr_key] = 1
                        else:
                            self.dot_tags_count_dict[curr_key] += 1

    def get_apos_tags_count(self, file_path):
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
                    if '\'' in cur_word:
                        curr_key = cur_tag
                        if curr_key not in self.apos_tags_count_dict:
                            self.apos_tags_count_dict[curr_key] = 1
                        else:
                            self.apos_tags_count_dict[curr_key] += 1

    def get_contains_number_tags_count(self, file_path):
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
                        if curr_key not in self.contain_number_tags_count_dict:
                            self.contain_number_tags_count_dict[curr_key] = 1
                        else:
                            self.contain_number_tags_count_dict[curr_key] += 1

    def get_nnwords_tags_count(self, file_path):
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n',line)
                del splited_words[-1]
                splited_words.append("*_*")  # add '*' at the beginning
                splited_words.append("*_*")  # add '*' at the beginning
                for word_idx in range(len(splited_words) - 2):
                    _, cur_tag = splited_words[word_idx].split('_')
                    nnword, _ = splited_words[word_idx + 2].split('_')
                    if (nnword, cur_tag) not in self.nnwords_tags_count_dict:
                        self.nnwords_tags_count_dict[(nnword, cur_tag)] = 1
                    else:
                        self.nnwords_tags_count_dict[(nnword, cur_tag)] += 1

    def get_ppwords_tags_count(self, file_path):
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n',line)
                del splited_words[-1]
                splited_words.insert(0,"*_*")  # add '*' at the beginning
                splited_words.insert(0,"*_*")  # add '*' at the beginning
                for word_idx in range(2,len(splited_words)):
                    _, cur_tag = splited_words[word_idx].split('_')
                    ppword, _ = splited_words[word_idx - 2].split('_')
                    if (ppword, cur_tag) not in self.ppwords_tags_count_dict:
                        self.ppwords_tags_count_dict[(ppword, cur_tag)] = 1
                    else:
                        self.ppwords_tags_count_dict[(ppword, cur_tag)] += 1

    def get_all_counts(self, file_path):
        self.get_word_tag_pair_count(file_path)
        self.get_pre_suf_tag_pair_count(file_path, True)
        self.get_pre_suf_tag_pair_count(file_path, False)
        self.get_capitalized_tags_count(file_path)
        self.get_trigram_tags_count(file_path)
        self.get_trigram_tags_count(file_path)
        self.get_bigram_tags_count(file_path)
        self.get_unigram_tags_count(file_path)
        self.get_Allcapitalized_tags_count(file_path)
        self.get_hyphen_tags_count(file_path)
        self.get_contains_number_tags_count(file_path)
        self.get_nnwords_tags_count(file_path)
        self.get_ppwords_tags_count(file_path)
        self.get_dot_tags_count(file_path)
        self.get_apos_tags_count(file_path)
        self.get_contain_capital_tags_count(file_path)
