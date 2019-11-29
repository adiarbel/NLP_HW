from collections import OrderedDict
import re

class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.words_suffix_tags_count_dict = OrderedDict()
        self.words_prefix_tags_count_dict = OrderedDict()


        self.trigram_tags_count_dict = OrderedDict()
        self.bigram_tags_count_dict = OrderedDict()
        self.unigram_tags_count_dict = OrderedDict()

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n ', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

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
                split_words = re.split(' | \n ',line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    cur_part = ""
                    cur_word_len = len(cur_word)
                    pre_suf_slice_size = min(cur_word_len, 4)
                    if pre_suf_flag:
                        cur_part = cur_word[-pre_suf_slice_size:-1]
                    else:
                        cur_part = cur_word[0:pre_suf_slice_size-1]
                    for i in reversed(range(cur_word_len-1)):
                        if pre_suf_flag:
                            if (cur_part[-i:-1], cur_tag) not in self.words_suffix_tags_count_dict:
                                self.words_suffix_tags_count_dict[(cur_part[-i-1:-1], cur_tag)] = 1
                            else:
                                self.words_suffix_tags_count_dict[(cur_part[-i-1:-1], cur_tag)] += 1
                        else:
                            if (cur_part[0:i-1], cur_tag) not in self.words_prefix_tags_count_dict:
                                self.words_prefix_tags_count_dict[(cur_part[0:i], cur_tag)] = 1
                            else:
                                self.words_prefix_tags_count_dict[(cur_part[0:i], cur_tag)] += 1

    def get_trigram_tags_count(self, file_path):
        """
            Extract out of text all trigram counts
            We padded with one '*' at the beginning and end of sentence
            This is because the information captured by "* * tag" is also captured by the bigram
            :param file_path: full path of the file to read
                return all trigram counts with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n ',line)
                splited_words[-1] = "*_*"  # add '*' at the end
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
                splited_words = re.split(' |\n ',line)
                splited_words[-1] = "*_*"  # add '*' at the end
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
                splited_words = re.split(' |\n ',line)
                for word_idx in range(len(splited_words)):
                    _, cur_tag = splited_words[word_idx].split('_')
                    curr_key = cur_tag
                    if curr_key not in self.unigram_tags_count_dict:
                        self.unigram_tags_count_dict[curr_key] = 1
                    else:
                        self.unigram_tags_count_dict[curr_key] += 1