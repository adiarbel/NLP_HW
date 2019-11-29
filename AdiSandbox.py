from collections import OrderedDict
import re
class adi_feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.words_suffix_tags_count_dict = OrderedDict()
        self.words_prefix_tags_count_dict = OrderedDict()
        # ---Add more count dictionaries here---

    def get_pre_suf_tag_pair_count(self, file_path, pre_suf_flag):
        """
            Extract out of text all pre-suf/tag pairs
            :param file_path: full path of the file to read
            :param pre_suf_flag: flag to choose between suffix (True) and prefix (False)
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                split_words = re.split(' , | \n ',line)
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
                    for i in reversed(range(cur_word_len)):
                        if pre_suf_flag:
                            if (cur_part[-i:-1], cur_tag) not in self.words_suffix_tags_count_dict:
                                self.words_suffix_tags_count_dict[(cur_part[-i:-1], cur_tag)] = 1
                            else:
                                self.words_suffix_tags_count_dict[(cur_part[-i:-1], cur_tag)] += 1
                        else:
                            if (cur_part[0:i-1], cur_tag) not in self.words_prefix_tags_count_dict:
                                self.words_prefix_tags_count_dict[(cur_part[0:i-1], cur_tag)] = 1
                            else:
                                self.words_prefix_tags_count_dict[(cur_part[0:i-1], cur_tag)] += 1

    def get_word_tag_pair_count(self, file_path):
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
                    if (cur_word, cur_tag) not in self.words_tags_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

