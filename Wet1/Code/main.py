import numpy as np
import scipy
from feature_statistics_class import feature_statistics_class
import operator

path = r"..\Data\train1.wtag"
a = feature_statistics_class()
a.get_word_tag_pair_count(path)
a.get_pre_suf_tag_pair_count(path,True)
print(a.words_suffix_tags_count_dict.keys())
maximum  = max(a.words_suffix_tags_count_dict, key=a.words_suffix_tags_count_dict.get)
print(maximum, a.words_suffix_tags_count_dict[maximum])
sorted_d = sorted(a.words_suffix_tags_count_dict.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_d)
#
# a = feature_statistics_class()
# a.get_trigram_tags_count(path)
# a.get_bigram_tags_count(path)
# a.get_unigram_tags_count(path)
# print(a.bigram_tags_count_dict[('DT','VBG')])