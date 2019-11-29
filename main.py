import numpy as np
import scipy
from feature_statistics_class import feature_statistics_class


path = r"..\Data\train1.wtag"
#a = feature_statistics_class()
#a.get_word_tag_pair_count(path)
#a.get_pre_suf_tag_pair_count(path,True)
#print(a.words_suffix_tags_count_dict.keys())



a = feature_statistics_class()
a.get_trigram_tags_count(path)
a.get_bigram_tags_count(path)
a.get_unigram_tags_count(path)
print(a.bigram_tags_count_dict[('DT','VBG')])