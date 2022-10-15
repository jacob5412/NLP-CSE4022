#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 08:22:33 2018

@author: jacob

//https://www.jasondavies.com/wordcloud/

"""
import re

import nltk
from nltk.corpus import inaugural

Obama = inaugural.words(fileids="2009-Obama.txt")

# declare a dictionary
word_freq = {}
for tok in Obama:
    if tok in word_freq:
        word_freq[tok] += 1
    else:
        word_freq[tok] = 1

max_dict = {}

while len(max_dict) < 5:
    max_val = 0
    for key in word_freq:
        if (
            max_val < word_freq[key]
            and re.match(r"[A-Za-z]+", key)
            and key not in max_dict
        ):
            max_key = key
            max_val = word_freq[key]
    max_dict[max_key] = max_val

print("The five most frequent words are: ")
for key in max_dict:
    print(key, ":", max_dict[key])

fd = nltk.FreqDist(Obama)
fd.plot(30, cumulative=False)
