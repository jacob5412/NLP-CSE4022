#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 08:10:33 2018

@author: jacobjohn

"""
import re
from nltk.corpus import inaugural

Obama = inaugural.words(fileids='2009-Obama.txt')

#declare a dictionary
word_freq = {}
for tok in Obama:
    if len(tok) <= 3:
        if tok in word_freq:
            word_freq[tok] += 1
        else:
            word_freq[tok] = 1
print(word_freq)