#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 07:13:26 2018

@author: jacob
"""
import nltk
from nltk.corpus import brown

cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
genres = ["news", "religion", "hobbies", "science_fiction", "romance"]
modals = ["can", "could", "may", "might", "must", "will"]
cfd.tabulate(conditions=genres, samples=modals)
