#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 08:19:21 2018

@author: jacobjohn
"""
import nltk
from nltk.corpus import inaugural
import matplotlib.pyplot

cfd = nltk.ConditionalFreqDist(
        (target,fileid[:4]) #first four characters - years
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america','citizen']
        if w.lower().startswith(target))
cfd.plot()
