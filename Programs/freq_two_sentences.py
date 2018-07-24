#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:34:25 2018

@author: jacobjohn

Write a program to count the frequency of tokenzied words from two sentences
"""
#input two sentences
print("Enter two sentences: ")
lines = []
i = 0
while(i < 2):
    line = input()
    lines.append(line)
    i += 1

#declare a dictionary
word_freq = {}
for tok in lines:
    tok = tok.split()
    for t in tok:
        if t in word_freq:
            word_freq[t] += 1
        else:
            word_freq[t] = 1
            
print("Frequency distribution of words are: ",word_freq)