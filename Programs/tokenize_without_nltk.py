#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:10:43 2018

@author: jacobjohn

Write a program to tokenize
 a) A sentence
 b) Mutliple sentences
"""

#Tokenize an individual sentence
string_sentence = str(input("Enter a sentence: "))
string_tok = string_sentence.split()
print("Tokens are as follows: ",string_tok)

#Tokenize mulitple sentences
print("Enter multiple sentences: ")
lines = []
while True:
    line = input()
    if line:
        lines.append(line)
    else:
        break
tok = []
for t in lines:
    tok.append(t.split())
print("Tokens for multiple sentences are as follows: ",tok)