#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jacob

"""
import nltk
import tweepy
from nltk.corpus import inaugural, stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import TweetTokenizer

##Using Obama's inaugural speech
Obama = inaugural.words(fileids="2009-Obama.txt")

##stopword removal
stop_words = set(stopwords.words("english"))
filtered_sentence = [w for w in Obama if not w in stop_words]
print("After stopword removal: ", Obama)

##CMU wordlist
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[10000:10025]:
    print("CMU word list: ", entry)

##Wordnet
id = wn.synsets("motorcar")  # you get an id for subsets
id[0].lemma_names()  # head words/lemmas in the subset

##NLTK pipeline

texts = [
    """This is a sentence. So is this one."""
]  # paste text after the three quotes, organize into lines

for text in texts:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        print(tagged_words)

##Implementing tokenization
# Twitter aware tokenizer

auth = tweepy.OAuthHandler(
    "1SwbipX3sXhWt3FuouDjkwBoO", "aWrGq84Vyex3e44Lg7UENhOS5WfbqWMvPwJBwBCJJqTbHUONG8"
)
auth.set_access_token(
    "2836413980-FAAt3qj1pM51RCvw52x6E3RauFDSJc49NVzxlfQ",
    "G91NNwByh5SAPbdXAS4uQSRHsEKfaNpmLFyqC9EvfoiIT",
)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
tknzr = TweetTokenizer()
for tweet in public_tweets:
    print("Tweet: ", tweet.text)
    sent = nltk.sent_tokenize(tweet.text)
    print("Sentence tokenization: ", sent)
    word = nltk.word_tokenize(tweet.text)
    print("Word tokenization: ", word)
    tweett = tknzr.tokenize(tweet.text)
    print("Tweet tokenized: ", tweett)
    print("\n")
