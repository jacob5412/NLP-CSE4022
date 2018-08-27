

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jacobjohn

"""
import nltk
from nltk.corpus import stopwords
from nltk.corpus import inaugural
from nltk.corpus import wordnet as wn
import tweepy

##Using Obama's inaugural speech
Obama = inaugural.words(fileids = '2009-Obama.txt')

##stopword removal
stop_words = set(stopwords.words('english')) 
filtered_sentence = [w for w in Obama if not w in stop_words]
print("After stopword removal: ", Obama)
```

    After stopword removal:  ['My', 'fellow', 'citizens', ':', 'I', 'stand', 'here', ...]



```python
##CMU wordlist
entries = nltk.corpus.cmudict.entries()
print("Number of entries: ", len(entries))
for entry in entries[10000:10025]:
    print("CMU word list: ", entry)
```

    Number of entries:  133737
    CMU word list:  ('belford', ['B', 'EH1', 'L', 'F', 'ER0', 'D'])
    CMU word list:  ('belfry', ['B', 'EH1', 'L', 'F', 'R', 'IY0'])
    CMU word list:  ('belgacom', ['B', 'EH1', 'L', 'G', 'AH0', 'K', 'AA0', 'M'])
    CMU word list:  ('belgacom', ['B', 'EH1', 'L', 'JH', 'AH0', 'K', 'AA0', 'M'])
    CMU word list:  ('belgard', ['B', 'EH0', 'L', 'G', 'AA1', 'R', 'D'])
    CMU word list:  ('belgarde', ['B', 'EH0', 'L', 'G', 'AA1', 'R', 'D', 'IY0'])
    CMU word list:  ('belge', ['B', 'EH1', 'L', 'JH', 'IY0'])
    CMU word list:  ('belger', ['B', 'EH1', 'L', 'G', 'ER0'])
    CMU word list:  ('belgian', ['B', 'EH1', 'L', 'JH', 'AH0', 'N'])
    CMU word list:  ('belgians', ['B', 'EH1', 'L', 'JH', 'AH0', 'N', 'Z'])
    CMU word list:  ('belgique', ['B', 'EH0', 'L', 'ZH', 'IY1', 'K'])
    CMU word list:  ("belgique's", ['B', 'EH0', 'L', 'JH', 'IY1', 'K', 'S'])
    CMU word list:  ('belgium', ['B', 'EH1', 'L', 'JH', 'AH0', 'M'])
    CMU word list:  ("belgium's", ['B', 'EH1', 'L', 'JH', 'AH0', 'M', 'Z'])
    CMU word list:  ('belgo', ['B', 'EH1', 'L', 'G', 'OW2'])
    CMU word list:  ('belgrade', ['B', 'EH1', 'L', 'G', 'R', 'EY0', 'D'])
    CMU word list:  ('belgrade', ['B', 'EH1', 'L', 'G', 'R', 'AA2', 'D'])
    CMU word list:  ("belgrade's", ['B', 'EH1', 'L', 'G', 'R', 'EY0', 'D', 'Z'])
    CMU word list:  ("belgrade's", ['B', 'EH1', 'L', 'G', 'R', 'AA2', 'D', 'Z'])
    CMU word list:  ('belgrave', ['B', 'EH1', 'L', 'G', 'R', 'EY2', 'V'])
    CMU word list:  ('beli', ['B', 'EH1', 'L', 'IY0'])
    CMU word list:  ('belich', ['B', 'EH1', 'L', 'IH0', 'K'])
    CMU word list:  ('belie', ['B', 'IH0', 'L', 'AY1'])
    CMU word list:  ('belied', ['B', 'IH0', 'L', 'AY1', 'D'])
    CMU word list:  ('belief', ['B', 'IH0', 'L', 'IY1', 'F'])



```python
##Wordnet
id = wn.synsets('motorcar') #you get an id for subsets
id[0].lemma_names() #head words/lemmas in the subset
```




    ['car', 'auto', 'automobile', 'machine', 'motorcar']




```python
##NLTK pipeline

texts = ["""This is a sentence. So is this one."""] #paste text after the three quotes, organize into lines

for text in texts:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        print(tagged_words)
    
```

    [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NN'), ('.', '.')]
    [('So', 'RB'), ('is', 'VBZ'), ('this', 'DT'), ('one', 'NN'), ('.', '.')]



```python
##Implementing tokenization
#Twitter aware tokenizer

auth = tweepy.OAuthHandler("", "")
auth.set_access_token("", "")

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print("Tweet: ",tweet.text)
    sent = nltk.sent_tokenize(tweet.text)
    print("Sentence tokenization: ", sent)
    word = nltk.word_tokenize(tweet.text)
    print("Word tokenization: ", word)
    print("\n")
```

    Tweet:  TOI Quick Edit | Hardik Patel returns with another quota agitation but India needs politicians who talk of growing… https://t.co/J5K6caTYyG
    Sentence tokenization:  ['TOI Quick Edit | Hardik Patel returns with another quota agitation but India needs politicians who talk of growing… https://t.co/J5K6caTYyG']
    Word tokenization:  ['TOI', 'Quick', 'Edit', '|', 'Hardik', 'Patel', 'returns', 'with', 'another', 'quota', 'agitation', 'but', 'India', 'needs', 'politicians', 'who', 'talk', 'of', 'growing…', 'https', ':', '//t.co/J5K6caTYyG']
    
    
    Tweet:  A Plane That Runs On Fuel Made by 500 Families, A First In India https://t.co/gtQ7Vob6zF #NDTVNewsBeeps https://t.co/6QudGLMe7b
    Sentence tokenization:  ['A Plane That Runs On Fuel Made by 500 Families, A First In India https://t.co/gtQ7Vob6zF #NDTVNewsBeeps https://t.co/6QudGLMe7b']
    Word tokenization:  ['A', 'Plane', 'That', 'Runs', 'On', 'Fuel', 'Made', 'by', '500', 'Families', ',', 'A', 'First', 'In', 'India', 'https', ':', '//t.co/gtQ7Vob6zF', '#', 'NDTVNewsBeeps', 'https', ':', '//t.co/6QudGLMe7b']
    
    
    Tweet:  RT @Sports_NDTV: Asian Games: Aggressive display by India in Q4, Monika scores another one. India 3-0 Thailand in women's #Hockey Pool B ma…
    Sentence tokenization:  ['RT @Sports_NDTV: Asian Games: Aggressive display by India in Q4, Monika scores another one.', "India 3-0 Thailand in women's #Hockey Pool B ma…"]
    Word tokenization:  ['RT', '@', 'Sports_NDTV', ':', 'Asian', 'Games', ':', 'Aggressive', 'display', 'by', 'India', 'in', 'Q4', ',', 'Monika', 'scores', 'another', 'one', '.', 'India', '3-0', 'Thailand', 'in', 'women', "'s", '#', 'Hockey', 'Pool', 'B', 'ma…']
    
    
    Tweet:  An eagle catches a fish with its talons. https://t.co/aae22yAiac
    Sentence tokenization:  ['An eagle catches a fish with its talons.', 'https://t.co/aae22yAiac']
    Word tokenization:  ['An', 'eagle', 'catches', 'a', 'fish', 'with', 'its', 'talons', '.', 'https', ':', '//t.co/aae22yAiac']
    
    
    Tweet:  The 1,000km rainforest trek to protect an Amazon village from an uncontacted tribe https://t.co/w3eW9DkjiH
    Sentence tokenization:  ['The 1,000km rainforest trek to protect an Amazon village from an uncontacted tribe https://t.co/w3eW9DkjiH']
    Word tokenization:  ['The', '1,000km', 'rainforest', 'trek', 'to', 'protect', 'an', 'Amazon', 'village', 'from', 'an', 'uncontacted', 'tribe', 'https', ':', '//t.co/w3eW9DkjiH']
    
    
    Tweet:  RT @moviesndtv: After Roka, #PriyankaChopra, #NickJonas Pick Malibu For A Brunch Date https://t.co/fOLuZRrFMB https://t.co/OHpndcvPwK
    Sentence tokenization:  ['RT @moviesndtv: After Roka, #PriyankaChopra, #NickJonas Pick Malibu For A Brunch Date https://t.co/fOLuZRrFMB https://t.co/OHpndcvPwK']
    Word tokenization:  ['RT', '@', 'moviesndtv', ':', 'After', 'Roka', ',', '#', 'PriyankaChopra', ',', '#', 'NickJonas', 'Pick', 'Malibu', 'For', 'A', 'Brunch', 'Date', 'https', ':', '//t.co/fOLuZRrFMB', 'https', ':', '//t.co/OHpndcvPwK']
    
    
    Tweet:  Living in the company of the dead in Kerala village https://t.co/JR6NBTLE10 via @TOICitiesNews https://t.co/50dBWZIqLq
    Sentence tokenization:  ['Living in the company of the dead in Kerala village https://t.co/JR6NBTLE10 via @TOICitiesNews https://t.co/50dBWZIqLq']
    Word tokenization:  ['Living', 'in', 'the', 'company', 'of', 'the', 'dead', 'in', 'Kerala', 'village', 'https', ':', '//t.co/JR6NBTLE10', 'via', '@', 'TOICitiesNews', 'https', ':', '//t.co/50dBWZIqLq']
    
    
    Tweet:  Lead story now on https://t.co/Fbzw6mR9Q5: https://t.co/kTxZCPyO6e
    
    #NDTVLeadStory https://t.co/8fFj4GMXQl
    Sentence tokenization:  ['Lead story now on https://t.co/Fbzw6mR9Q5: https://t.co/kTxZCPyO6e\n\n#NDTVLeadStory https://t.co/8fFj4GMXQl']
    Word tokenization:  ['Lead', 'story', 'now', 'on', 'https', ':', '//t.co/Fbzw6mR9Q5', ':', 'https', ':', '//t.co/kTxZCPyO6e', '#', 'NDTVLeadStory', 'https', ':', '//t.co/8fFj4GMXQl']
    
    
    Tweet:  RT @Gadgets360: PUBG HP Omen Challenger Series Announced https://t.co/wIbcdFz9Iy #PUBG
    Sentence tokenization:  ['RT @Gadgets360: PUBG HP Omen Challenger Series Announced https://t.co/wIbcdFz9Iy #PUBG']
    Word tokenization:  ['RT', '@', 'Gadgets360', ':', 'PUBG', 'HP', 'Omen', 'Challenger', 'Series', 'Announced', 'https', ':', '//t.co/wIbcdFz9Iy', '#', 'PUBG']
    
    
    Tweet:  Another mass shooting unfolded in Florida, this time at a tournament for competitive players of the football video… https://t.co/WtdUfXhZgv
    Sentence tokenization:  ['Another mass shooting unfolded in Florida, this time at a tournament for competitive players of the football video… https://t.co/WtdUfXhZgv']
    Word tokenization:  ['Another', 'mass', 'shooting', 'unfolded', 'in', 'Florida', ',', 'this', 'time', 'at', 'a', 'tournament', 'for', 'competitive', 'players', 'of', 'the', 'football', 'video…', 'https', ':', '//t.co/WtdUfXhZgv']
    
    
    Tweet:  RT @toisports: .@Pvsindhu1 to fight for gold, @NSaina fetches bronze in Asian Games 
    
    READ: https://t.co/yZqAWYHvQP
    
    #badminton #PVSindhu #…
    Sentence tokenization:  ['RT @toisports: .', '@Pvsindhu1 to fight for gold, @NSaina fetches bronze in Asian Games \n\nREAD: https://t.co/yZqAWYHvQP\n\n#badminton #PVSindhu #…']
    Word tokenization:  ['RT', '@', 'toisports', ':', '.', '@', 'Pvsindhu1', 'to', 'fight', 'for', 'gold', ',', '@', 'NSaina', 'fetches', 'bronze', 'in', 'Asian', 'Games', 'READ', ':', 'https', ':', '//t.co/yZqAWYHvQP', '#', 'badminton', '#', 'PVSindhu', '#', '…']
    
    
    Tweet:  RT @Sports_NDTV: Asian Games: Rani Rampal scores another one early in Q4.  India 2-0 Thailand in women's #Hockey Pool B match
    
    #AsianGames…
    Sentence tokenization:  ['RT @Sports_NDTV: Asian Games: Rani Rampal scores another one early in Q4.', "India 2-0 Thailand in women's #Hockey Pool B match\n\n#AsianGames…"]
    Word tokenization:  ['RT', '@', 'Sports_NDTV', ':', 'Asian', 'Games', ':', 'Rani', 'Rampal', 'scores', 'another', 'one', 'early', 'in', 'Q4', '.', 'India', '2-0', 'Thailand', 'in', 'women', "'s", '#', 'Hockey', 'Pool', 'B', 'match', '#', 'AsianGames…']
    
    
    Tweet:  The 1,000km rainforest trek to protect an Amazon village from an uncontacted tribe | Dom Phillips and Gary Calton https://t.co/w3eW9D2HU7
    Sentence tokenization:  ['The 1,000km rainforest trek to protect an Amazon village from an uncontacted tribe | Dom Phillips and Gary Calton https://t.co/w3eW9D2HU7']
    Word tokenization:  ['The', '1,000km', 'rainforest', 'trek', 'to', 'protect', 'an', 'Amazon', 'village', 'from', 'an', 'uncontacted', 'tribe', '|', 'Dom', 'Phillips', 'and', 'Gary', 'Calton', 'https', ':', '//t.co/w3eW9D2HU7']
    
    
    Tweet:  Chelsea manager Maurizio Sarri surprised by Newcastle United’s tactics https://t.co/eiDz8KxtEk
    Sentence tokenization:  ['Chelsea manager Maurizio Sarri surprised by Newcastle United’s tactics https://t.co/eiDz8KxtEk']
    Word tokenization:  ['Chelsea', 'manager', 'Maurizio', 'Sarri', 'surprised', 'by', 'Newcastle', 'United', '’', 's', 'tactics', 'https', ':', '//t.co/eiDz8KxtEk']
    
    
    Tweet:  Here's how we built a voicebot for a fast food chain.
    https://t.co/GwYyiQosxs
    Sentence tokenization:  ["Here's how we built a voicebot for a fast food chain.", 'https://t.co/GwYyiQosxs']
    Word tokenization:  ['Here', "'s", 'how', 'we', 'built', 'a', 'voicebot', 'for', 'a', 'fast', 'food', 'chain', '.', 'https', ':', '//t.co/GwYyiQosxs']
    
    
    Tweet:  Traveling with your dogs just got a lot easier https://t.co/r6HYB44tGt
    Sentence tokenization:  ['Traveling with your dogs just got a lot easier https://t.co/r6HYB44tGt']
    Word tokenization:  ['Traveling', 'with', 'your', 'dogs', 'just', 'got', 'a', 'lot', 'easier', 'https', ':', '//t.co/r6HYB44tGt']
    
    
    Tweet:  Assad’s Syria recorded its own atrocities. The world can’t ignore them | Sara Afshar https://t.co/UCZX6VAhPz
    Sentence tokenization:  ['Assad’s Syria recorded its own atrocities.', 'The world can’t ignore them | Sara Afshar https://t.co/UCZX6VAhPz']
    Word tokenization:  ['Assad', '’', 's', 'Syria', 'recorded', 'its', 'own', 'atrocities', '.', 'The', 'world', 'can', '’', 't', 'ignore', 'them', '|', 'Sara', 'Afshar', 'https', ':', '//t.co/UCZX6VAhPz']
    
    
    Tweet:  Ethereal underworld: exploring Helsinki's colossal new art bunker https://t.co/OhCqYbEvmt
    Sentence tokenization:  ["Ethereal underworld: exploring Helsinki's colossal new art bunker https://t.co/OhCqYbEvmt"]
    Word tokenization:  ['Ethereal', 'underworld', ':', 'exploring', 'Helsinki', "'s", 'colossal', 'new', 'art', 'bunker', 'https', ':', '//t.co/OhCqYbEvmt']
    
    
    Tweet:  While Trump tweets about white farmers, indigenous peoples face annihilation | Raj Patel https://t.co/4GOL0hVSOS
    Sentence tokenization:  ['While Trump tweets about white farmers, indigenous peoples face annihilation | Raj Patel https://t.co/4GOL0hVSOS']
    Word tokenization:  ['While', 'Trump', 'tweets', 'about', 'white', 'farmers', ',', 'indigenous', 'peoples', 'face', 'annihilation', '|', 'Raj', 'Patel', 'https', ':', '//t.co/4GOL0hVSOS']
    
    
    Tweet:  RT @Gadgets360: The 10.or D2 sports a 18:9 display and a 13-megapixel Sony sensor. But does it offer the best value at Rs. 7,999? We review…
    Sentence tokenization:  ['RT @Gadgets360: The 10.or D2 sports a 18:9 display and a 13-megapixel Sony sensor.', 'But does it offer the best value at Rs.', '7,999?', 'We review…']
    Word tokenization:  ['RT', '@', 'Gadgets360', ':', 'The', '10.or', 'D2', 'sports', 'a', '18:9', 'display', 'and', 'a', '13-megapixel', 'Sony', 'sensor', '.', 'But', 'does', 'it', 'offer', 'the', 'best', 'value', 'at', 'Rs', '.', '7,999', '?', 'We', 'review…']
    
    

