

```python
"""
Program to import Obama's 2006 inaugurual speech and display the frequency of the top five occuring words
Course: CSE4022 - Natural Language Processing
Author: Jacob John
"""
import nltk
import re
from nltk.corpus import inaugural 

#import Obama's speech
Obama = inaugural.words(fileids = '2009-Obama.txt')

#declaring a dictionary for word frequency
word_freq = {}
for tok in Obama:
    if tok in word_freq:
        word_freq[tok] += 1
    else:
        word_freq[tok] = 1

#finding top five most frequent words
max_dict = {}
while len(max_dict) < 5:
    max_val = 0
    for key in word_freq:
        if max_val < word_freq[key] and re.match(r'[A-Za-z]+',key) and key not in max_dict:
            max_key = key
            max_val = word_freq[key]
    max_dict[max_key] = max_val

#displaying frequency distribution
print("The five most frequent words are: ")
for key in max_dict:
    print(key+":",max_dict[key])
    
#plotting
fd = nltk.FreqDist(Obama)
fd.plot(30,cumulative=False)
```

    The five most frequent words are: 
    the: 126
    and: 105
    of: 82
    to: 66
    our: 58



![png](output_0_1.png)

