

```python
'''
Get two texts
Remove stop words
Map the text to vector spaces
Compute cosine(vec1, vec2)
Use SciPy
Take a call if you should do Stemming or not
'''
```


```python
#We shall calculate the similarity between two documents
#These two documents will be produced by stemming the same sentence
#The first will be the result of Porter stemming and the second will be a result of Lemmatization

#The below program uses the Porter Stemming Algorithm for stemming.
import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

document_1 = []
word_data = "It originated from the idea that there are readers who prefer learning new skills from the comforts of their drawing rooms"
# First Word tokenization
nltk_tokens = nltk.word_tokenize(word_data)
#Next find the roots of the word
for w in nltk_tokens:
    print("Actual: %s  Stem: %s"  % (w,porter_stemmer.stem(w)))
    document_1.append(porter_stemmer.stem(w))

print('\n')
port = ' '.join(word for word in document_1)
print("After porter stemming we get:",port)
    
#When we execute the above code, it produces the following result
```

    Actual: It  Stem: It
    Actual: originated  Stem: origin
    Actual: from  Stem: from
    Actual: the  Stem: the
    Actual: idea  Stem: idea
    Actual: that  Stem: that
    Actual: there  Stem: there
    Actual: are  Stem: are
    Actual: readers  Stem: reader
    Actual: who  Stem: who
    Actual: prefer  Stem: prefer
    Actual: learning  Stem: learn
    Actual: new  Stem: new
    Actual: skills  Stem: skill
    Actual: from  Stem: from
    Actual: the  Stem: the
    Actual: comforts  Stem: comfort
    Actual: of  Stem: of
    Actual: their  Stem: their
    Actual: drawing  Stem: draw
    Actual: rooms  Stem: room
    
    
    After porter stemming we get: It origin from the idea that there are reader who prefer learn new skill from the comfort of their draw room



```python
#Lemmatization is similar to stemming but it brings context to the words
document_2 = []
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk_tokens = nltk.word_tokenize(word_data)
for w in nltk_tokens:
    print("Actual: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w)))
    document_2.append(wordnet_lemmatizer.lemmatize(w))

print('\n')
lem = ' '.join(word for word in document_2)
print("After lemmatization we get:",lem)
        
#it produces the following result:
```

    Actual: It  Lemma: It
    Actual: originated  Lemma: originated
    Actual: from  Lemma: from
    Actual: the  Lemma: the
    Actual: idea  Lemma: idea
    Actual: that  Lemma: that
    Actual: there  Lemma: there
    Actual: are  Lemma: are
    Actual: readers  Lemma: reader
    Actual: who  Lemma: who
    Actual: prefer  Lemma: prefer
    Actual: learning  Lemma: learning
    Actual: new  Lemma: new
    Actual: skills  Lemma: skill
    Actual: from  Lemma: from
    Actual: the  Lemma: the
    Actual: comforts  Lemma: comfort
    Actual: of  Lemma: of
    Actual: their  Lemma: their
    Actual: drawing  Lemma: drawing
    Actual: rooms  Lemma: room
    
    
    After lemmatization we get: It originated from the idea that there are reader who prefer learning new skill from the comfort of their drawing room



```python
#Defining and computing cosine similarity
import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

vector1 = text_to_vector(port)
vector2 = text_to_vector(lem)

cosine = get_cosine(vector1, vector2)

print('Cosine:', cosine)
```

    Cosine: 0.88



```python
#Since cosine similarity is only 0.88, it shows that the documents that resulted were drastically different between 
#stemming and lemmatization

#what if we stem the lemmatized sentence again using Lancaster stemmer instead?

document_3 = []
from nltk.stem import LancasterStemmer 
stemmerLan = LancasterStemmer() 
nltk_tokens = nltk.word_tokenize(lem)
for w in nltk_tokens:
    print("Actual: %s  Lancaster: %s"  % (w,stemmerLan.stem(w)))
    document_3.append(stemmerLan.stem(w))
    
print('\n')
lanc = ' '.join(word for word in document_3)
print("After lemmatization and Lancaster stemming we get:",lanc)
```

    Actual: It  Lancaster: it
    Actual: originated  Lancaster: origin
    Actual: from  Lancaster: from
    Actual: the  Lancaster: the
    Actual: idea  Lancaster: ide
    Actual: that  Lancaster: that
    Actual: there  Lancaster: ther
    Actual: are  Lancaster: ar
    Actual: reader  Lancaster: read
    Actual: who  Lancaster: who
    Actual: prefer  Lancaster: pref
    Actual: learning  Lancaster: learn
    Actual: new  Lancaster: new
    Actual: skill  Lancaster: skil
    Actual: from  Lancaster: from
    Actual: the  Lancaster: the
    Actual: comfort  Lancaster: comfort
    Actual: of  Lancaster: of
    Actual: their  Lancaster: their
    Actual: drawing  Lancaster: draw
    Actual: room  Lancaster: room
    
    
    After lemmatization and Lancaster stemming we get: it origin from the ide that ther ar read who pref learn new skil from the comfort of their draw room



```python
#Since Lemmatization and Lancaster together are not stemming the words correctly, we shall only use lemmatization
#we shall now compare president Obama's inaugural speech in English vs Globish after Lemmatization 
#and stop word removal

#Source: http://www.jpn-globish.com/file/obama-speech.pdf

#English
document_1 = """I stand here today humbled by the task before us,
grateful for the trust you have bestowed, mindful of the
sacrifices borne by our ancestors. I thank President
Bush for his service to our nation, as well as the
generosity and cooperation he has shown throughout
this transition.
Forty-four Americans have now taken the presidential
oath. The words have been spoken during rising tides
of prosperity and the still waters of peace. Yet, every
so often the oath is taken amidst gathering clouds and
raging storms. At these moments, America has carried
on not simply because of the skill or vision of those in
high office, but because We the People have remained
faithful to the ideals of our forbearers, and true to our
founding documents.
So it has been. So it must be with this generation of
Americans.
That we are in the midst of crisis is now well
understood. Our nation is at war, against a farreaching
network of violence and hatred. Our
economy is badly weakened, a consequence of greed
and irresponsibility on the part of some, but also our
collective failure to make hard choices and prepare
the nation for a new age.
Homes have been lost; jobs shed; businesses
shuttered. Our health care is too costly; our schools
fail too many; and each day brings further evidence
that the ways we use energy strengthen our
adversaries and threaten our planet.
These are the indicators of crisis, subject to data and
statistics. Less measurable but no less profound is a
sapping of confidence across our land – a nagging
fear that America’s decline is inevitable, and that the
next generation must lower its sights.
Today I say to you that the challenges we face are
real. They are serious and they are many. They will
not be met easily or in a short span of time. But know
this, America – they will be met. On this day, we
gather because we have chosen hope over fear, unity
of purpose over conflict and discord.
On this day, we come to proclaim an end to the petty
grievances and false promises, the recriminations and
worn out dogmas, that for far too long have strangled
our politics. """

#Globish
document_2 = """I stand here today full of respect for the work before us.
I want to thank you for the trust you have given, and I
remember the sacrifices made by our ancestors. I thank
President Bush for his service to our nation, as well as
for the spirit of giving and cooperation he has shown
during this change-over.
Forty four Americans have now been sworn in as
president. The words have been spoken during rising
waves of wealth and well-being and the still waters of
peace. Yet, every so often, these words of honor are
spoken surrounded by gathering clouds and wild storms.
At these times, America has carried on not simply
because those in high office were skilled or could see
into the future. But it has been because We the People
have kept believing in the values of our first fathers, and
stayed true to the documents that created our country.
So it has been. So it must be with this modern-day
population of Americans
It is well understood now that we are in the middle of a
crisis. Our nation is at war, against a far-reaching,
organized system of violence and hate. Our economy
has been badly weakened. This is the result of extreme
desire for great wealth by some people, and failure to
act responsibly. But we have all failed to make hard
choices and to get the nation ready for a new age.
Homes have been lost; jobs given up; businesses have
closed. Our health care costs too much; our schools fail
too many; and each day brings further proof that the
ways we use energy make those against us stronger and
threaten our world
These are the ways we can measure a crisis. Another
problem is just as great, but we cannot measure it as
easily. It is the draining of our own belief in America --
a fear that America’s fall is surely coming and that
future Americans must lower their hopes.
Today I say to you that the trials we face are real. They
are serious and they are many. They will not be met
easily or in a short time. But know this, America -- they
will be met. On this day we gather because we have
chosen hope over fear. We have chosen united purpose
over fighting and over noisy argument.
On this day we come to announce an end to narrowminded
arguing, to the lies, and to the accusing and
worn out teachings that for far too long have killed our
politics. """
```


```python
##stopword removal
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
document1 = []
document2 = []

import re
document1 = re.split('; |, |\*|\n',document_1)
document2 = re.split('; |, |\*|\n',document_2)

doc1_tok = [nltk.word_tokenize(w) for w in document1]
flat_list1 = [item for sublist in doc1_tok for item in sublist]

doc2_tok = [nltk.word_tokenize(w) for w in document2]
flat_list2 = [item for sublist in doc2_tok for item in sublist]

filter1 = [w for w in flat_list1 if not w in stop_words]
filter2 = [w for w in flat_list2 if not w in stop_words]

```


```python
##Lemmatization
doc_1 = []
doc_2 = []

for w in filter1:
    doc_1.append(stemmerLan.stem(w))

lem1 = ' '.join(word for word in document_1)

for w in filter2:
    doc_2.append(stemmerLan.stem(w))

lem2 = ' '.join(word for word in document_2)

##Cosine Similarity
vector1 = text_to_vector(lem1)
vector2 = text_to_vector(lem2)

cosine = get_cosine(vector1, vector2)

print('Cosine similarity between English and Globish after Lemmatization:', cosine)
```

    Cosine similarity between English and Globish after Lemmatization: 0.9976707707618088



```python
#Tagging the words
nltk.pos_tag(filter1)
```




    [('I', 'PRP'),
     ('stand', 'VBP'),
     ('today', 'NN'),
     ('humbled', 'VBD'),
     ('task', 'NN'),
     ('us', 'PRP'),
     (',', ','),
     ('grateful', 'JJ'),
     ('trust', 'NN'),
     ('bestowed', 'VBN'),
     ('mindful', 'JJ'),
     ('sacrifices', 'NNS'),
     ('borne', 'JJ'),
     ('ancestors', 'NNS'),
     ('.', '.'),
     ('I', 'PRP'),
     ('thank', 'VBD'),
     ('President', 'NNP'),
     ('Bush', 'NNP'),
     ('service', 'NN'),
     ('nation', 'NN'),
     ('well', 'RB'),
     ('generosity', 'JJ'),
     ('cooperation', 'NN'),
     ('shown', 'VBN'),
     ('throughout', 'IN'),
     ('transition', 'NN'),
     ('.', '.'),
     ('Forty-four', 'JJ'),
     ('Americans', 'NNPS'),
     ('taken', 'VBN'),
     ('presidential', 'JJ'),
     ('oath', 'NN'),
     ('.', '.'),
     ('The', 'DT'),
     ('words', 'NNS'),
     ('spoken', 'VBN'),
     ('rising', 'VBG'),
     ('tides', 'NNS'),
     ('prosperity', 'NN'),
     ('still', 'RB'),
     ('waters', 'VBZ'),
     ('peace', 'NN'),
     ('.', '.'),
     ('Yet', 'CC'),
     ('every', 'DT'),
     ('often', 'RB'),
     ('oath', 'VBZ'),
     ('taken', 'VBN'),
     ('amidst', 'RP'),
     ('gathering', 'VBG'),
     ('clouds', 'NNS'),
     ('raging', 'VBG'),
     ('storms', 'NNS'),
     ('.', '.'),
     ('At', 'IN'),
     ('moments', 'NNS'),
     ('America', 'NNP'),
     ('carried', 'VBD'),
     ('simply', 'RB'),
     ('skill', 'JJ'),
     ('vision', 'NN'),
     ('high', 'JJ'),
     ('office', 'NN'),
     ('We', 'PRP'),
     ('People', 'VBP'),
     ('remained', 'JJ'),
     ('faithful', 'JJ'),
     ('ideals', 'NNS'),
     ('forbearers', 'NNS'),
     ('true', 'JJ'),
     ('founding', 'JJ'),
     ('documents', 'NNS'),
     ('.', '.'),
     ('So', 'RB'),
     ('.', '.'),
     ('So', 'CC'),
     ('must', 'MD'),
     ('generation', 'NN'),
     ('Americans', 'NNPS'),
     ('.', '.'),
     ('That', 'DT'),
     ('midst', 'NN'),
     ('crisis', 'NN'),
     ('well', 'RB'),
     ('understood', 'RB'),
     ('.', '.'),
     ('Our', 'PRP$'),
     ('nation', 'NN'),
     ('war', 'NN'),
     ('farreaching', 'NN'),
     ('network', 'NN'),
     ('violence', 'NN'),
     ('hatred', 'VBD'),
     ('.', '.'),
     ('Our', 'PRP$'),
     ('economy', 'NN'),
     ('badly', 'RB'),
     ('weakened', 'VBD'),
     ('consequence', 'NN'),
     ('greed', 'NN'),
     ('irresponsibility', 'NN'),
     ('part', 'NN'),
     ('also', 'RB'),
     ('collective', 'JJ'),
     ('failure', 'NN'),
     ('make', 'VBP'),
     ('hard', 'JJ'),
     ('choices', 'NNS'),
     ('prepare', 'VB'),
     ('nation', 'NN'),
     ('new', 'JJ'),
     ('age', 'NN'),
     ('.', '.'),
     ('Homes', 'VBZ'),
     ('lost', 'VBN'),
     ('jobs', 'NNS'),
     ('shed', 'VBN'),
     ('businesses', 'NNS'),
     ('shuttered', 'VBD'),
     ('.', '.'),
     ('Our', 'PRP$'),
     ('health', 'NN'),
     ('care', 'NN'),
     ('costly', 'JJ'),
     ('schools', 'NNS'),
     ('fail', 'VBP'),
     ('many', 'JJ'),
     ('day', 'NN'),
     ('brings', 'VBZ'),
     ('evidence', 'NN'),
     ('ways', 'NNS'),
     ('use', 'VBP'),
     ('energy', 'NN'),
     ('strengthen', 'NN'),
     ('adversaries', 'NNS'),
     ('threaten', 'VBP'),
     ('planet', 'NN'),
     ('.', '.'),
     ('These', 'DT'),
     ('indicators', 'NNS'),
     ('crisis', 'NN'),
     ('subject', 'JJ'),
     ('data', 'NN'),
     ('statistics', 'NNS'),
     ('.', '.'),
     ('Less', 'RBR'),
     ('measurable', 'JJ'),
     ('less', 'RBR'),
     ('profound', 'JJ'),
     ('sapping', 'VBG'),
     ('confidence', 'NN'),
     ('across', 'IN'),
     ('land', 'NN'),
     ('–', 'NN'),
     ('nagging', 'VBG'),
     ('fear', 'NN'),
     ('America', 'NNP'),
     ('’', 'NNP'),
     ('decline', 'NN'),
     ('inevitable', 'JJ'),
     ('next', 'JJ'),
     ('generation', 'NN'),
     ('must', 'MD'),
     ('lower', 'VB'),
     ('sights', 'NNS'),
     ('.', '.'),
     ('Today', 'NN'),
     ('I', 'PRP'),
     ('say', 'VBP'),
     ('challenges', 'NNS'),
     ('face', 'VBP'),
     ('real', 'JJ'),
     ('.', '.'),
     ('They', 'PRP'),
     ('serious', 'JJ'),
     ('many', 'JJ'),
     ('.', '.'),
     ('They', 'PRP'),
     ('met', 'VBD'),
     ('easily', 'RB'),
     ('short', 'JJ'),
     ('span', 'NN'),
     ('time', 'NN'),
     ('.', '.'),
     ('But', 'CC'),
     ('know', 'VBP'),
     ('America', 'NNP'),
     ('–', 'NNP'),
     ('met', 'VBD'),
     ('.', '.'),
     ('On', 'IN'),
     ('day', 'NN'),
     ('gather', 'CC'),
     ('chosen', 'VBN'),
     ('hope', 'VBP'),
     ('fear', 'JJ'),
     ('unity', 'NN'),
     ('purpose', 'VBP'),
     ('conflict', 'NN'),
     ('discord', 'NN'),
     ('.', '.'),
     ('On', 'IN'),
     ('day', 'NN'),
     ('come', 'VB'),
     ('proclaim', 'JJ'),
     ('end', 'NN'),
     ('petty', 'NN'),
     ('grievances', 'NNS'),
     ('false', 'JJ'),
     ('promises', 'NNS'),
     ('recriminations', 'NNS'),
     ('worn', 'VBP'),
     ('dogmas', 'RB'),
     ('far', 'RB'),
     ('long', 'RB'),
     ('strangled', 'JJ'),
     ('politics', 'NNS'),
     ('.', '.')]


