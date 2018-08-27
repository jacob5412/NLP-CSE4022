

```python
import nltk
from nltk.corpus import brown,inaugural,genesis
import matplotlib.pyplot as plt
```


```python
cfd = nltk.ConditionalFreqDist((len(word.lower()),fileid[:4]) 
                               for fileid in inaugural.fileids() 
                               for word in inaugural.words(fileid)
                              if word.isalpha())
```


```python
cfd.plot(figure=plt.figure(figsize=(18, 16)))
```


![png](output_2_0.png)



```python
cfd.plot(conditions=range(2,6),figure=plt.figure(figsize=(18, 16)))
```


![png](output_3_0.png)

