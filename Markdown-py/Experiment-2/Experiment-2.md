

```python
# Download stanford segmenter and unzip it
# >> wget http://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip
# >> unzip stanford-segmenter-2018-02-27.zip
# Download stanford pos tagger full and unzip it
# >> wget http://nlp.stanford.edu/software/stanford-postagger-full-2018-02-2.zip
# >> unzip stanford-postagger-full-2018-02-2.zip
# make sure the folder is in the same directory
# example chinese data for input: 应有尽有的丰富选择定将为您的旅程增添无数的赏心乐事
import subprocess
import os

path = os.path.expanduser("~/Downloads/") #directory where segmenter is stored
os.chdir(path)
text = input("Enter chinese text: ")
text = "echo '" + text + "' > input.txt"
subprocess.run(text, shell=True, check=True)
subprocess.run(
    'bash stanford-segmenter-2018-02-27/segment.sh ctb input.txt UTF-8 0 > output.txt', shell=True)
print("\nSegmented text is:")
subprocess.run('cat output.txt', shell=True, check=True)
f = open('output.txt', 'r')
text2 = f.read() #reading segmented text as input
print(text2)
wd = os.getcwd()
path = os.path.expanduser("~/Downloads/stanford-postagger-full-2018-02-27/") #directory where POS tagger was unzipped on my computer
os.chdir(path)
text2 = "echo '" + text2 + "' > input.txt"
subprocess.run(text2, shell=True, check=True)
subprocess.run('bash stanford-postagger.sh models/chinese-distsim.tagger input.txt > output.txt', shell=True)
f = open('output.txt', 'r')
text3 = f.read() #reading tagged text as input
print("\nAfter adding POS tags:")
print(text3)
subprocess.run('cat output.txt', shell=True, check=True)
os.chdir(wd)

```

    Enter chinese text: 应有尽有的丰富选择定将为您的旅程增添无数的赏心乐事
    
    Segmented text is:
    应有尽有 的 丰富 选择 定 将 为 您 的 旅程 增添 无数 的 赏心 乐事
    
    
    After adding POS tags:
    应有尽有#VV 的#DEC 丰富#JJ 选择#NN 定#VV 将#AD 为#P 您#PN 的#DEG 旅程#NN 增添#VV 无数#CD 的#DEG 赏心#NN 乐事#NN
    

