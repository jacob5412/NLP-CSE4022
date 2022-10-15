##run this program on a unix terminal
#
# Download stanford segmenter and unzip it
# >> wget http://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip
# >> unzip stanford-segmenter-2018-02-27.zip
# Download stanford pos tagger full and unzip it
# >> wget http://nlp.stanford.edu/software/stanford-postagger-full-2018-02-2.zip
# >> unzip stanford-postagger-full-2018-02-2.zip
# make sure the folder is in the same directory
# example chinese data for input: 应有尽有的丰富选择定将为您的旅程增添无数的赏心乐事
import os
import subprocess

text = input("Enter chinese text: ")
text = "echo '" + text + "' > input.txt"
subprocess.run(text, shell=True, check=True)
subprocess.run(
    "bash stanford-segmenter-2018-02-27/segment.sh ctb input.txt UTF-8 0 > output.txt",
    shell=True,
)
print("\nSegmented text is:")
subprocess.run("cat output.txt", shell=True, check=True)
f = open("output.txt", "r")
text2 = f.read()  # reading segmented text as input
wd = os.getcwd()
path = os.path.expanduser(
    "~/Downloads/stanford-postagger-full-2018-02-27/"
)  # directory where POS tagger was unzipped on my computer
os.chdir(path)
text2 = "echo '" + text2 + "' > input.txt"
subprocess.run(text2, shell=True, check=True)
subprocess.run(
    "bash stanford-postagger.sh models/chinese-distsim.tagger input.txt > output.txt",
    shell=True,
)
print("\nAfter adding POS tags:")
subprocess.run("cat output.txt", shell=True, check=True)
os.chdir(wd)
