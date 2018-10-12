from tkinter import *
import subprocess
import os
import shlex

fields = 'File Path', 'Chinese Text'
entries = []

def fetch(entries):
    fin = Tk()
    fin.title('Segmented text')
    for entry in entries:
        if('File Path' in entry[0]): path = entry[1].get()
        if('Chinese Text' in entry[0]): text = entry[1].get()
    print(path, text)
    text = "echo '" + text + "' > input.txt"
    subprocess.run(text, shell=True, check=True)
    subprocess.run('bash {}/segment.sh ctb input.txt UTF-8 0 > output.txt'.format(path), shell=True)
    f = open('output.txt', 'r')
    text2 = f.read() #reading segmented text as input
    print(text2)
    msg = Message(fin, text = (text2 + "\n Copy of text is in the output.txt file"))
    msg.config(width=1000000, padx=10, pady=10)
    msg.pack()
    fin.mainloop()

def makeform(root, fields):
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=15, text=field, anchor='w')
        ent = Entry(row, width=50)
        row.pack(side=TOP, fill=X, padx=10, pady=10)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries
    
root = Tk()
root.title('Chinese Language Segmenter')
ents = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch(e)))
b1 = Button(root, text='Ok', command=(lambda e=ents: fetch(e)))
b1.pack(side=LEFT, padx=10, pady=10)
b1.flash()
b2 = Button(root, text='Quit', command=root.quit)
b2.pack(side=LEFT, padx=10, pady=10)
b2.flash()
root.mainloop()

