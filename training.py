import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
import sys

uniqueBi=0

biWord0={}
biWord1={}

triWord1={}
triWord0Word1={}
triWord1Word2={}

unigram={}
bigram={}
trigram={}

def addUni(key,gram1):
    if key in gram1.keys():
        gram1[key]+=1
        return False
    else:
        gram1[key]=1
        return True

def addBi(word0,word1,gram2):
    if not(word0 in gram2.keys()):
        gram2[word0]={}
    if word1 in gram2[word0].keys():
        gram2[word0][word1]+=1
        return False
    else:
        gram2[word0][word1]=1
        return True

def addTri(word0,word1,word2,gram3):
    if not(word0 in gram3.keys()):
        gram3[word0]={}
    if not(word1 in gram3[word0].keys()):
        gram3[word0][word1]={}
    if word2 in gram3[word0][word1].keys():
        gram3[word0][word1][word2]+=1
        return False
    else:
        gram3[word0][word1][word2]=1
        return True

for i in range(1,len(sys.argv)):
    sentences=[]
    with open(sys.argv[i],'r') as file:
        text=''
        for line in file.readlines():
            text+=line.strip().lower()+' '
        sentences=sent_tokenize(text)
        biWord1['<s>']=1

        for sentence in sentences:
            words=['<WTF>','<s>']
            for word in word_tokenize(sentence):
                words.append(word)
            words.append('</s>')

            if i > 1:
                for j in range(0,len(words)):
                    if(not(words[j] in unigram.keys())):
                        words[j]='<unk>'

            for k in range(0, len(words)):
                addUni(words[k],unigram)
                if k < len(words)-1:
                    if addBi(words[k],words[k+1],bigram):
                        addUni(words[k+1],biWord1)
                        addUni(words[k],biWord0)
                        if(not(words[k]=='<WTF>')):
                            uniqueBi+=1
                if k < len(words)-2:
                    if addTri(words[k],words[k+1],words[k+2],trigram):
                        addUni(words[k+1],triWord1)
                        addBi(words[k+1],words[k+2],triWord1Word2)
                        addBi(words[k],words[k+1],triWord0Word1)

with open('ngram.model','w') as writeFile:
    for uni in unigram.keys():
        if not(uni=='<s>' or uni=='<WTF>'):
            writeFile.write(str(biWord1[uni]/uniqueBi)+' '+uni+'\n')
    for bi0 in bigram.keys():
        for bi1 in bigram[bi0].keys():
            if not (bi0=='<WTF>'):
                writeFile.write(str(max(triWord1Word2[bi0][bi1]-0.75,0)/triWord1[bi0])+' '+bi0+' '+bi1+' '+str(0.75*biWord0[bi0]/triWord1[bi0])+'\n')
    for tri0 in trigram.keys():
        for tri1 in trigram[tri0].keys():
            for tri2 in trigram[tri0][tri1].keys():
                writeFile.write(str(max(trigram[tri0][tri1][tri2]-0.75,0)/bigram[tri0][tri1])+' '+tri0+' '+tri1+' '+tri2+' '+str(0.75*triWord0Word1[tri0][tri1]/bigram[tri0][tri1])+'\n')