import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
import sys
import numpy as np

uniPhi={}

biPhi={}
biLam={}

triPhi={}
triLam={}

numTri=0
perplexity=0

orderedTrigrams=[]

def observed(order,gram,model):
    if order==3:
        if gram[0] in model.keys():
            if gram[1] in model[gram[0]].keys():
                if gram[2] in model[gram[0]][gram[1]].keys():
                    return True
    elif order==2:
        if gram[0] in model.keys():
            if gram[1] in model[gram[0]].keys():
                return True
    return False

def pKN3(trigram,phi,lam):
    if observed(3,trigram,phi[3]):
        return phi[3][trigram[0]][trigram[1]][trigram[2]]+lam[2][trigram[0]][trigram[1]]*pKN2((trigram[1],trigram[2]),phi,lam)
    elif observed(2,(trigram[0],trigram[1]),lam[2]):
        return lam[2][trigram[0]][trigram[1]]*pKN2((trigram[1],trigram[2]),phi,lam)
    else:
        return pKN2((trigram[1],trigram[2]),phi,lam)

def pKN2(bigram,phi,lam):
    if observed(2,bigram,phi[2]):
        return phi[2][bigram[0]][bigram[1]]+lam[1][bigram[0]]*pKN1(bigram[1],phi,lam)
    else:
        return lam[1][bigram[0]]*pKN1(bigram[1],phi,lam)

def pKN1(unigram,phi,lam):
    return phi[1][unigram]

with open(sys.argv[1], 'r') as file:
    for line in file.readlines():
        gram=line.strip().split(' ')
        if len(gram)==2:
            uniPhi[gram[1]]=float(gram[0])
        elif len(gram)==4:
            if not(gram[1] in biPhi.keys()):
                biPhi[gram[1]]={}
            biPhi[gram[1]][gram[2]]=float(gram[0])
            biLam[gram[1]]=float(gram[3])
        elif len(gram)==5:
            if not(gram[1] in triPhi.keys()):
                triPhi[gram[1]]={}
                triLam[gram[1]]={}
            if not(gram[2] in triPhi[gram[1]].keys()):
                triPhi[gram[1]][gram[2]]={}
            triPhi[gram[1]][gram[2]][gram[3]]=float(gram[0])
            triLam[gram[1]][gram[2]]=float(gram[4])

#int values being the order
Phi={1:uniPhi,2:biPhi,3:triPhi}
Lam={1:biLam,2:triLam}

with open(sys.argv[2], 'r') as test:
    text=''
    for line in test.readlines():
        text+=line.strip().lower()+' '
    sentences=sent_tokenize(text)

    count=3

    for sentence in sentences:
        words=['<WTF>','<s>']
        for word in word_tokenize(sentence):
            words.append(word)
        words.append('</s>')

        for j in range(0,len(words)):
            if not(words[j] in uniPhi.keys() or words[j]=='<s>' or words[j]=='<WTF>'):
                words[j]='<unk>'

        trigrams = list(nltk.trigrams(words))

        for trigram in trigrams:
            numTri+=1
            prob=pKN3(trigram,Phi,Lam)
            perplexity+=np.log2(prob)
            if count>0:
                orderedTrigrams.append(trigram)
        
        count-=1
    
    print('a)')
    print('Perplexity: '+str(np.exp2(-(1/numTri)*perplexity)))

    print('b)')
    print('word surprisal entropy entropy-reduction')
    
    prevEntropy=0
    for tri in orderedTrigrams:
        pKN=pKN3(tri,Phi,Lam)
        surprisal=-np.log2(pKN)
        entropy=0
        if not(tri[2]=='</s>'):
            for vocab in uniPhi.keys():
                enpKN=pKN3((tri[1],tri[2],vocab),Phi,Lam)
                entropy+=enpKN*np.log2(enpKN)
        entropy=-entropy
        print(tri[2]+' '+str(surprisal)+' '+str(entropy)+' '+str(max(prevEntropy-entropy,0)))
        prevEntropy=entropy