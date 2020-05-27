#!/usr/bin/env python
# coding: utf-8

# # IMPORTING ALL THE REQUIRED MODULES

# In[6]:


import pandas as pd
from nltk import word_tokenize
import operator
import numpy as np


# # READING THE DICTIONARY

# In[7]:


with open("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\english2.txt") as f:
    dic = f.readlines()


# # THE EDIT DISTANCE CALCULATION

# In[8]:


def edit_dist(word,wlen,dic):
    edit_dist_words={}
    for i in dic:
        col_index = 0
        dic_word_len = len(i)  ## storing the length of the dictionary word
        tab = np.zeros([dic_word_len+1,wlen+1]) ## creating an empty array intialized with zeroes
        l = len(i)
        for k in range(dic_word_len+1):    ## column initialization
            tab[k][0]=l
            l-=1
        for k in range(wlen+1):           ## row initialization
            tab[dic_word_len][k]=k*2
        for l in range(len(word)):
            col_index += 1
            row_index = len(i)-1
            for l2 in range(dic_word_len):
                if(word[l]!=i[l2]):
                    a= 1+tab[row_index+1][col_index] # deletion cost
                    b= 2+tab[row_index][col_index-1] #insertion cost
                    c= 3+tab[row_index+1][col_index-1] # substitution cost
                    tab[row_index][col_index] = min(a,b,c)
                if(word[l]==i[l2]):
                    a= 1+tab[row_index+1][col_index] # deletion cost
                    b= 2+tab[row_index][col_index-1] #insertion cost
                    c= 0+tab[row_index+1][col_index-1] # substitution cost
                    tab[row_index][col_index] = min(a,b,c)
                row_index-=1
        edit_dist_words[i]=tab[0][len(word)]
    sorted_dist = sorted(edit_dist_words.items(), key=operator.itemgetter(1))
    return sorted_dist


# # FIND MINIMUM FUNCTION

# In[9]:


def min(a,b,c):
    if(a<b and a<c):
        return a
    elif(b<a and  b<c):
        return b
    else:
        return c


# # CODE EXECUTION STARTS HERE

# In[10]:


q = input("ENTER THE SENTENCE::")
q_list = word_tokenize(q.lower())
flag=0
for i in q_list:
    if i not in dic:
        print("="*127)
        print("THE WORD '{}' IS NOT PRESENT IN THE DICTIONARY...".format(i))
        k = int(input("ENTER NUMBER OF SUGGESTIONS TO BE RETRIEVED FOR WORD '{}'::".format(i)))
        flag=1
        wlen = len(i)
        sorted_l=edit_dist(i,wlen,dic)
        print("\nTHE SUGGESTIONS FOR THE WORD '{}' ARE::\n".format(i))
        for m in sorted_l[0:k]:
            print(m[0])
        
    else:
        continue
if(flag==0):
    print("ALL THE WORDS ARE PRESENT IN THE DICTIONARY....")

