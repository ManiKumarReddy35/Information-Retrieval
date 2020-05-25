#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING ALL THE REQUIRED MODULES

# In[1]:


import re
from nltk import word_tokenize
import glob  
from nltk.corpus import stopwords
import time
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
import os
import sys 
from tqdm import tqdm
import pickle
from collections import Counter
import operator
import math
import json


# ## DATA PREPROCESSING

# In[2]:


def preprocess(data):
    stopword=""
    cleantext = " "
    stopword = set(stopwords.words('english'))
    for i in word_tokenize(data):
        i=i.lower()
        if i not in stopword and len(i)>2:
            pattern1 = '[-!.?$\[\]/\}#=<>"\*:,|_~;()^\']'
            pattern2 = '[\n\n]+'
            pattern3 = '[\ \ ]+'
            wout_sc = re.sub(pattern1,'',i) #removing special characters
            wout_el = re.sub(pattern2,'\n',wout_sc) # removing empty lines (which are greater than 2)
            wout_mspaces = re.sub(pattern3,' ',wout_el) # removing multiple spaces
            cleaned_text = wout_mspaces.strip()
            i=lemmatizer.lemmatize(i)
            cleantext = cleantext+i+" "
    return cleantext.strip()


# ## READING ALL FILES IN THE DIRECTORY TO CREATE,
# ## i)DICTIONARY OF UNIQUE WORDS
# ## ii)DICTIONARY OF FILE AND CORRESPONDING PRE-PROC DATA

# In[31]:


def readfiles(path):
    files = glob.glob(path)
    total_data=""
    dict_for_df={}
    dict_for_pp={} # dictionary with key as file name, value as its data after pre-proc
    for f in files:
        head,tail = os.path.split(f)
        if(tail!="FARNON" and tail!="SRE"):
            try:
                with open(f,"r") as data:  ## READING THE DATA
                    fdata=data.read()
                    f_data_pp = preprocess(fdata)
                    dict_for_pp[tail] = f_data_pp
                    total_data = total_data+f_data_pp
            except :
                ## SNOW-MAID && ARCHIVE FILES ENTER THIS BLOCK
                with open(f,encoding='latin-1') as data:  ## READING THE DATA WHICH IS IN LATIN-1 ENCODING
                    fdata=data.read()
                    f_data_pp = preprocess(fdata)
                    dict_for_pp[tail] = f_data_pp
                    total_data = total_data+f_data_pp
        else:
            ## READING FILES PRESENT IN SUB FOLDERS
            folder_files = glob.glob(f+"\*")
            for f in folder_files:
                head,tail = os.path.split(f)
                try:
                    with open(f,"r") as data:  ## READING THE DATA
                        fdata=data.read()
                        f_data_pp = preprocess(fdata)
                        dict_for_pp[tail] = f_data_pp
                        total_data = total_data+f_data_pp
                except:
                    with open(f,encoding='latin-1') as data:  ## READING THE DATA WHICH IS IN LATIN-1 ENCODING
                        fdata=data.read()
                        f_data_pp = preprocess(fdata)
                        dict_for_pp[tail] = f_data_pp
                        total_data = total_data+f_data_pp
    
    for i in word_tokenize(total_data):
        if i not in dict_for_df:
            dict_for_df[i]=0
    return dict_for_df,dict_for_pp


# In[32]:


start  = time.time()
dict_for_df,dict_for_pp = readfiles("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\stories\\*")
end=time.time()-start
print("LENGTH OF THE VOCABULARY::",len(dict_for_df))
print("TIME TAKEN FOR GENERATING DICTIONARY::",time.strftime("%H:%M:%S", time.gmtime(end)))


# ## COMPUTING DOCUMENT FREQUENCY VALUES

# In[38]:


files = glob.glob("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\stories\\*")
start  = time.time()
for i in tqdm(dict_for_df,position=0,leave=True): ## for every term in vocab
    count=0
    for j in dict_for_pp: ## for every pre-proc data file in dict
        if i in dict_for_pp[j]:
            count+=1
    dict_for_df[i]=count ## assigning the document frequency


# ## COMPUTING TERM FREQUENCY VALUES

# In[68]:


dict_for_tf={}
for i in tqdm(dict_for_df,position=0,leave=True): ## iterating through vocab
    dict_for_tf[i]={} ##creating dictionary of dictionaries dict[term][filename] gives term freq of that term in given filename 
    for j in dict_for_pp: ## iterating through each file
        count=0
        for k in dict_for_pp[j].split():
            if(i==k):
                count+=1
        dict_for_tf[i][j]=count ## assigning tf for a term "i" in a file "j"


# ## GENERATING THE DOC-WISE TERM FREQUENCY COUNT

# In[215]:


countt=0
doc_wise_freq_count ={}  ## key = doc-id, value =  sum of term - frequencies of that doc-id
list_of_files = list(dict_for_tf['trial'].keys()) ## these are list of files read..."can be any key not only 'trial' "
vocab = list(dict_for_df.keys()) ## contains the terms in the vocabulary
for i in tqdm(list_of_files,position=0,leave=True):
    for j in vocab:
        countt=countt+dict_for_tf[j][i]
    doc_wise_freq_count[i] = countt
    countt=0


# ## GENERATING THE DOC-WISE MAX TERM FREQUENCY COUNT PRESENT

# In[224]:


doc_wise_max_freq_count ={}  ## key = doc-id, value =  sum of term - frequencies of that doc-id
list_of_files = list(dict_for_tf['trial'].keys()) ## these are list of files read..."can be any key not only 'trial' "
vocab = list(dict_for_df.keys()) ## contains the terms in the vocabulary
for i in tqdm(list_of_files,position=0,leave=True):
    max=0
    for j in vocab:
        if(dict_for_tf[j][i]>max):
            max=dict_for_tf[j][i]
    doc_wise_max_freq_count[i] = max


# ## EXTRACTING THE TITLES

# In[297]:


## LOADING SRE FOLDER TITLES..

with open('titles.json','r') as titles:  
    dict_titles = json.load(titles)

## LOADING THE TITLES OF FILES IN STORIES FOLDER...(i.e, WHICH ARE NOT IN SUB FOLDERS..)

file = open("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\stories\\index.html", 'r')
dataset={}
text = file.read().strip()
file.close()
file_name = re.findall('><A HREF="(.*)">', text)
file_title = re.findall('<BR><TD> (.*)\n', text)
for j in range(len(file_title)):
    dict_titles[file_name[j+2]] = file_title[j]
print(len(dict_titles))


# ## PICKLE THE TERM FREQUENCIES

# In[72]:


pickle_out = open("term_freq","wb")
pickle.dump(dict_for_tf, pickle_out)


# ## PICKLE THE DOCUMENT FREQUENCIES

# In[77]:


pickle_out = open("document_freq","wb")
pickle.dump(dict_for_df, pickle_out)


# ## PICKLE THE DOCUMENT WISE TERM FREQUENCIES COUNT

# In[216]:


pickle_out = open("document_wise_freq_count","wb")
pickle.dump(doc_wise_freq_count, pickle_out)


# ## PICKLE THE DOCUMENT WISE MAX TERM FREQUENY VALUE

# In[226]:


pickle_out = open("document_wise_max_freq","wb")
pickle.dump(doc_wise_max_freq_count, pickle_out)


# ## PICKLE THE TITLES

# In[311]:


pickle_out = open("fname-title","wb")
pickle.dump(dict_titles, pickle_out)


# ## LOADING THE PICKLE FILE OF TERM FREQUENCIES

# In[4]:


pickle_in = open("term_freq","rb")
dict_for_tf = pickle.load(pickle_in)


# ## LOADING THE PICKLE FILE OF DOCUMENT FREQUENCIES

# In[5]:


pickle_in = open("document_freq","rb")
dict_for_df = pickle.load(pickle_in)


# ## LOADING THE PICKLE FILE OF DOCUMENT WISE TERM FREQUENCIES COUNT

# In[6]:


pickle_in = open("document_wise_freq_count","rb")
doc_wise_freq_count = pickle.load(pickle_in)


# ## LOADING THE PICKLE FILE OF DOCUMENT WISE MAX TERM FREQUENCIES

# In[7]:


pickle_in = open("document_wise_max_freq","rb")
doc_wise_max_freq_count = pickle.load(pickle_in)


# ## LOADING THE TITLES

# In[8]:


pickle_in = open("fname-title","rb")
dict_titles = pickle.load(pickle_in)


# ## TF-IDF VARIANT-1-RAW COUNT VARIATE OF BOTH TF AND DF

# In[9]:


def tfidf_1(q_pp,dict_for_tf,dict_for_idf,k1,tw,dict_titles):
    l=[]
    list_of_files = list(dict_for_tf['trial'].keys())
    if(tw=="n" or tw=="N"):
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = tf_idf[i]*math.log10(471/df) ##mulitplying term frequency with document frequency...
            l.append(tf_idf)
    else:
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = tf_idf[i]*math.log10(471/df) ##mulitplying term frequency with document frequency...
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    tf_idf[j] = 1.2*tf_idf[j] ## giving 20% more weightage to that doc if term is present in that doc's title.
            l.append(tf_idf)
    Cdict=Counter({})
    for k in range(len(l)):
        Cdict= Cdict+Counter(l[k])  ##ADDING THE VALUES OF COMMON KEYS ACROSS THE DICTIONARIES..
    top_doc=dict(Cdict)
    top_doc_sort = sorted(top_doc.items(), key=operator.itemgetter(1),reverse=True)
    #print(top_doc_sort)
    print("THE TOP-{} DOCUMENTS RETRIVED BASED ON RAW COUNT VARIATE OF BOTH TF AND DF ARE::".format(k1))
    print("="*124)
    for m in range(k1):
        print("{}. {}".format(m+1,top_doc_sort[m][0]))


# In[ ]:





# ## TF-IDF VARIANT-2-LOG NORMALIZATION VARIATE OF TF AND INVERSE VARIATE OF DF

# In[10]:


def tfidf_2(q_pp,dict_for_tf,dict_for_idf,k1,tw,dict_titles):
    l=[]
    list_of_files = list(dict_for_tf['trial'].keys())
    if(tw=="n" or tw=="N"):
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = math.log10(1+tf_idf[i]) * math.log10(471/df) ##mulitplying term frequency with document frequency...
            l.append(tf_idf)
    else:
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = math.log10(1+tf_idf[i]) * math.log10(471/df) ##mulitplying term frequency with document frequency...
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    tf_idf[j] = 1.2*tf_idf[j]        ## giving 20% more weightage to that doc if term is present in it's title.
            l.append(tf_idf)

    Cdict=Counter({})
    for k in range(len(l)):
        Cdict= Cdict+Counter(l[k])  ##ADDING THE VALUES OF COMMON KEYS ACROSS THE DICTIONARIES..
    top_doc=dict(Cdict)
    top_doc_sort = sorted(top_doc.items(), key=operator.itemgetter(1),reverse=True)
    #print(top_doc_sort)
    print("\nTHE TOP-{} DOCUMENTS RETRIVED BASED ON LOG NORMALIZATION VARIATE OF TF AND INVERSE VARIATE OF DF ARE::".format(k1))
    print("="*124)
    for m in range(k1):
        print("{}. {}".format(m+1,top_doc_sort[m][0]))


# ## TF-IDF VARIANT-3-BINARY WEIGHTING SCHEME FOR TF AND INVERSE VARIATE OF DF

# In[11]:


def tfidf_3(q_pp,dict_for_tf,dict_for_idf,k1,tw,dict_titles):
    l=[]
    list_of_files = list(dict_for_tf['trial'].keys())
    if(tw=="n" or tw=="N"):
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                if(tf_idf[i]>0):
                    tf_idf[i] = 1 * math.log10(471/df) ##mulitplying term frequency with document frequency...
                else:
                    tf_idf[i] = 0
            l.append(tf_idf)
    else:
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                if(tf_idf[i]>0):
                    tf_idf[i] = 1 * math.log10(471/df) ##mulitplying term frequency with document frequency...
                else:
                    tf_idf[i] = 0
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    tf_idf[j] = 1.2*tf_idf[j]        ## giving 20% more weightage to that doc if term is present in it's title.
            l.append(tf_idf)
        
    Cdict=Counter({})
    for k in range(len(l)):
        Cdict= Cdict+Counter(l[k])  ##ADDING THE VALUES OF COMMON KEYS ACROSS THE DICTIONARIES..
    top_doc=dict(Cdict)
    top_doc_sort = sorted(top_doc.items(), key=operator.itemgetter(1),reverse=True)
    print("\nTHE TOP-{} DOCUMENTS RETRIVED BASED ON BINARY WEIGHTING VARIATE OF TF AND INVERSE VARIATE OF DF ARE::".format(k1))
    print("="*124)
    for m in range(k1):
        print("{}. {}".format(m+1,top_doc_sort[m][0]))


# ## TF-IDF VARIANT-4-NORMALIZED TERM FREQUENCY (DIVIDING BY TOTAL TERM FREQUENCY COUNT IN THAT DOC AFTER PRE-PROC)FOR TF AND INVERSE VARIATE OF DF

# In[12]:


def tfidf_4(q_pp,dict_for_tf,dict_for_idf,doc_wise_freq_count,k1,tw,dict_titles):
    l=[]
    list_of_files = list(dict_for_tf['trial'].keys())
    if(tw=="n" or tw=="N"):
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = (tf_idf[i]/doc_wise_freq_count[i])*math.log10(471/df) ##mulitplying term frequency with document frequency...
            l.append(tf_idf)
    else:
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                tf_idf={}
                for i in list_of_files: ## if term is not present in dictionary the tf_idf values wrt all docs will be zero..
                    tf_idf[i] = 0
            for i in tf_idf:
                tf_idf[i] = (tf_idf[i]/doc_wise_freq_count[i])*math.log10(471/df) ##mulitplying term frequency with document frequency...
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    tf_idf[j] = 1.2*tf_idf[j]        ## giving 20% more weightage to that doc if term is present in it's title.
            l.append(tf_idf)
        
    Cdict=Counter({})
    for k in range(len(l)):
        Cdict= Cdict+Counter(l[k])  ##ADDING THE VALUES OF COMMON KEYS ACROSS THE DICTIONARIES..
    top_doc=dict(Cdict)
    top_doc_sort = sorted(top_doc.items(), key=operator.itemgetter(1),reverse=True)
    #print(top_doc_sort)
    print("\nTHE TOP-{} DOCUMENTS RETRIVED BASED ON NORMALIZED TERM FREQUENCY FOR TF AND INVERSE VARIATE OF DF ARE::".format(k1))
    print("="*124)
    for m in range(k1):
        print("{}. {}".format(m+1,top_doc_sort[m][0]))


# ## TF-IDF VARIANT-5-DOUBLE NORMALIZATION(0.5) FOR TF AND INVERSE VARIATE OF DF

# In[13]:


def tfidf_5(q_pp,dict_for_tf,dict_for_idf,doc_wise_max_freq_count,k1,tw,dict_titles):
    l=[]
    list_of_files = list(dict_for_tf['trial'].keys())
    if(tw=="n" or tw=="N"):
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                print("KEY ERROR...")
            for i in tf_idf:
                tf_idf[i] = (0.5+(0.5*tf_idf[i]/doc_wise_max_freq_count[i]))*math.log10(471/df) ##mulitplying variant of term frequency with document frequency...
            l.append(tf_idf)
    else:
        for w in word_tokenize(q_pp):
            try:
                df = dict_for_idf[w] ## document frequency..
                tf_idf = dict_for_tf[w].copy() ## copying tf values..
            except:
                print("KEY ERROR...")
            for i in tf_idf:
                tf_idf[i] = (0.5+(0.5*tf_idf[i]/doc_wise_max_freq_count[i]))*math.log10(471/df) ##mulitplying variant of term frequency with document frequency...
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    tf_idf[j] = 1.2*tf_idf[j]        ## giving 20% more weightage to that doc if term is present in it's title.
            l.append(tf_idf)
        
    Cdict=Counter({})
    for k in range(len(l)):
        Cdict= Cdict+Counter(l[k])  ##ADDING THE VALUES OF COMMON KEYS ACROSS THE DICTIONARIES..
    top_doc=dict(Cdict)
    top_doc_sort = sorted(top_doc.items(), key=operator.itemgetter(1),reverse=True)
    #print(top_doc_sort)
    print("\nTHE TOP-{} DOCUMENTS RETRIVED BASED ON DOUBLE NORMALIZATION(0.5) FOR TF AND INVERSE VARIATE OF DF ARE::".format(k1))
    print("="*124)
    for m in range(k1):
        print("{}. {}".format(m+1,top_doc_sort[m][0]))


# # CODE EXECUTION STARTS HERE...

# In[25]:


q = input("ENTER THE QUERY::")
k = int(input("ENTER THE NUMBER OF DOCUMENTS TO BE RETRIEVED:: "))
tw = input("DO YOU WANT TO GIVE ADDED WEIGHTAGE TO THE TITLES WHILE MATCHING THE RELEVANT DOCS::(Y/N)")
q_pp = preprocess(q)
print("QUERY AFTER PRE-PROCESSING::",q_pp)
print("\n")
tfidf_1(q_pp,dict_for_tf,dict_for_df,k,tw,dict_titles)
tfidf_2(q_pp,dict_for_tf,dict_for_df,k,tw,dict_titles)
tfidf_3(q_pp,dict_for_tf,dict_for_df,k,tw,dict_titles)
tfidf_4(q_pp,dict_for_tf,dict_for_df,doc_wise_freq_count,k,tw,dict_titles)
tfidf_5(q_pp,dict_for_tf,dict_for_df,doc_wise_max_freq_count,k,tw,dict_titles)


# ## REFERENCES

# - https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# - My Previous NLP assignments for basic tasks
