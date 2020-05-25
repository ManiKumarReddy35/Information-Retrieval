#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING REQUIRED MODULES

# In[1]:


import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
import pickle
import operator
import math
import more_itertools as mit


# ## LOADING TERM FREQUENCIES

# In[2]:


pickle_in = open("term_freq","rb")
dict_for_tf = pickle.load(pickle_in)


# ## LOADING DOCUMENT FREQUENCIES

# In[3]:


pickle_in = open("document_freq","rb")
dict_for_df = pickle.load(pickle_in)


# ## LOADING DICTIONARY HAVING DOCUMENT WISE MAX FREQUENCY COUNT 

# In[4]:


pickle_in = open("document_wise_max_freq","rb")
doc_wise_max_freq_count = pickle.load(pickle_in)


# ## LOADING THE TITLES

# In[5]:


pickle_in = open("fname-title","rb")
dict_titles = pickle.load(pickle_in)


# ## DATA PRE-PROCESSING

# In[6]:


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


# ## GENERATING TF-IDF VECTORS OF QUERY LENGTH W.R.T EACH DOCUMENT FOR THE TERMS PRESENT IN THE QUERY

# ###### USING DOUBLE NORMALIZATION VARIATE OF TF AND INVERSE VARIATE OF DF

# In[7]:


def get_tf_idfs(q_pp,dict_for_tf,dict_for_df,doc_wise_max_freq_count):
##term frequencies of query words in docs, key:doc_name;value:[term-frequencies of query terms w.r.t corresponding doc]
    tf_idf_wrt_docs={}
    list_of_files=list(dict_for_tf['trial'].keys())
    for doc in list_of_files:
        l=[]
        for w in word_tokenize(q_pp):
            try:
                tf = dict_for_tf[w][doc]
                df = dict_for_df[w]
                tf_idf = (0.5+(0.5*tf/doc_wise_max_freq_count[doc]))*math.log10(len(list_of_files)/df)
                ##Using DOUBLE NORMALIZATION(0.5) FOR TF AND INVERSE VARIATE OF DF
            except:
                tf_idf=0
            l.append(tf_idf)
            ## we keep on appending tf-idf of every word in the query w.r.t a doc and finally assign it as a value in dict
        tf_idf_wrt_docs[doc]= l##TF-IDF for query words w.r.t document "doc" is assigned with list "l" which contains tfidf calculation 
    return tf_idf_wrt_docs ## each doc value is of query length


# ## GENERATING TF-IDF VECTOR FOR THE QUERY

# In[8]:


def tf_idf_query(q_pp,dict_for_df):
    l=[]
    tf_q={}
    for w in word_tokenize(q_pp):
        if w not in tf_q:
            tf_q[w]=1
        else:
            tf_q[w]+=1
    max=0
    for i in tf_q:
        if(tf_q[i]>max):
            max = tf_q[i]  ## finding max frequency value which is to be used in tf DOUBLE NORM' variate
    for w in word_tokenize(q_pp):
        tf = tf_q[w]
        if dict_for_df[w]:
            idf = math.log10(471/dict_for_df[w]) ## we need to take df value present in the dictionary...
        else: 
            idf=0
        tf_idf = (0.5+(0.5*tf/max))*idf
                ##Using DOUBLE NORMALIZATION(0.5) FOR TF AND INVERSE VARIATE OF DF
        l.append(tf_idf)
    return l


# ## PERFORMING COSINE-SIMALARITY AND ADDING WEIGHTAGE TO THE DOC  IF QUERY TERM IS IN THE TITLE OF THAT DOC

# In[11]:


def cosine_sim(q_pp,tf_idf_wrt_doc,tf_idf_q,k,tw,dict_titles): ##tw = title weightage
    result={} # stores dot product value of query vector with various doc vectors
    for i in tf_idf_wrt_doc:
        dot = mit.dotproduct(tf_idf_q,tf_idf_wrt_doc[i])
        result[i] = dot
    if(tw=="n" or tw=="N"):
        result_sort = sorted(result.items(), key=operator.itemgetter(1),reverse=True)
    else:
        for w in word_tokenize(q_pp):
            for j in dict_titles:
                if w in word_tokenize(preprocess(dict_titles[j])):
                    result[j] = 1.2*result[j] ## increasing weightage of doc by 20% if query term is present in title..
        result_sort = sorted(result.items(), key=operator.itemgetter(1),reverse=True)   
    print("\n\t\t\t\tTOP-{} DOCUMENTS RETREIVED BASED ON COSINE SIMILARITY ARE::\n".format(k))
    print("="*124)
    for r in range(k):
        print(r+1,".",result_sort[r][0])


# ## CODE EXECUTION STARTS HERE

# In[22]:


q = input("ENTER THE QUERY::")
k = int(input("ENTER THE NUMBER OF DOCUMENTS TO BE RETRIEVED:: "))
tw = input("DO YOU WANT TO GIVE ADDED WEIGHTAGE TO THE DOCS IF ANY QUERY TERM IS IN THE TITLE OF THAT DOC::(Y/N)")
q_pp = preprocess(q)
print("QUERY AFTER PRE-PROCESSING::",q_pp)
print("="*124)
tf_idf_wrt_doc= get_tf_idfs(q_pp,dict_for_tf,dict_for_df,doc_wise_max_freq_count)
tf_idf_q = tf_idf_query(q_pp,dict_for_df)
cosine_sim(q_pp,tf_idf_wrt_doc,tf_idf_q,k,tw,dict_titles)


# he arrived in a valley where a stream crossed large shoulders and a bull's neck :6ablemen.txt
# Please, ladies, take a number Many maidens they would lay Pretty Bain he then did :dicksong.txt
# The first shooter shakes  the dice, turns the cup upside shooter's statement is accepted as true by the player :dicegame.txt
# defenses, and kill the necromancer Stereotypically, they had to rely on a few local hunters :knuckle.txt
# Krael had asked every empire from each Galaxy to send him some donations :sre08.txt
# 
