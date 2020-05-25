#!/usr/bin/env python
# coding: utf-8

# # IMPORTING ALL THE REQUIRED MODULES

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
import time
import operator


# # DATA PRE-PROCESSING

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


# # PROCEDURE FOR READING THE FILES AND COMPUTING JACCARD SCORE

# In[3]:


def readfiles(path,query):
    files = glob.glob(path) 
    jac_score={}
    query_pp = preprocess(query)
    print("QUERY AFTER PRE-PROCESSING::",query_pp)
    for f in files:
        head,tail = os.path.split(f)
        if(tail!="FARNON" and tail!="SRE"):
            try:
                with open(f,"r") as data:  ## READING THE DATA
                    fdata=data.read()
                    f_data_pp = preprocess(fdata)
                    jac_score[tail]= len(set(word_tokenize(query_pp)) & set(word_tokenize(f_data_pp)))/len(set(word_tokenize(query_pp)) | set(word_tokenize(f_data_pp)))
                    #JACCARD CO-EFFICIENT FORMULA...
            except :
                ## SNOW-MAID && ARCHIVE FILES ENTER THIS BLOCK
                with open(f,encoding='latin-1') as data:  ## READING THE DATA WHICH IS IN LATIN-1 ENCODING
                    fdata=data.read()
                    f_data_pp = preprocess(fdata)
                    jac_score[tail]= len(set(word_tokenize(query_pp)) & set(word_tokenize(f_data_pp)))/len(set(word_tokenize(query_pp)) | set(word_tokenize(f_data_pp)))
                    #JACCARD CO-EFFICIENT FORMULA...
        else:
            ## READING FILES PRESENT IN FOLDERS
            folder_files = glob.glob(f+"\*")
            for f in folder_files:
                try:
                    with open(f,"r") as data:  ## READING THE DATA
                        fdata=data.read()
                        f_data_pp = preprocess(fdata)
                        jac_score[tail]= len(set(word_tokenize(query_pp)) & set(word_tokenize(f_data_pp)))/len(set(word_tokenize(query_pp)) | set(word_tokenize(f_data_pp)))
                        #JACCARD CO-EFFICIENT FORMULA...
                except:
                    with open(f,encoding='latin-1') as data:  ## READING THE DATA WHICH IS IN LATIN-1 ENCODING
                        fdata=data.read()
                        f_data_pp = preprocess(fdata)
                        jac_score[tail]= len(set(word_tokenize(query_pp)) & set(word_tokenize(f_data_pp)))/len(set(word_tokenize(query_pp)) | set(word_tokenize(f_data_pp)))
                        #JACCARD CO-EFFICIENT FORMULA...
                    
    jac_score_sort = sorted(jac_score.items(), key=operator.itemgetter(1),reverse=True)
    #SORTING FILES IN DESCENDING ORDER BASED ON ASSIGNED JACCARD SCORE
    return jac_score_sort      


# # CODE EXECUTION STARTS HERE...

# In[8]:


q = input("ENTER THE QUERY::") #"associate with The Three Gables"
k = int(input("ENTER THE NUMBER OF RELEVANT DOCUMENTS THAT ARE TO BE RETRIEVED::"))
start=time.time()
jac_score = readfiles("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\stories\\*",q)
end=time.time()-start
print("TIME TAKEN FOR READING::",time.strftime("%H:%M:%S", time.gmtime(end)))
print("TOP-{} DOCUMENTS FOR THE QUERY '{}' ARE::\n".format(k,q))
for m in jac_score[0:k]:
    print("-",m[0])


# In[ ]:


sre08.txt

