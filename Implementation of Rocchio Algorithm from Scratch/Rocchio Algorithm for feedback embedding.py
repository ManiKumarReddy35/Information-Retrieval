#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING REQUIRED MODULES

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
import numpy as np
import pandas as  pd
from prettytable import PrettyTable
import math
import warnings
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# ## DATA PRE-PROCESSING

# In[2]:


def preprocess(data):
    stopword=""
    cleantext = " "
    stopword = set(stopwords.words('english'))
    for i in word_tokenize(data):
        i=i.lower()
        if i not in stopword and len(i)>2:
            pattern1 = '[!.?$\[\]/\}#=<>"\*:,|_~;()^\']'
            pattern2 = '[\n\n]+'
            pattern3 = '[\ \ ]+'
            wout_sc = re.sub(pattern1,'',i) #removing special characters
            wout_el = re.sub(pattern2,'\n',wout_sc) # removing empty lines (which are greater than 2)
            wout_mspaces = re.sub(pattern3,' ',wout_el) # removing multiple spaces
            cleaned_text = wout_mspaces.strip()
            cleaned_text=lemmatizer.lemmatize(cleaned_text)
            cleantext = cleantext+cleaned_text+" "
    return cleantext.strip()+" "+"path"  ## using as a key furtherly to get all file names...


# ## READING THE DATA & BUILDING TFs , DFs
# ## CORPUS CONSIDERED :: "20 news groups" 
# ## FOLDERS CONSIDERED: "comp.graphics", "sci.med,talk.politics.misc", "rec.sport.hockey", "sci.space". 
# ## so give the query as i/p from docs of above folders only,if not we encounter "keyerrors"

# In[3]:


def readfiles(path):
    folders = glob.glob(path)
    pos_list_with_tf={}
    df_dict={}
    for f in tqdm(folders,leave=True,position=0):
        head,tail = os.path.split(f)
        if(tail=="comp.graphics"or tail=="sci.med"or tail=="talk.politics.misc"or tail=="rec.sport.hockey"or tail=="sci.space"):
            for file in glob.glob(f+"\*"):
                flag=0
                head1,tail1 = os.path.split(file)
                with open(file) as newlinetest:   ## identifying the first new line in the data.. to remove meta-data
                    indi = newlinetest.readlines()
                for newline in range(len(indi)):
                    if(indi[newline]!="\n"):
                        continue
                    break
                filedata=[]
                try:
                    with open(file,"r") as data:  ## READING THE DATA ignoring the meta-data
                        filedata.append(data.readlines()[newline+1:])
                        for l in range(len(filedata)):
                            inter_data = ' '.join(filedata[l])
                    preproc_data = preprocess(inter_data)
                    for i in word_tokenize(preproc_data):
                        if i not in pos_list_with_tf: ## repeated words over-ride
                            pos_list_with_tf[i] = {tail+"_"+tail1:1+math.log10(preproc_data.count(i))}
                            ## using log-normalization variate of tf..
                        else:
                            pos_list_with_tf[i].update({tail+"_"+tail1:1+math.log10(preproc_data.count(i))})
                    unique_words = set(word_tokenize(preproc_data))
                    for j in list(unique_words):
                        if j not in df_dict:
                            df_dict[j]=1
                        else:
                            df_dict[j]+=1
                except e:
                    print("Exception Occured..")
    
    for i in df_dict:
        df_dict[i] = math.log10(5000/df_dict[i])  ## performing log(N/DF) normalization.. here total docs are 5000
    return pos_list_with_tf,df_dict


# ## GENERATING VECTORS OF QUERY SIZE 

# In[27]:


def gen_vec(q_pp,pos_list_with_tf,df_dict):
    temp_vec={}
    q_vec=[]
    doc_vec={}
    list_of_files = list(set(pos_list_with_tf['path'].keys())) ## list of files in the corpus(word "path" is present in all docs)
    for i in q_pp.split(): ## computing term frequencies for query..
        if i not in q_vec:
            temp_vec[i]=1
        else:
            temp_vec[i]+=1
    
    for i in temp_vec:     ## computing tf-idf of every term in query..
        temp_vec[i] = (1+math.log10(temp_vec[i]))*df_dict[i]

    for i in temp_vec: ## assigning the values of the keys to form a vector
        q_vec.append(temp_vec[i])
    
    for i in list_of_files:
        temp=[]
        for j in q_pp.split():
            try:
                temp.append(pos_list_with_tf[j][i]*df_dict[j]) #computing tf-idf and adding the value to the set
            except:
                temp.append(int(0)) #if the Q term is not present in that file its tf-idf will be 0..
        doc_vec[i] = temp
    return q_vec,doc_vec


# ## COMPUTING COSINE-SIMILARITY AND RETURNING THE FINAL DOCS

# In[28]:


def compute_cossim(q_vec,doc_vec,k,ite,rel_docs):
    q_vec_np = np.array(q_vec)
    results={}
    for i in doc_vec:
        A_dot_B = (q_vec_np) @ np.array(doc_vec[i]) 
        mod_A = np.sqrt(q_vec_np @ q_vec_np)
        mod_B = np.sqrt(np.array(doc_vec[i]) @ np.array(doc_vec[i]))
        value = A_dot_B/(mod_A*mod_B)
        if math.isnan(value):
            results[i] = -99999999  #if any of the vector magnitude is 0,cosine-sim becomes 0/0 which is undefined so assigning 
                                    #some large negative value
        else:
            results[i] = value
    results = dict(sorted(results.items(), key=operator.itemgetter(1),reverse=True))
    file_names=list(results.keys())
    t = PrettyTable(['Rank','Document Name','Score'])
    if(ite==-1):
        for i in range(k):
            t.add_row([i+1,file_names[i],results[file_names[i]]])
        list_of_docs=[]
        for row in t:    ## getting the document names from pretty table to incorporate feedback
            row.border = False
            row.header = False
            list_of_docs.append(row.get_string(fields=['Document Name']).strip())
        print("QUERY VECTOR::",q_vec)
        print(t)
    else:
        for i in range(k):
            if (file_names[i] in rel_docs): 
                t.add_row([i+1,"**"+file_names[i],results[file_names[i]]])
            else:
                t.add_row([i+1,file_names[i],results[file_names[i]]])
        list_of_docs=[]
        for row in t:    ## getting the document names from pretty table to incorporate feedback
            row.border = False
            row.header = False
            list_of_docs.append(row.get_string(fields=['Document Name']).strip())
        print("QUERY VECTOR AFTER ITERATION-{} IS, \n{}".format(ite+1,q_vec))
        print(t)
    return list_of_docs


# ## COMPUTING & RETURNING CENTROID FOR GIVEN DOCS

# In[29]:


def compute_centroid(q_pp,doc_list,doc_vec):
    centroid = []
    for i in range(len(q_pp.split())): ## traversing for the entire words of the query..
        temp=[]
        for j in doc_list: ## traversing all the docs..
            try:
                temp.append(doc_vec[j][i]) ## retrieving tf-idf value of word-i from every document vector in doc_list..
            except:
                print("FILE '{}' NOT FOUND.. PLEASE VERIFY THE FILE-NAME ENTERED..".format(j))
                sys.exit()
        centroid.append(sum(temp)/len(temp)) ## computing centroid for first co-ordinates of provided vector
    return centroid


# ## COMPUTING M.A.P OF GIVEN QUERIES AFTER EVERY ITERATION

# In[30]:


def MAP(avg_prec,k2,query_count):
    if(k2>0):
        num_iter = k2
        sub_lists = [avg_prec[x:x+num_iter] for x in range(0, len(avg_prec), num_iter)] ##sub-lists having avg-precisions iteration wise
        SAP = list(map(sum, zip(*sub_lists)))                                       ##(contnd..) num of sub-lists = num of queries
        ## sum of average precisions iteration wise for given queries
        ## SAP[0] indicates sum of average precisions of given queries after iteration-1 of feedback embedding.. 
        ## "SAP[0]/query_count" gives MAP of given queries after iteration-1
        for i in range(len(SAP)):
            print("="*125)
            print("MAP WITH EMBEDDING OF FEEDBACK FOR GIVEN {} QUERIES AFTER ITERATION-{} IS {}".format(query_count,i+1,SAP[i]/query_count))
    else:
        print("="*125)
        print("MAP WITHOUT EMBEDDING OF ANY FEEDBACK FOR ALL GIVEN {} QUERIES IS {}".format(query_count,sum(avg_prec)/query_count))


# ## CODE EXECUTION STARTS HERE

# In[34]:


query_count = int(input("ENTER NUMBER OF QUERIES::"))
avg_prec=[]
for que in range(query_count):
    q = input("ENTER THE QUERY-{}::".format(que+1))
    feed_back = 0
    k = int(input("ENTER THE NUMBER OF DOCS TO BE RETRIEVED(K)::"))
    q_pp = preprocess(q)
    print("QUERY AFTER PRE-PROCESSING::",q_pp)
    print("="*125)
    pos_list_with_tf,df_dict = readfiles("C:\\Users\\Sai Kumar\\Desktop\\SEM-2\\IR\\20NEWS\\20_newsgroups\\*")
    q_vec,doc_vec = gen_vec(q_pp,pos_list_with_tf,df_dict)
    new_query_vec = q_vec.copy()
    print("\n\t\t\t DOCS RETRIEVED BASED ON COSINE-SIMILARITY SCORE WITHOUT ANY EMBEDDING OF FEEDBACK FOR QUERY \n\t\t\t\t\t'{}'\n".format(q))
    print("="*125)
    doc_names = compute_cossim(q_vec,doc_vec,k,-1,[])
    print("="*125)
    dec = input("DO YOU WANT TO EMBED FEEDBACK(Y/N);IF YES EMBED FOR ALL QUERIES EQUAL NUMBER OF TIMES::")
    if(dec=="Y" or dec =="y"):
        rel_doc_count = int(input("ENTER THE PERCENTAGE OF DOCS TO BE MARKED RELEVANT::"))
        ## count of docs that are to be marked relevant among retrieved
        rel_doc_count = int((rel_doc_count/100)*k)
        k2 = int(input("HOW MANY TIMES YOU WANT TO INCORPORATE FEEDBACK TO THE MODEL::"))
        for ite in range(k2):
            feed_back+=1
            rel_docs=[]
            irrel_docs=[]
            d_name = input("ENTER THE NAME OF THE DIRECTORY FOR MARKING THE RELEVANT DOCS::")
            ## giving the target directory name whose docs are to be considered relevant..
            count=0
            doc_count=0 ## count for maintaining the docs retrieved..
            rel=0 ## relevant docs count retrieved till that iteration..
            relevant_doc_count=0 ## total relevant docs count in top-k
            prec=[] ## precision values for ith iteration
            recall=[]

            for doc in doc_names: 
            ## creating list of relevant and irrelevant docs only top-n docs(of given folder) should be considered as relevant
                if d_name in doc and count<rel_doc_count and "**" not in doc:
                    rel_docs.append(doc) ## collecting the list of relevant docs
                    count+=1
                else:
                    if "**" not in doc:
                        irrel_docs.append(doc) ## collecting the list of irrelevant docs
            
            rel_docs_centroid = compute_centroid(q_pp,rel_docs,doc_vec)##computing centroids of relevant doc vectors
            irrel_docs_centroid = compute_centroid(q_pp,irrel_docs,doc_vec)## computing centroids of irrelevant doc vectors

            for i1 in range(len(rel_docs_centroid)):
                rel_docs_centroid[i1] = 0.7*rel_docs_centroid[i1]

            for i1 in range(len(irrel_docs_centroid)):
                irrel_docs_centroid[i1] = 0.25*irrel_docs_centroid[i1]

            new_query_vec = list((np.array(new_query_vec)+np.array(rel_docs_centroid)-np.array(irrel_docs_centroid)))## updating query vector
            
            print("="*125)
            print("\n\t\t\tDOCS RETRIEVED BASED ON COSINE-SIMILARITY SCORE WITH EMBEDDING OF FEEDBACK(ITERATION-{})\n".format(ite+1))
            print("="*125)
            
            doc_names = compute_cossim(new_query_vec,doc_vec,k,ite,rel_docs)
            
            
            for doc in doc_names:
                if d_name in doc:
                    relevant_doc_count+=1 ## count of relevant docs in top-k
            
            for doc in doc_names:
                doc_count+=1
                if d_name in doc:
                    rel+=1 ## increment the count of rel doc
                    prec.append(rel/doc_count) ## computing the precision values
                    recall.append(rel/relevant_doc_count) ## computing recall values
                    
            plt.title("PRECISION-RECALL CURVE WITH EMBEDDING OF FEEDBACK AFTER ITERATION-{}".format(ite+1))
            plt.xlabel("RECALL")
            plt.ylabel("PRECISION")
            plt.plot(recall,prec)
            plt.show()
            plot_tsne(q_vec,new_query_vec,ite+1)
            print("AVERAGE PRECISION WITH EMBEDDING OF FEEDBACK FOR QUERY '{}' AFTER ITERATION-{} IS \n{}".format(q,ite+1,sum(prec)/len(prec)))
            print("="*125)
            avg_prec.append(sum(prec)/len(prec))##appending all avg precisions in one list
    
    else:
        k2=0
        continue
        
MAP(avg_prec,k2,query_count)


# ## EXAMPLES

# Query: Pretty good opinions on biochemistry machines
# 
# Ground Truth :Documents inside folder sci.med
# 
# Query: Scientific tools for preserving rights and body
# 
# Ground Truth :Documents inside folder talk.politics.misc
# 
# Query: Frequently asked questions on State-of-the-art visualisation tools
# 
# Ground Truth :Documents inside folder sci.med
