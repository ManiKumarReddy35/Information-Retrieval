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


# ## PROCEDURE FOR READING THE FILES AND BUILDING THE POSTING LISTS

# In[3]:


def readfiles(path):
    files = glob.glob(path) 
    plist={}
    univ_set = []
    for f in files:
        docs = glob.glob(f+"\*")
        for k in docs:
            with open(k) as newlinetest:
                indi = newlinetest.readlines()
            for newline in range(len(indi)):
                if(indi[newline]=="\n"):
                    newline+=1
                    break
            filedata=[]
            head, tail = os.path.split(k)
            univ_set.append(int(tail))
            try:
                with open(k) as fdata:
                    filedata.append(fdata.readlines()[newline:])
                    for l in range(len(filedata)):
                        inter_data = ' '.join(filedata[l])
                    preproc_data = preprocess(inter_data)
                    for j in word_tokenize(preproc_data):
                        if j not in plist:
                            head, tail = os.path.split(k)
                            int_tail = int(tail)
                            plist[j] = [int_tail]
                        else:
                            head, tail = os.path.split(k)
                            int_tail = int(tail)
                            if int_tail not in plist[j]:
                                plist[j].append(int_tail)
                            
            except:
                print("Exception Occured..")
    return plist,univ_set


# ## DATA PREPROCESSING PROCEDURE FOR BOOLEAN RETRIVAL

# In[4]:


def preprocess(no_metadata):
    stopword=""
    stopword = set(stopwords.words('english'))
    pattern1 = '[0-9!?$\[\]/\}#=<>"\*:,|_~;()^-]'
    pattern2 = '[\n\n]+'
    pattern3 = '[\ \ ]+'
    pattern4 = '[^0-9a-zA-Z]+'
    wout_sc = re.sub(pattern1,'',no_metadata) #removing special characters
    wout_el = re.sub(pattern2,'\n',wout_sc) # removing empty lines (which are greater than 2)
    wout_mspaces = re.sub(pattern3,' ',wout_el) # removing multiple spaces
    wout_numwords = re.sub(pattern4,' ',wout_mspaces)
    cleaned_text = wout_numwords.strip()
    cleaned_text = word_tokenize(cleaned_text)#cleaned_text.split(" ")
    cleantext = " "
    for i in cleaned_text:
        if i not in stopword and len(i)>2:
            i=i.lower()
            i=lemmatizer.lemmatize(i) 
            cleantext = cleantext+i+" "
    return cleantext


# # RUN THE BELOW CODE TO BUILD POSTING LISTS

# In[5]:


start=time.time()
plist,universal_set = readfiles("C:\\Users\\Sai Kumar\\Desktop\\New folder\\20_newsgroups\\*")
end=time.time()-start
print("TIME TAKEN FOR EXECUTION::",time.strftime("%H:%M:%S", time.gmtime(end)))
universal_set.sort()


# In[ ]:





# ## LOGICAL OPERATIONS DEFINTIONS

# In[12]:


def operation(l1,l2,op):
    l1.sort()
    l2.sort()
    if(op=="or"):
        comp=0
        int_res=[]
        i=0
        j=0
        while(i<len(l1) and j<len(l2)):
            if(l1[i]==l2[j]):
                comp=comp+1
                int_res.append(l1[i])
                i=i+1
                j=j+1
            elif(l1[i]<l2[j]):
                comp=comp+1
                int_res.append(l1[i])
                i=i+1
            else:
                int_res.append(l2[j])
                j=j+1
        if(i<len(l1)):
            for k in range(i,len(l1)):
                int_res.append(l1[k])
        else:
            for k1 in range(j,len(l2)):
                int_res.append(l2[k1])
        return int_res,comp
        
    if(op=="and"):
        i=0
        j=0
        comp=0
        int_res=[]
        while(i<len(l1) and j<len(l2)):
            if(l1[i]==l2[j]):
                comp=comp+1
                int_res.append(l1[i])
                i=i+1
                j=j+1
            elif(l1[i]<l2[j]):
                i=i+1
                comp=comp+1
            else:
                j=j+1            
        return int_res,comp

def operation_not(l1,univ):
    return list(set(univ)-set(l1)),len(l1)


# # QUESTION 1 CODE EXECUTION STARTS HERE

# In[19]:


comps=0
q = input("ENTER THE QUERY::")
q_split_np = word_tokenize(q)
q_split=[]
for i in q_split_np:
    if i not in {"AND","OR","NOT"}:
        q_split.append(preprocess(i).strip())
    else:
        q_split.append(i)

print("QUERY AFTER PREPROCESSING AND WORD TOKENIZING::",q_split)
for i in range(len(q_split)):
    if(q_split[i]!="AND" and q_split[i]!="OR" and q_split[i]!="NOT"):
        try:
            term_list = plist[q_split[i]]
            q_split.pop(i)
            q_split.insert(i,term_list)
        except:
            print("key "+q_split[i]+" not found..")

i=0
while (i<len(q_split)): #for i in range(len(q_split)):
    if(q_split[i]=="NOT"):
        universal = universal_set.copy()
        not_result,compn = operation_not(q_split[i+1],universal)
        q_split.insert(i,not_result)
        del q_split[i+1]
        del q_split[i+1]
        comps=comps+compn
        i=0
    else:
        i=i+1
    if(len(q_split)==1):
        break

i=0
while (i<len(q_split)):
    if(q_split[i]=="AND"):
        and_result,compa = operation(q_split[i-1],q_split[i+1],"and")
        q_split.insert(i,and_result)
        del q_split[i-1] #i-1
        del q_split[i] #i
        del q_split[i] #i
        comps=comps+compa
        i=0
    else:
        i = i+1
    if(len(q_split)==1):
        break

i=0
while (i<len(q_split)):
    if(q_split[i]=="OR"):
        or_result,compo = operation(q_split[i-1],q_split[i+1],"or")
        q_split.insert(i,or_result)
        del q_split[i-1] #i-1
        del q_split[i] #i
        del q_split[i] #i
        comps=comps+compo
        i=0
    else:
        i = i+1
    if(len(q_split)==1):
        break

print("NUMBER OF DOCS RETRIEVED::",len(q_split[0]))
print("NUMBER OF COMPARISIONS::",comps)
print("THE LIST OF DOCS RETRIVED ARE::\n",q_split[0])


# ## VERIFICATION WITH LOGICAL OPERATORS

# In[20]:


len(set(plist['boy']) | set(universal_set.copy())-set(plist['girl']))


# # QUESTION 2

# ## DATA PREPROCESSING PROCEDURE FOR POSITIONAL INDEXING

# In[77]:


## since it is phrase query we need exact matches therefore avoid removing some patters,stopwords,performing lemmatization 
def preprocess2(no_metadata):
    pattern2 = '[\n\n]+'
    pattern3 = '[\ \ ]+'
    wout_el = re.sub(pattern2,'\n',no_metadata) # removing empty lines (which are greater than 2)
    wout_mspaces = re.sub(pattern3,' ',wout_el) # removing multiple spaces
    cleaned_text = wout_mspaces.strip()
    cleaned_text = word_tokenize(cleaned_text)#cleaned_text.split(" ")
    cleantext = " "
    for i in cleaned_text:
            i=i.lower()
            cleantext = cleantext+i+" "
    return cleantext


# ## BUILDING THE DICTIONARY FOR POSITIONAL INDEXING (2nd PART OF ASSIGNMENT)

# In[78]:


def readfiles2(path):
    files = glob.glob(path) 
    posdict={}
    file_pos_dict={}
    for f in files:
        head_global, tail_global = os.path.split(f)
        if(tail_global in ["comp.graphics","rec.motorcycles"] ):
            docs = glob.glob(f+"\*")
            for k in docs:
                with open(k) as newlinetest:
                    indi = newlinetest.readlines()
                for newline in range(len(indi)):
                    if(indi[newline]=="\n"):
                        newline+=1
                        break
                head,tail = os.path.split(k)
                filedata = []
                pos=-1
                try:
                    with open(k) as fdata:
                        filedata.append(fdata.readlines()[newline:]) ## change code here
                except:
                    print("Exception Occured")
                no_meta = " "
                for i in range(len(filedata)):
                    data = ' '.join(filedata[i])
                    no_meta = no_meta+data
                no_meta_pp = preprocess2(no_meta)
                for j in word_tokenize(no_meta_pp):
                    pos=pos+1
                    if j not in posdict:
                        file_pos_dict={}
                        file_pos_dict[int(tail)]=[pos]
                        posdict[j]=[len(file_pos_dict),file_pos_dict]
                    else:
                        if int(tail) not in posdict[j][1].keys():
                            posdict[j][1][int(tail)] =[pos] 
                            posdict[j][0]+=1
                        else:
                            posdict[j][1][int(tail)].append(pos)
    return posdict


# In[79]:


start=time.time()
pos_dict = readfiles2("C:\\Users\\Sai Kumar\\Desktop\\New folder\\20_newsgroups\\*")
end=time.time()-start
print("TIME TAKEN FOR EXECUTION::",time.strftime("%H:%M:%S", time.gmtime(end)))


# # QUESTION 2 CODE EXECUTION STARTS HERE

# In[128]:


q2 = input("ENTER THE QUERY::")
q_split_np2 = word_tokenize(q2)
q_split2=[]
query2=[]
for i in q_split_np2:
    pp = preprocess2(i).strip()
    q_split2.append(pp)
    query2.append(pp) #this query2 is for evaluating positional indexes without AND since AND disturbs relative postion of words
    q_split2.append("AND") # in order to make use of AND function above
del q_split2[len(q_split2)-1] # deleting the last appended "and"
print("QUERY AFTER PREPROCESSING AND WORD TOKENIZING AND MODIFYING::",q_split2)
files_list=[]
for i in q_split2:
    if(i!="AND"):
        if i in pos_dict:
            files_list.append(list(pos_dict[i][1].keys()))
        else:
            print("Key "+i+" not found...")
    else:
        files_list.append("AND")
#print(files_list)
i=0
while (i<len(files_list)):
    if(files_list[i]=="AND"):
        and_result,compa = operation(files_list[i-1],files_list[i+1],"and")
        files_list.insert(i,and_result)
        del files_list[i-1] #i-1
        del files_list[i] #i
        del files_list[i] #i
        i=0
    else:
        i = i+1
    if(len(files_list)==1):
        break

## =================================== CODE SEGMENT FOR VERYFYING POSITIONAL INDEXING ========================================## 

n=len(query2)-1
final_ans=[]
for f in and_result:
    count=0
    for w1 in range(len(query2)):
        l1 = pos_dict[query2[w1]][1][f]
        for w2 in range(query2.index(query2[w1])+1,len(query2)):
            flag=0
            l2 = pos_dict[query2[w2]][1][f]
            for i in range(len(l1)):
                for j in range(len(l2)):          
                    if((l2[j]-l1[i]) == (w2-w1)):
                        count+=1
                        flag=1 # if one consectuive match is found we need not go further thats why we are breaking from loops..
                        break
                if(flag==1):
                    break                
    if(count==(n)*(n+1)/2):
        final_ans.append(f)
print("NUMBER OF FILES MATCHING THE PHRASE QUERY "+q2+" ARE::",len(final_ans))            
print("THE LIST OF FILES MATCHING THE PHRASE QUERY "+q2+" ARE::\n",final_ans)            


# ## REFERENCES

# 1. Documentation of glob module::https://www.poftut.com/python-glob-function-to-match-path-directory-file-names-with-examples/
# 2. Code snippets from my previous NLP assignments for basic tasks of reading and pre-processing data.
