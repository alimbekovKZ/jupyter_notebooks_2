
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os

from sklearn.preprocessing import LabelEncoder,LabelBinarizer


import warnings
warnings.filterwarnings('ignore')
import scipy.sparse
from scipy import linalg
from scipy.special import iv
import scipy.sparse as sp

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD

import argparse
import time


# In[2]:


NROWS = None
# NROWS = 50000

PATH = '../input'
train_identity = pd.read_csv(os.path.join(PATH,'train_identity.csv'), nrows=NROWS)
train_transaction = pd.read_csv(os.path.join(PATH, 'train_transaction.csv'), nrows=NROWS)
test_identity = pd.read_csv(os.path.join(PATH,'test_identity.csv'), nrows=NROWS)
test_transaction = pd.read_csv(os.path.join(PATH, 'test_transaction.csv'), nrows=NROWS)


# In[3]:


test_transaction['isFraud'] = -1


# In[4]:


# train_identity.shape,train_transaction.shape,test_identity.shape,test_transaction.shape


# In[5]:


identity = pd.concat([train_identity,test_identity],axis=0,ignore_index=True)
transaction = pd.concat([train_transaction,test_transaction],axis=0,ignore_index=True)


# In[6]:


# identity.shape,transaction.shape


# In[7]:


data = pd.merge(transaction,identity,on='TransactionID',how='left')


# In[8]:


excluded_features = ['TransactionDT','TransactionID','isFraud']
categorical_features = ['P_emaildomain','ProductCD', 'R_emaildomain',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                       'addr1', 'addr2', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                        'id_12', 'id_13','id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',
                       'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
                       'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34',
                       'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
numerical_features = [item for item in data.columns if item not in excluded_features + categorical_features]


# In[9]:


def transform(df,numerical_max_bins=256):
    for col in df.columns:
        if col in excluded_features:
            print("drop columns: ",col)
            df = df.drop(col, axis=1)
        if col in categorical_features:
            if df[col].dtype==object:
                MISS = 'MISS'
            else:
                MISS = -1.0
            df[col] = df[col].fillna(MISS)
            lb = LabelEncoder()
            df[col] = lb.fit_transform(df[col].values)
        if col in numerical_features:
            df[col] = df[col].fillna(-1.0)
            df[col] = pd.qcut(df[col].values,q=numerical_max_bins,labels=False,duplicates='drop')
            if df[col].nunique() == 1:
                print("drop columns: ",col)
                df = df.drop(col, axis=1)
            else:
                lb = LabelEncoder()
                df[col] = lb.fit_transform(df[col].values)
    return df  


# In[10]:


def df2text(df,corpus_file = '../features/corpus.txt'):
    fout = open(corpus_file,'w')
    columns = df.columns
    values = df.values
    for i in range(values.shape[0]):
        tmp = values[i,:]
        text = ' '.join([col+'_'+str(val) for col, val in zip(columns,tmp)])
        
        fout.write(text+'\n')
    fout.close()


# In[11]:


group_features = [['P_emaildomain','R_emaildomain'],
      ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
      ['addr1', 'addr2'],
      ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']]


# In[12]:


def group_transform(data,group_features):
    for group in group_features:
        col_name = '_'.join(group)
        data[col_name] = 'g'
        for item in group:
            data[col_name] += '_' + data[item].astype(str)
    return data


# In[13]:


data = transform(data,numerical_max_bins=256)
data = group_transform(data,group_features)


# In[14]:


data.to_csv('../features/transform.corpus.csv',header=True,index=False)
df2text(data,corpus_file='../features/corpus.txt')


# In[16]:


import fasttext


# In[17]:


w2v = fasttext.train_unsupervised(input='../features/corpus.txt',
                                  model='cbow',
                                  dim=50,
                                  ws=data.shape[1],
                                  thread=48,
                                  minn=0,
                                  maxn=0,
                                  minCount=1)


# In[ ]:


w2v.save_model('../features/corpus.fasttext.cbow.model')


# In[ ]:


import fasttext
w2v = fasttext.load_model('../features/corpus.fasttext.cbow.model')
corpus_df = pd.read_csv('../features/transform.corpus.csv')
train_raw =pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_raw = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')


# In[ ]:


def get_sentence_embeddings(text,sep=' ',dim=100,fasttext_embeddings=None):
    v = np.zeros(dim)
    words = text.strip().split(sep)
    cnt = 0
    for word in words:
        if word in fasttext_embeddings:
            v += fasttext_embeddings[word]
            cnt +=1
    return v/cnt if cnt!=0 else v


# In[ ]:


def get_word_vector(x,prefix='card1_'):
    try:
        x = prefix + str(int(x))
    except:
        x = prefix + 'MISS'
    return w2v.get_word_vector(x)

def get_vector_feature(column='card1',dim=100):
    MISS = column + '_MISS'
    value = corpus_df[column].map(lambda x: w2v.get_word_vector(column+'_'+str(x)))
    return pd.DataFrame(data=np.array([item.tolist() for item in value]), 
                        columns=[column + '_' + str(i) for i in range(dim)])

def get_embed_feature(column='card1',dim=50):
    
    embed_card1 = get_vector_feature(column=column,dim=dim)
    train_embed_card1, test_embed_card1 = embed_card1[:590540],embed_card1[590540:]
    train_embed_card1.index = train_raw.index
    test_embed_card1.index = test_raw.index    
    
    return train_embed_card1,test_embed_card1


# In[ ]:


# train_embed_email,test_embed_email = get_embed_feature(column='P_emaildomain_R_emaildomain')
# train_embed_M,test_embed_M = get_embed_feature(column='M1_M2_M3_M4_M5_M6_M7_M8_M9')
# train_embed_addr,test_embed_addr = get_embed_feature(column='addr1_addr2')
train_embed_card,test_embed_card = get_embed_feature(column='card1_card2_card3_card4_card5_card6')


# In[ ]:


train_embed_card.to_csv('../features/train.embed_card.csv',index=True,header=True)
test_embed_card.to_csv('../features/test.embed_card.csv',index=True,header=True)

print(train_embed_card.head())