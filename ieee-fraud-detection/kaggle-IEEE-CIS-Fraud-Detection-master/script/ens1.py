
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 500)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, space_eval

from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
from time import time
from tqdm import tqdm_notebook

from xgboost import XGBClassifier
import os

import gc
import warnings
warnings.filterwarnings('ignore')


# # 导入数据

# In[2]:


NROWS = None
# NROWS = 3000


# In[3]:


train_identity = pd.read_csv('../input/train_identity.csv', nrows=NROWS)
train_transaction = pd.read_csv('../input/train_transaction.csv', nrows=NROWS)
train = train_transaction.merge(train_identity, how='left', on='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', nrows=NROWS)
test_transaction = pd.read_csv('../input/test_transaction.csv', nrows=NROWS)
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

sub = pd.read_csv('../input/sample_submission.csv', nrows=NROWS)

gc.enable()
del train_identity, train_transaction
del test_identity, test_transaction
gc.collect()

print("train.shape:", train.shape)
print("test.shape:", test.shape)
train.head(3)


# ## 内存优化

# In[4]:


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[: 3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min  and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum()/ 1024**2
    print('Memory usage after optimization is: {:.2f} MB, {:.1f}% reduction'.          format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[5]:


train = reduce_mem_usage(train)
test  = reduce_mem_usage(test)

# In[6]:


# train.to_csv('../temp/train_temp' + str(NROWS) + '.csv', index=False, header=True)
# test.to_csv('../temp/test_temp'+ str(NROWS) + '.csv', index=False, header=True)


# # 特征工程

# In[7]:


# train = pd.read_csv('../temp/train_temp' + str(NROWS) + '.csv')
# test = pd.read_csv('../temp/test_temp'+ str(NROWS) + '.csv')


# In[8]:


# train.head(3)


# In[9]:


# train.columns


# ## 特征分类: 按特征名分类

# In[10]:


train['null'] = train.isna().sum(axis=1)
test['null'] = test.isna().sum(axis=1)


# ### 时间相关特征(TransactionDT)

# In[11]:


def transform_TransactionDT(df):
    START_DATE = '2017-12-01'
    start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['Date'] = df['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))

    df['Weekday'] = df['Date'].dt.dayofweek
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Morning'] = (df['Hour'] >= 7) & (df['Hour'] <= 11).astype('int')
    df['Noon'] = (df['Hour'] >=12) & (df['Hour'] <= 18).astype('int')
    df['Evening'] = (df['Hour'] >= 19) & (df['Hour'] <=23).astype('int')
    df['Midnight'] = (df['Hour'] >= 0) & (df['Hour'] <=6).astype('int')
    
    del df['Date']
    
    return df


# In[12]:


train = transform_TransactionDT(train)
test = transform_TransactionDT(test)


# ### 金额(TransactionAmt)

# In[13]:


train['TransactionAmt'] = train['TransactionAmt'].astype(float)
train['TransAmtLog'] = np.log(train['TransactionAmt'])
train['TransAmtDemical'] = train['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()

test['TransactionAmt'] = test['TransactionAmt'].astype(float)
test['TransAmtLog'] = np.log(test['TransactionAmt'])
test['TransAmtDemical'] = test['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()


# ### ProductCD

# In[14]:


train.ProductCD.unique()


# ### card特征提取

# In[15]:


# # 对card特征 Label Encoding
# card = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

# for f in tqdm_notebook(card):
#     lbl = LabelEncoder()
#     temp = pd.DataFrame(train[f].astype(str).append(test[f].astype(str)))
#     lbl.fit(temp[f])
#     train[f + '_endoce'] = lbl.transform(list(train[f].astype(str)))
#     test[f + '_endoce'] = lbl.transform(list(test[f].astype(str))) 


# In[16]:


# card1是类别型特征,对card1的字段进行提取

train["card1_len"] = train["card1"].apply(lambda x: len(str(x)))
test["card1_len"]  = test["card1"].apply(lambda x: len(str(x)))

train["card1_first"] = train["card1"].apply(lambda x: str(x)[0])
test["card1_first"]  = test["card1"].apply(lambda x: str(x)[0])

# card2特征提取
# train["card2_first"] = train["card2"].apply(lambda x: str(x)[0])
# test["card2_first"]  = test["card2"].apply(lambda x: str(x)[0])

# train["card2_second"] = train["card2"].apply(lambda x: str(x)[1])
# test["card2_second"]  = test["card2"].apply(lambda x: str(x)[1])

# train["card2_last"] = train["card2"].apply(lambda x: str(x)[2])
# test["card2_last"]  = test["card2"].apply(lambda x: str(x)[2])


# In[17]:


# 是否缺失的标记
train["card1_na"] = 0
train.loc[train["card1"].isna(), "card1_na"] = 1
test["card1_na"] = 0
test.loc[test["card1"].isna(), "card1_na"] = 1

train["card2_na"] = 0
train.loc[train["card2"].isna(), "card2_na"] = 1
test["card2_na"] = 0
test.loc[test["card2"].isna(), "card2_na"] = 1

# train["card3_na"] = 0
# train.loc[train["card3"].isna(), "card3_na"] = 1
# test["card3_na"] = 0
# test.loc[test["card3"].isna(), "card3_na"] = 1

# train["card4_na"] = 0
# train.loc[train["card4"].isna(), "card4_na"] = 1
# test["card4_na"] = 0
# test.loc[test["card4"].isna(), "card4_na"] = 1

train["card5_na"] = 0
train.loc[train["card5"].isna(), "card5_na"] = 1
test["card5_na"] = 0
test.loc[test["card5"].isna(), "card5_na"] = 1

# train["card6_na"] = 0
# train.loc[train["card6"].isna(), "card6_na"] = 1
# test["card6_na"] = 0
# test.loc[test["card6"].isna(), "card6_na"] = 1


# In[18]:


# card字段拼接的统计

train['card_str'] = train["card1"].apply(lambda x: str(x)) + "_" + train["card2"].apply(lambda x: str(x)) +                "_" + train["card3"].apply(lambda x: str(x)) + "_" + train["card4"].apply(lambda x: str(x)) +                "_" + train["card5"].apply(lambda x: str(x)) + "_" + train["card6"].apply(lambda x: str(x))

test['card_str'] = test["card1"].apply(lambda x: str(x)) + "_" + test["card2"].apply(lambda x: str(x)) +                "_" + test["card3"].apply(lambda x: str(x)) + "_" + test["card4"].apply(lambda x: str(x)) +                "_" + test["card5"].apply(lambda x: str(x)) + "_" + test["card6"].apply(lambda x: str(x))

train['card_count_full'] = train['card_str'].map(pd.concat([train['card_str'], test['card_str']], ignore_index=True).value_counts(dropna=False))
test['card_count_full'] = test['card_str'].map(pd.concat([test['card_str'], test['card_str']], ignore_index=True).value_counts(dropna=False))


# In[19]:


train['TransactionAmt_to_std_card_str'] = train['TransactionAmt'] / train.groupby(['card_str'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card_str'] = test['TransactionAmt'] / test.groupby(['card_str'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_mean_card_str'] = train['TransactionAmt'] / train.groupby(['card_str'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card_str'] = test['TransactionAmt'] / test.groupby(['card_str'])['TransactionAmt'].transform('mean')


# In[20]:


train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

train['card2_count_full'] = train['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))
test['card2_count_full'] = test['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))
test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))

train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))
test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))

train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))
test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))

train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))
test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))


# In[21]:


train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_mean_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_mean_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card6'] = train['TransactionAmt'] / train.groupby(['card6'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card6'] = train['TransactionAmt'] / train.groupby(['card6'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card6'] = test['TransactionAmt'] / test.groupby(['card6'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card6'] = test['TransactionAmt'] / test.groupby(['card6'])['TransactionAmt'].transform('std')


# In[22]:


train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')


# In[23]:


train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')


# ### card svd特征

# In[24]:


from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

def get_dc_feature(df_train, df_test, n_comp=12, used_features=None):
    """
    构造分解特征
    """
    if not used_features:
        used_features = df_test.columns
        
    train = df_train.copy()
    test  = df_test.copy()    
        
    # tSVD
#     tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
#     tsvd_results_train = tsvd.fit_transform(train[used_features])
#     tsvd_results_test = tsvd.transform(test[used_features])

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train[used_features])
    pca2_results_test = pca.transform(test[used_features])

#     # ICAz
#     ica = FastICA(n_components=n_comp, random_state=420)
#     ica2_results_train = ica.fit_transform(train[used_features])
#     ica2_results_test = ica.transform(test[used_features])

#     # GRP
#     grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
#     grp_results_train = grp.fit_transform(train[used_features])
#     grp_results_test = grp.transform(test[used_features])

#     # SRP
#     srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
#     srp_results_train = srp.fit_transform(train[used_features])
#     srp_results_test = srp.transform(test[used_features])
    
    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

#         train['ica_' + str(i)] = ica2_results_train[:, i - 1]
#         test['ica_' + str(i)] = ica2_results_test[:, i - 1]

#         train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#         test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

#         train['grp_' + str(i)] = grp_results_train[:, i - 1]
#         test['grp_' + str(i)] = grp_results_test[:, i - 1]

#         train['srp_' + str(i)] = srp_results_train[:, i - 1]
#         test['srp_' + str(i)] = srp_results_test[:, i - 1]

    return train, test

print(train.shape, test.shape)
used_features = ['card1', 'card2','card3', 'card5']

train[used_features] = train[used_features].fillna(-1.0)
test[used_features]  = test[used_features].fillna(-1.0)

train, test = get_dc_feature(train, test, n_comp=3, used_features=used_features)
print(train.shape, test.shape)


# ### address
# both are for purchaser, addr1 as billing region, addr2 as billing country

# In[25]:


train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))
test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')


# ### distance

# In[26]:


train["dist1_plus_dist2"]  = train["dist1"] + train["dist2"]
train["dist1_minus_dist2"] = train["dist1"] - train["dist2"]
train["dist1_times_dist2"]  = train["dist1"] * train["dist2"]
train["dist1_divides_dist2"] = train["dist1"] / train["dist2"]

test["dist1_plus_dist2"]  = test["dist1"] + test["dist2"]
test["dist1_minus_dist2"] = test["dist1"] - test["dist2"]
test["dist1_times_dist2"]  = test["dist1"] * test["dist2"]
test["dist1_divides_dist2"] = test["dist1"] / test["dist2"]


# ### 邮箱

# In[27]:


def transform_email(df):
    for col in ['P_emaildomain', 'R_emaildomain']:
        col1 = col.replace('domain', '_suffix')
        df[col1] = df[col].str.rsplit('.', expand=True).iloc[:, -1]
        
        col2 = col.replace('domain', 'Corp')
        df[col2] = df[col]
        df.loc[df[col].isin(['gmail.com', 'gmail']), col2] = 'Google'
        df.loc[df[col].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk', 'yahoo.co.jp', 
                             'yahoo.de', 'yahoo.fr', 'yahoo.es', 'yahoo.com.mx', 
                             'ymail.com']), col2] = 'Yahoo'
        df.loc[df[col].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es', 
                             'hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr', 
                             'hotmail.fr']), col2] = 'Microsoft'
        df.loc[df[col].isin(['aol.com', 'verizon.net']), col2] = 'Verizon'
        df.loc[df[col].isin(['att.net', 'sbcglobal.net', 'bellsouth.net']), col2] = 'AT&T'
        df.loc[df[col].isin(['icloud.com', 'mac.com', 'me.com']), col2] = 'Apple'
        df.loc[df[col2].isin(df[col2].value_counts()[df[col2].value_counts() <= 1000].index), col2] = 'Others'
    
    return df


# In[28]:


train = transform_email(train)
test = transform_email(test)


# In[29]:


# train['P_email']=(train['P_emaildomain']=='xmail.com')
# train['R_email']=(train['R_emaildomain']=='xmail.com')
# test['P_email']=(test['P_emaildomain']=='xmail.com')
# test['R_email']=(test['R_emaildomain']=='xmail.com')


# ### C1-C14
# 
# C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
# 

# ### D1-D15
# timedelta, such as days between previous transaction, etc.

# ### M1-M9
# match, such as names on card and address, etc.

# In[30]:


MFeatures = ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]
for feature in MFeatures:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


# ### Vxxx
# Vesta engineered rich features, including ranking, counting, and other entity relations.

# ### id相关特征

# In[31]:


# # lastest_browser

# a = np.zeros(train.shape[0])
# train["lastest_browser"] = a
# a = np.zeros(test.shape[0])
# test["lastest_browser"] = a

# def browser(df):
#     df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
#     return df

# train=browser(train)
# test=browser(test)


# In[32]:


def transform_id_cols(df):
#     df['id_01_cut'] = pd.cut(df['id_01'], bins=[-100, -30, -20, -10, -5, 0])
    
    df['id_05_d'] = df['id_05']
    df['id_05_d'].where(df[df['id_05_d'].notnull()]['id_05_d'] == 0, 1, inplace=True)
    
#     df['id_06_cut'] = pd.cut(df['id_06'], bins=[-100, -10, -5, 0])
    df['id_06_d'] = df['id_06']
    df['id_06_d'].where(df[df['id_06_d'].notnull()]['id_06_d'] == 0, 1, inplace=True)
    
    # Dealing with id_30
    df['id_30_count'] = df['id_30'].map(df['id_30'].value_counts(dropna=False))
    df['System'] = df['id_30'].astype('str').str.split('.', expand=True)[0].str.split('_', expand=True)[0]
    df['SystemCorp'] = df['System'].str.split(expand=True)[0]
    
    # Dealing with id_31
    df['LastestBrowser'] = df['id_31']
    df.loc[df['LastestBrowser'].isin(['samsung browser 7.0', 'opera 53.0', 'mobile safari 10.0', 'chrome 63.0 for android', 
                                       'google search application 49.0', 'firefox 60.0', 'edge 17.0', 'chrome 69.0', 
                                       'chrome 67.0 for android', 'chrome 64.0', 'chrome 63.0 for ios', 'chrome 65.0', 
                                       'chrome 64.0 for android', 'chrome 64.0 for ios', 'chrome 66.0', 
                                       'chrome 65.0 for android', 'chrome 65.0 for ios', 'chrome 66.0 for android', 
                                       'chrome 66.0 for ios']), 'LastestBrowser'] = 1
    df.loc[df['LastestBrowser'].str.len() > 1, 'LastestBrowser'] = 0
    
    df['id_31_count'] = df['id_31'].map(df['id_31'].value_counts(dropna=False))
    
    df['MSBrowser'] = df['id_31'].str.contains('edge|ie|microsoft', case=False) * 1
    df['AppleBrowser'] = df['id_31'].str.contains('safari', case=False) * 1
    df['GoogleBrowser'] = df['id_31'].str.contains('chrome', case=False) * 1
    
    df['BrowserType'] = df['id_31']
    df.loc[df['BrowserType'].str.contains('samsung', case=False, na=False), 'BrowserType'] = 'Samsung'
    df.loc[df['BrowserType'].str.contains('safari', case=False, na=False), 'BrowserType'] = 'Apple'
    df.loc[df['BrowserType'].str.contains('chrome|google', case=False, na=False), 'BrowserType'] = 'Google'
    df.loc[df['BrowserType'].str.contains('firefox', case=False, na=False), 'BrowserType'] = 'Mozilla'
    df.loc[df['BrowserType'].str.contains('edge|ie|microsoft', case=False, na=False, regex=True), 'BrowserType'] = 'Microsoft'
    df.loc[df['BrowserType'].isin(df['BrowserType'].value_counts()                                      [df['BrowserType'].value_counts()< 1000].index), ['BrowserType']] = 'other'
    
    # Dealing with id_33
    df['id_33_count'] = df['id_33'].map(df['id_33'].value_counts(dropna=False))
    df['DisplaySize'] = df['id_33'].str.split('x', expand=True)[0].astype('float')                            * df['id_33'].str.split('x', expand=True)[1].astype('float')
    df['DisplaySize'].replace(0, np.nan, inplace=True)
    df['DisplaySize'] = (df['DisplaySize'] / df['DisplaySize'].min()).round(0)
    
    # Try easy combining
    for feature in ['id_02__id_20', 'id_13__id_17', 'id_02__D8', 'D11__DeviceInfo', 
                    'DeviceInfo__P_emaildomain', 'card2__dist1', 'card1__card5', 
                    'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
        f1, f2 = feature.split('__')
        df[feature] = df[f1].astype(str) + '_' + df[f2].astype(str)
    for col in ['id_30', 'id_31', 'id_33', 'DeviceInfo']:
        df[col + '_DeviceTpye'] = train[col] + '_' + train['DeviceType']
           
    return df


# In[33]:


train = transform_id_cols(train)
test  = transform_id_cols(test)


# ### DeviceType

# In[34]:


train.DeviceType.unique()


# ### DeviceInfo

# In[35]:


def transform_DeviceInfo(df):
    df['DeviceCorp'] = df['DeviceInfo']
    df.loc[df['DeviceInfo'].str.contains('HUAWEI|HONOR', case=False, na=False, regex=True), 'DeviceCorp'] = 'HUAWEI'
    df.loc[df['DeviceInfo'].str.contains('OS', na=False, regex=False), 'DeviceCorp'] = 'APPLE'
    df.loc[df['DeviceInfo'].str.contains('Idea|TA', case=False, na=False), 'DeviceCorp'] = 'Lenovo'
    df.loc[df['DeviceInfo'].str.contains('Moto|XT|Edison', case=False, na=False), 'DeviceCorp'] = 'Moto'
    df.loc[df['DeviceInfo'].str.contains('MI|Mi|Redmi', na=False), 'DeviceCorp'] = 'Mi'
    df.loc[df['DeviceInfo'].str.contains('VS|LG|EGO', na=False), 'DeviceCorp'] = 'LG'
    df.loc[df['DeviceInfo'].str.contains('ONE TOUCH|ALCATEL', case=False, na=False, regex=False), 'DeviceCorp'] = 'ALCATEL'
    df.loc[df['DeviceInfo'].str.contains('ONE A', na=False, regex=False), 'DeviceCorp'] = 'ONEPLUS'
    df.loc[df['DeviceInfo'].str.contains('OPR6', na=False, regex=False), 'DeviceCorp'] = 'HTC'
    df.loc[df['DeviceInfo'].str.contains('Nexus|Pixel', case=False, na=False, regex=True), 'DeviceCorp'] = 'google'
    df.loc[df['DeviceInfo'].str.contains('STV', na=False, regex=False), 'DeviceCorp'] = 'blackberry'
    df.loc[df['DeviceInfo'].str.contains('ASUS', case=False, na=False, regex=False), 'DeviceCorp'] = 'ASUS'
    df.loc[df['DeviceInfo'].str.contains('BLADE', case=False, na=False, regex=False), 'DeviceCorp'] = 'ZTE'
    
    df['DeviceCorp'] = df['DeviceInfo'].astype('str').str.split(':', expand=True)[0].                                str.split('-', expand=True)[0].str.split(expand=True)[0]
    
    df.loc[df['DeviceInfo'].isin(['rv', 'SM', 'GT', 'SGH']), 'DeviceCorp'] = 'SAMSUNG'
    df.loc[df['DeviceInfo'].str.startswith('Z', na=False), 'DeviceCorp'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.startswith('KF', na=False), 'DeviceCorp'] = 'Amazon'
    
    for i in ['D', 'E', 'F', 'G']:
        df.loc[df['DeviceInfo'].str.startswith(i, na=False), 'DeviceCorp'] = 'SONY'

    minority = df['DeviceCorp'].value_counts()[df['DeviceCorp'].value_counts() < 100].index
    df.loc[df['DeviceCorp'].isin(minority), 'DeviceCorp'] = 'Other'
    df['DeviceCorp'] = df['DeviceCorp'].str.upper()
    
    return df


# In[36]:


train = transform_DeviceInfo(train)
test  = transform_DeviceInfo(test)


# ### 类别型变量labelEncoder

# In[37]:


target = "isFraud"
# Label Encoding
for f in tqdm_notebook([feature for feature in train.columns if feature != target]):
    if train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = LabelEncoder()
        temp = pd.DataFrame(train[f].astype(str).append(test[f].astype(str)))
        lbl.fit(temp[f])
        train[f] = lbl.transform(list(train[f].astype(str)))
        test[f] = lbl.transform(list(test[f].astype(str))) 


# ### 构造特征

# In[38]:


def transform_number(df):
    df['id_02_log'] = np.log10(df['id_02'])
    
    df['C5_d'] = df['C5']
    df['C5_d'].where(df['C5'] == 0, 1, inplace=True)
    
    df['D8_mul_D9'] = df['D8'] * df['D9']
    
    df['TransAmt_mul_dist1'] = df['TransactionAmt'] * df['dist1']
    df['TransAmt_per_TransDT'] = df['TransactionAmt'] * 24 * 60 * 60 / df['TransactionDT']
    
    return df


# In[39]:


train = transform_number(train)
test  = transform_number(test)


# ### 交叉特征

# The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) 
# and transactions posterior to it with either user account, email address or billing address 
# directly linked to these attributes as fraud too. 
# If none of above is reported and found beyond 120 days, then we define as legit transaction (isFraud=0).

# ### 原交叉特征

# In[40]:


for feature in tqdm_notebook(['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 
                              'P_emaildomain__C2', 
                              'P_emaildomain__card1', 'P_emaildomain__card2',
                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 
                              'addr1__card1', 'card2__card4', 'card4__card6'
                             ]):

    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))


# ### 增加后的交叉特征

# In[41]:


# for feature in tqdm_notebook(['P_emaildomain__card1', 'P_emaildomain__card2', 'P_emaildomain__card3', 'P_emaildomain__card4', 'P_emaildomain__card5', 'P_emaildomain__card6',
#                 'R_emaildomain__card1', 'R_emaildomain__card2', 'R_emaildomain__card3', 'R_emaildomain__card4', 'R_emaildomain__card5', 'R_emaildomain__card6',
#                 'addr1__card2', 'addr1__card3', 'addr1__card4', 'addr1__card5', 'addr1__card6',
#                 'addr2__card3', 'addr2__card4', 'addr2__card5', 'addr2__card6',
#                 'card3__card4', 'card3__card5', 'card3__card6',
#                 'card4__card5', 'card4__card6',
#                 'card5__card6',
#                 'id_02__id_20', 'id_02__D8', 
#                 'card2__dist1', 'card2__id_20',
#                 'P_emaildomain__DeviceInfo', 'P_emaildomain__C2',
#                 'D11__DeviceInfo']):

#     f1, f2 = feature.split('__')
#     train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
#     test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

#     le = LabelEncoder()
#     le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
#     train[feature] = le.transform(list(train[feature].astype(str).values))
#     test[feature] = le.transform(list(test[feature].astype(str).values))


# ### 高阶交叉特征

# In[42]:


for feature in tqdm_notebook([
                              'P_emaildomain__card1__card2', 'addr1__card1__card2'
                             ]):

    f1, f2, f3 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str) + '_' + train[f3].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str) + '_' + test[f3].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))


# # 缺失值填充

# # 线下验证

# ### 按照时间划分数据集

# In[43]:


train.head(3)


# In[44]:


test.head(3)


# In[45]:


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
test_X = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)


# In[46]:


# del train
# gc.collect()


# In[47]:


X.shape, test_X.shape


# In[48]:


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }


# ### 删除不重要的特征

# In[49]:


# feature_keep = 505

# all_data = lgb.Dataset(X, label=y)
# feature_clf  = lgb.train(params, all_data, num_boost_round = 800, valid_sets = [all_data], verbose_eval=100) 
# feature_importances = pd.DataFrame()
# feature_importances['feature'] = X.columns
# feature_importances['score'] = feature_clf.feature_importance()
# feature_importances = feature_importances.sort_values(by="score", ascending=False)
# feature_used = list(feature_importances.head(feature_keep)['feature'])

# X = X[feature_used]
# test_X = test_X[feature_used]


# In[50]:


# drop_features = ["id_31"]
# # drop_features = ["id_31", "D15", "D10"]
# X = X.drop(drop_features, axis=1)
# test_X = test_X.drop(drop_features, axis=1)


# In[51]:


# cols_to_drop=['V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
#               'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
#               'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
#               'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120']


# print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

# X = X.drop(cols_to_drop, axis=1)
# test_X = test_X.drop(cols_to_drop, axis=1)    


# In[52]:


folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
#     clf = lgb.train(params, trn_data, num_boost_round = 10000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=500)
    clf = lgb.train(params, trn_data, num_boost_round = 10000, valid_sets = [val_data], verbose_eval=100, early_stopping_rounds=500)
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_0']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


# In[53]:



feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(7, 7))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(30), x='average', y='feature');
plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[59]:


PREDICT = True
if PREDICT:
    # clf right now is the last model, trained with 80% of data and validated with 20%
    best_iter = clf.best_iteration
    print("best_iteration: ", best_iter)
    clf = lgb.LGBMClassifier(**params, num_boost_round=int(best_iter * 1.1))
    clf.fit(X, y)

    # all_data = lgb.Dataset(X, label=y)
    # all_clf  = lgb.train(params, all_data, num_boost_round = int(best_iter * 1.20), valid_sets = [all_data], verbose_eval=100)

    sub['isFraud'] = clf.predict_proba(test_X)[:, 1]
    # sub['isFraud'] = all_clf.predict(test_X)
    sub.to_csv('ieee_cis_fraud_detection_v6.csv', index=False)


# ### xbg模型

# In[62]:


if PREDICT:
    print("xgb model")
    sub = pd.read_csv('../input/sample_submission.csv', nrows=NROWS)

    n_fold = 6
    folds = KFold(n_splits=n_fold,shuffle=True)

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    xgb_sub=sub.copy()
    xgb_sub['isFraud'] = 0

    #fill in -111 for categoricals
    X = X.fillna(-111)
    test_X = test_X.fillna(-111)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        xgb = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,
                        learning_rate=0.05, max_delta_step=0, max_depth=9,
                        min_child_weight=1, missing=-111, n_estimators=500, n_jobs=1,
                        nthread=None, objective='binary:logistic', random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                        subsample=0.9, verbosity=1
        )

        X_train_, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train_, y_valid = y.iloc[train_index], y.iloc[valid_index]
        xgb.fit(X_train_,y_train_)
        del X_train_,y_train_
        pred=xgb.predict_proba(test_X)[:,1]
        val=xgb.predict_proba(X_valid)[:,1]
        del xgb, X_valid
        print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
        del val,y_valid
        xgb_sub['isFraud'] = xgb_sub['isFraud']+pred/n_fold
        del pred
        gc.collect()
        
    xgb_sub.to_csv('sub_xgb.csv', index=False)


# ### ensemble

# In[63]:


if PREDICT:
    sub1 = pd.read_csv('sub_xgb.csv')
    sub2 = pd.read_csv('ieee_cis_fraud_detection_v6.csv')

    final_sub=sub1.copy()
    final_sub['isFraud'] = final_sub['isFraud']*0.6 + sub2['isFraud']*0.4
    final_sub.to_csv('final_sub_v1.csv', index=False)


# # 结果记录

# - file/线下mean/线下fold5/线上
# - ieee_cis_fraud_detection_v2.csv/0.92431/0.93523/0.9416
# 

# # 实验

# - 不带交叉特征: 0.9021, 0.924753
# - 原交叉特征:  0.90323, 0.927644
# - 增加交叉特征: 0.902691, 0.924993
# - 优化交叉特征: 0.90404, 0.927612
# - 高阶交叉特征1 2个: 0.90372, 0.928267 最优
# - 高阶交叉特征2 10个: 0.902636, 0.925677
# - 高阶交叉特征2 5个: 0.903707, 0.926686

# - 删除不重要的特征
# - 保留全部:          0.90372,  0.928267
# - feature_keep=400: 0.90240,  0.927373
# - feature_keep=480: 0.90302,  0.926563
# - feature_keep=180: 0.90218,  0.926139
# - feature_keep=505: 0.904084, 0.926456
# 

# - 删除特征
# - https://www.kaggle.com/iasnobmatsu/xgb-model-with-feature-engineering
# 
# - 保留全部:            0.9037,  0.9283
# - 删除D15:            0.9038,   0.9254
# - 删除D15和D10:       0.9036,   0.9264    
# - 删除D15,D10,id_13:  0.9038,   0.9271

# - 增加svd等特征
# - 原始:                   0.9037,  0.9283
# - card +15个特征:         0.9016,  0.9241
# - card +3个pca特征:       0.9022, 0.9255
# 

# - 增加 TransactionAmt_to_mean_card 特征
# - 原始, card1和card4:                   0.9037, 0.9283
# - 增加card2, card3, card5, card6:       0.9032, 0.9274         
# 

# - card1 特征挖掘
# - 不增加:                                0.9020, 0.9261     
# - 增加card1_len和card1_first:            0.9037, 0.9293
# - 增加card1_len和card1_first, 去掉pca:    0.9034, 0.9274

# - card2 特征挖掘
# - 不增加:                                0.9031, 0.9276
# - 增加card2_first, second, last:         0.90204, 0.9242    

# - card每个值的缺失情况:
# - 未加特征之前:    0.9035, 0.9290
# - 加特征之后:      

# - P_email
# - 增加前: 0.9032, 0.9298
# - 增加后: 0.9033, 0.9275

# - lastest_browser
# - 增加前: 0.9032, 0.9298
# - 增加后: 0.9013, 0.9252

# - cols_to_drop
# - 删除前: 0.9032, 0.9298
# - 删除后: 0.9033, 0.9275

# - card字段拼接统计 (10万条样本)
# - 增加前: 0.9128, 0.9093
# - 增加后: 0.9129, 0.9107
# 
# - card字段拼接统计 (5万条样本)
# - 增加前: 0.9032, 0.9298
# - 增加后: 0.9030, 0.9284
# 
# - TransactionAmt_to_std_card_str和TransactionAmt_to_mean_card_str:
# - 增加前: 0.9030, 0.9284
# - 增加后: 0.9022, 0.9285
# 

# - 对card特征 Label Encoding
# - 增加前: 0.9022, 0.9285
# - 增加后: 0.9014, 0.9267

# nohup python -u ens1.py > ens1.log 2>&1 &