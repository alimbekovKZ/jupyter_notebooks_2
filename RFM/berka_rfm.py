import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.figure_factory as ff
import seaborn as sns
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.title('Berka RFM')

DATA_URL = ('data/trans.asc')

start_date = dt.datetime(1993,1,1)
end_date = dt.datetime(2000,1,1)

# functions that process the format of the birth_number.

# returns the middle two digits of a six digit integer.
def get_mid2_dig(x):
    return int(x/100) % 100

# returns the month of birth_number.
def get_month(x):
    mth = get_mid2_dig(x)
    if mth > 50:
        return mth - 50
    else:
        return mth

# returns the month of birth_number.
def get_day(x):
    return x % 100

# returns the year of birth_number.
def get_year(x):
    return int(x/10000)

# returns the gender by examining birth_number.
def get_gender(x):
    mth = get_mid2_dig(x)
    if mth > 50:
        return 'F'
    else:
        return 'M'

def convert_date_to_days(x):
    td = x - start_date
    return td.days

def convert_int_to_date(x):
    yr = get_year(x) + 1900
    mth = get_month(x)
    day = get_day(x)
    return dt.datetime(yr, mth, day)

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL, sep=';', low_memory=False)
    NOW = dt.datetime(2011,12,10)
    data['NOW'] = NOW
    data['date'] = data['date'].map(convert_int_to_date)
    data['trans_date'] = data['date'].map(convert_date_to_days)
    del data['date']
    data['new_trans_data'] = (pd.to_datetime(data['NOW'])) + data['trans_date'].map(dt.timedelta)
    return data

def RScore(x,p,d):
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.40]:
        return 2
    elif x <= d[p][0.60]: 
        return 3
    elif x<=d[p][0.80]:
        return 4
    else:
        return 5
    
def FMScore(x,p,d):
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.40]:
        return 4
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]:
        return 2
    else:
        return 1

@st.cache
def buildRFM(data, now_d):
    #rfmTable = data.groupby('account_id').agg({'new_trans_data': lambda x: (now_d - x.max()).days, 'trans_id': lambda x: len(x), 'trans_amount': lambda x: x.sum()})
    rfmTable = data.groupby('account_id').agg(new_trans_data=('new_trans_data', lambda x: (now_d - x.max()).days), trans_id=('trans_id',lambda x: len(x)) ,trans_amount=('amount','sum'))
    rfmTable['new_trans_data'] = rfmTable['new_trans_data'].astype(int)
    rfmTable.rename(columns={'new_trans_data': 'recency', 'trans_id': 'frequency', 'trans_amount': 'monetary_value'}, inplace=True)
    quantiles = rfmTable.quantile(q=[0.20, 0.40, 0.60, 0.80])
    quantiles = quantiles.to_dict()
    segmented_rfm = rfmTable
    segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
    segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
    segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
    segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
    return segmented_rfm

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

NOW = dt.datetime(2011,12,10)

segmented_rfm = buildRFM(data, NOW)
st.subheader('RFM Score')
st.write(segmented_rfm.head(5))

st.subheader('Monetary value')
st.bar_chart(segmented_rfm.groupby('RFMScore').agg('monetary_value').mean())

st.subheader('Frequency')
st.bar_chart(segmented_rfm.groupby('RFMScore').agg('frequency').mean())

st.subheader('Recency')
st.bar_chart(segmented_rfm.groupby('RFMScore').agg('recency').mean())

df = pd.read_excel('Online Retail.xlsx', dtype={'CustomerID': str, 'InvoiceID': str}, parse_dates=['InvoiceDate'], infer_datetime_format=True)
df.dropna(subset=['CustomerID'], inplace=True)
n_orders = df.groupby(['CustomerID'])['InvoiceNo'].nunique()
mult_orders_perc = np.sum(n_orders > 1) / df['CustomerID'].nunique()
df = df[['CustomerID', 'InvoiceNo', 'InvoiceDate']].drop_duplicates()
df['order_month'] = df['InvoiceDate'].dt.to_period('M')
df['cohort'] = df.groupby('CustomerID')['InvoiceDate'] \
                 .transform('min') \
                 .dt.to_period('M') 

df_cohort = df.groupby(['cohort', 'order_month']) \
              .agg(n_customers=('CustomerID', 'nunique')) \
              .reset_index(drop=False)
df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(attrgetter('n'))
cohort_pivot = df_cohort.pivot_table(index = 'cohort',
                                     columns = 'period_number',
                                     values = 'n_customers')

cohort_size = cohort_pivot.iloc[:,0]
retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)

ax = sns.distplot(n_orders, kde=False, hist=True)
ax.set(title='Distribution of number of orders per customer',
       xlabel='# of orders', 
       ylabel='# of customers');
st.pyplot()

with sns.axes_style("white"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
    sns.heatmap(retention_matrix, 
                mask=retention_matrix.isnull(), 
                annot=True, 
                fmt='.0%', 
                cmap='RdYlGn', 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap=white_cmap, 
                ax=ax[0])

    fig.tight_layout()
    st.pyplot()	

