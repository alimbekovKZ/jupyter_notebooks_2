import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 500)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')

NROWS = None
# NROWS = 50000

train_identity = pd.read_csv('../input/train_identity.csv', nrows=NROWS)
train_transaction = pd.read_csv('../input/train_transaction.csv', nrows=NROWS)
train = train_transaction.merge(train_identity, how='left', on='TransactionID')
# train.to_csv('../temp/train_temp' + str(NROWS) + '.csv', index=False, header=True)

test = train.loc[len(train)//2:]
train = train.loc[:len(train)//2]
test = test[1:]
test.index = range(len(test))

test1 = test[:int(test.shape[0] * 0.2)]
test2 = test[int(test.shape[0] * 0.2):]

train.to_csv('../temp/train_label.csv', index=False)
test1.to_csv('../temp/test1_label.csv', index=False)
test2.to_csv('../temp/test2_label.csv', index=False)
test.to_csv('../temp/test_label.csv', index=False)

test[["TransactionID", "isFraud"]].to_csv('../temp/sample_submission_label.csv', index=False)
