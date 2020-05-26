# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# exit(0)

import os
import sys
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
print('torch version:', torch.__version__)


# setup external file ############################################################
TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]


#local
if 1:
    DATA_DIR = '/root/share/project/kaggle/2020/grapheme_classification/data'
    SUBMISSION_CSV_FILE = '/root/share/project/kaggle/2020/grapheme_classification/result/xxx-submission.csv'


    MYFILE_DIR = './myfile'
    DEBUG_DIR  = '/root/share/project/kaggle/2020/grapheme_classification/data'
    DESENET_CHECKPOINT_FILE = [
        '/root/share/project/kaggle/2020/grapheme_classification/result/run1/dense121-fold0/checkpoint/swa_fold0_no_bn_model.pth',
        '/root/share/project/kaggle/2020/grapheme_classification/result/run1/dense121-fold1/checkpoint/swa_fold1_no_bn_model.pth',
    ]


#kaggle
if 0:
    DATA_DIR = '/kaggle/input/bengaliai-cv19'
    SUBMISSION_CSV_FILE = 'submission.csv'

    MYFILE_DIR = '/kaggle/input/bengali-submit-0/myfile'
    DEBUG_DIR  = '/kaggle/input/bengali-submit-0'
    CHECKPOINT_FILE  = [
        '/kaggle/input/bengali-submit-0/densenet_checkpoint/swa_fold0_no_bn_model.pth',
        '/kaggle/input/bengali-submit-0/densenet_checkpoint/swa_fold1_no_bn_model.pth',
    ]



# from shutil import copyfile
# copyfile(src = '/kaggle/input/bengali-submit-0/myfile/resnet.py', dst = '/resnet.py')
sys.path.append(MYFILE_DIR)
#print(sys.path)
from etc import *
from densenet_model import Net as DenseNet

import warnings
warnings.filterwarnings('ignore')


#### net #########################################################################

def do_predict(net, input):

    def logit_to_probability(logit):
        probability=[]
        for l in logit:
            p = F.softmax(l,1)
            probability.append(p)
        return probability

    #-----
    num_ensemble = len(net)
    for i in range(num_ensemble):
        net[i].eval()


    probability=[0,0,0]
    #----
    for i in range(num_ensemble):
        logit = net[i](input)
        prob  = logit_to_probability(logit)
        probability = [p+q for p,q in zip(probability,prob)]

    #----
    probability = [p/num_ensemble for p in probability]
    predict = [torch.argmax(p,-1) for p in probability]
    predict = [p.data.cpu().numpy() for p in predict]
    predict = np.array(predict).T
    predict = predict.reshape(-1)

    return predict



## load net -----------------------------------
net = []
for checkpoint_file in DESENET_CHECKPOINT_FILE:
    n = DenseNet().cuda()
    n.load_state_dict(torch.load(checkpoint_file, map_location=lambda storage, loc: storage),strict=True)
    net.append(n)

#----------------------------------------------




def run_check_setup():
    row_id=[]
    target=[]
    batch_size = 16
    print('')
    print('start here !!!!')

    if 1:
        df = pd.read_csv(DEBUG_DIR+'/debug_image_data_0.csv')
        image_id = df['image_id'].values
        label    = df[TASK_NAME].values
        grapheme = df['grapheme'].values


        start_timer = timer()
        df = pd.read_parquet(DEBUG_DIR+'/debug_image_data_0.parquet', engine='pyarrow')
        print('pd.read_parquet() = %s'%(time_to_str((timer() - start_timer),'sec')))

        num_test = len(df)
        for b in range(0,num_test,batch_size):
            print('run_check_setup @',b)
            B = min(num_test,b+batch_size)-b

            image = df.iloc[b:b+B, range(1,32332+1)].values
            image_id = df.iloc[b:b+B, 0].values

            image = image.reshape(B,1,137, 236)
            image = np.tile(image, (1,3,1,1))
            image = image.astype(np.float32)/255

            #----
            input = torch.from_numpy(image).float().cuda()
            predict = do_predict(net, input)
            #----

            image_id = np.tile(image_id.reshape(B,1), (1,3,)) + ['_']  + TASK_NAME
            image_id = image_id.reshape(-1)

            row_id.append(image_id)
            target.append(predict)

    row_id = np.concatenate(row_id)
    target = np.concatenate(target)
    label  = label.reshape(-1)

    #---------
    correct = np.mean(target==label)
    print('correct = ',correct)
    print('')

    df = pd.DataFrame(zip(row_id, target, label), columns=['row_id', 'target', 'label'])
    df.to_csv(SUBMISSION_CSV_FILE, index=False)

    print(df[:25])
    exit(0)



'''
correct =  0.98

                         row_id  target  label
0         Train_0_grapheme_root      15     15
1       Train_0_vowel_diacritic       9      9
2   Train_0_consonant_diacritic       5      5
3         Train_1_grapheme_root     159    159
4       Train_1_vowel_diacritic       0      0
5   Train_1_consonant_diacritic       0      0
6         Train_2_grapheme_root      22     22
7       Train_2_vowel_diacritic       3      3
8   Train_2_consonant_diacritic       5      5
9         Train_3_grapheme_root      53     53
10      Train_3_vowel_diacritic       2      2
11  Train_3_consonant_diacritic       2      2
12        Train_4_grapheme_root      71     71
13      Train_4_vowel_diacritic       9      9
14  Train_4_consonant_diacritic       5      5
15        Train_5_grapheme_root     153    153
16      Train_5_vowel_diacritic       9      9
17  Train_5_consonant_diacritic       0      0
18        Train_6_grapheme_root      52     52
19      Train_6_vowel_diacritic       2      2
20  Train_6_consonant_diacritic       0      0
21        Train_7_grapheme_root     139    139
22      Train_7_vowel_diacritic       3      3
23  Train_7_consonant_diacritic       0      0
24        Train_8_grapheme_root      67     67

'''


########################################################################

def run_make_submission_csv():

    row_id=[]
    target=[]
    batch_size= 32

    print('\nstart here !!!!')
    for i in range(4):
        start_timer = timer()
        df  = pd.read_parquet(DATA_DIR+'/test_image_data_%d.parquet'%i, engine='pyarrow')
        #df  = pd.read_parquet(DATA_DIR+'/train_image_data_%d.parquet'%i, engine='pyarrow') #use this to test timing
        print('pd.read_parquet() = %s'%(time_to_str((timer() - start_timer),'sec')))

        start_timer = timer()
        num_test = len(df)
        for b in range(0,num_test,batch_size):
            if b%1000==0:
                print('test_image_data_%d.parquet @%06d, %s'%(i,b,time_to_str((timer() - start_timer),'sec')))
            #----
            B = min(num_test,b+batch_size)-b
            image = df.iloc[b:b+B, range(1,32332+1)].values
            image_id = df.iloc[b:b+B, 0].values

            image = image.reshape(B,1,137, 236)
            image = np.tile(image, (1,3,1,1))
            image = image.astype(np.float32)/255

            #----
            input = torch.from_numpy(image).float().cuda()
            predict = do_predict(net, input)
            #----

            image_id = np.tile(image_id.reshape(B,1), (1,3,)) + ['_']  + TASK_NAME
            image_id = image_id.reshape(-1)
            row_id.append(image_id)
            target.append(predict)
        print('')

    row_id = np.concatenate(row_id)
    target = np.concatenate(target)
    #---------

    df = pd.DataFrame(zip(row_id, target), columns=['row_id', 'target'])
    df.to_csv(SUBMISSION_CSV_FILE, index=False)



# main #################################################################
if __name__ == '__main__':
    run_check_setup()
    #run_make_submission_csv()

    print('\nsucess!')

