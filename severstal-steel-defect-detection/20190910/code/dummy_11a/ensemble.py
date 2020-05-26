from common import *



## combile classification and segmentation csv
if 1:
    label_csv ='/root/share/project/kaggle/2019/steel/delivery/20190910/result/resnet34-cls-full-foldb0-0/submit/resnet34-cls-tta-0.50.csv'
    mask_csv  ='/root/share/project/kaggle/2019/steel/delivery/20190910/result/resnet18-seg-full-softmax-foldb1-1-4balance/submit/resnet18-softmax-tta-0.50.csv'

    df_mask = pd.read_csv(mask_csv).fillna('')
    df_label = pd.read_csv(label_csv).fillna('')

    assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
    print((df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels'] != '').sum() ) #202
    df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels']=''


    csv_file = '/root/share/project/kaggle/2019/steel/delivery/20190910/result/merge-submission-11a.csv'
    df_mask.to_csv(csv_file, columns=['ImageId_ClassId', 'EncodedPixels'], index=False)

