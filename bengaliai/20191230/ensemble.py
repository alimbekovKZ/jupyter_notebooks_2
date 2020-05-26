import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from kaggle import *

# Stochastic Weight Averaging
# simply average the weights ... no bn post refinement
# https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/

def run_swa():

    # out_dir = \
    #     '/root/share/project/kaggle/2020/grapheme_classification/result/run1/resnet34-large-fold0'
    # snapshot = np.array([ 93000, 95000, 97000, 99000])
    # swa_name='swa_fold0_no_bn_model'

    out_dir = \
        'swa_result'
    snapshot = np.array([ 78000, 82000, 86000, 90000]) #cyclic learning rate
    swa_name='swa_fold1_no_bn_model'


    ## --- net ---
    num_snapshot = len(snapshot)

    swa_state_dict = torch.load(out_dir +'/checkpoint/%08d_model.pth'%(snapshot[0]), map_location=lambda storage, loc: storage)
    for k,v in swa_state_dict.items():
        swa_state_dict[k] = torch.zeros_like(v)

    for iter in snapshot:
        checkpoint = out_dir +'/checkpoint/%08d_model.pth'%(iter)
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        for k,v in state_dict.items():
            swa_state_dict[k] += v
        #---
        print(checkpoint)
    print('')

    #----
    for k,v in swa_state_dict.items():
        swa_state_dict[k] /= num_snapshot
    torch.save(swa_state_dict,out_dir +'/checkpoint/%s.pth'%swa_name)
    #torch.save(swa_state_dict,out_dir +'/checkpoint/swa_fold1_no_bn_model.pth')


############



def run_ensemble():
    ensemble_dir=[
        #'/root/share/project/kaggle/2020/grapheme_classification/result/run1/resnet34-fold0/submit/valid-swa-null',
        'dense121-fold0/submit/valid-swa-null',
        'dense121-fold1/submit/valid-swa-null',
        #'/root/share/project/kaggle/2020/grapheme_classification/result/run1/resnet34-fold0-cycle/submit/valid-swa-null',
        #'/root/share/project/kaggle/2020/grapheme_classification/result/run1/hrnet18-fold0/submit/valid-swa-null',
    ]
    out_dir = 'ensemble'



    ############################################################
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log.ensemble.txt',mode='a')

    if 1:
        test_probability = [0,0,0] # 8bit
        num_ensemble=0
        test_truth = None

        for t,d in enumerate(ensemble_dir):
            log.write('%d  %s\n'%(t,d))

            image_id    = read_list_from_file(d +'/image_id.txt')
            probability = read_pickle_from_file(d + '/probability.pickle')
            truth       = read_pickle_from_file(d + '/truth.pickle')

            test_probability = [p+q for p,q in zip(test_probability,probability)]
            num_ensemble += 1
        print('done')
        print('')

        probability = [p/num_ensemble for p in test_probability]

        #--------------

        recall, avgerage_recall = compute_kaggle_metric(probability, truth)
        log.write('avgerage_recall : %f\n'%(avgerage_recall))

        for i,name in enumerate(TASK_NAME):
            log.write('%28s  %f\n'%(name,recall[i]))
        log.write('\n')



        #----
'''
0 /root/share/project/kaggle/2020/grapheme_classification/result/run1/resnet34-fold0/submit/valid-swa-null
1 /root/share/project/kaggle/2020/grapheme_classification/result/run1/dense121-fold0/submit/valid-swa-null
done

avgerage_recall : 0.974879
               grapheme_root  0.963509
             vowel_diacritic  0.990332
         consonant_diacritic  0.982165



0  /root/share/project/kaggle/2020/grapheme_classification/result/run1/dense121-fold0/submit/valid-swa-null
1  /root/share/project/kaggle/2020/grapheme_classification/result/run1/resnet34-large-fold0/submit/valid-swa-null
done

avgerage_recall : 0.975554
               grapheme_root  0.964049
             vowel_diacritic  0.989347
         consonant_diacritic  0.984770
'''

# main #################################################################
if __name__ == '__main__':
    #print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_ensemble()
    #run_swa()

