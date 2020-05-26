import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *

from kaggle import *
import seaborn as sn

#
######################################################################################


def do_evaluate(net, test_dataset, augment=[], out_dir=None):

    test_loader = DataLoader(
        test_dataset,
        sampler     = SequentialSampler(test_dataset),
        batch_size  = 64,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    #----
    start_timer = timer()

    test_num  = 0
    test_id   = []
    test_probability = [[],[],[],[]]
    test_truth = [[],[],[],[]]

    start_timer = timer()
    for t, (input, truth, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment=0
            probability=[0,0,0,0]
            if 'null' in augment: #null
                logit =  data_parallel(net,input)  #net(input)
                prob  = logit_to_probability(logit)

                probability = [p+q**0.5 for p,q in zip(probability,prob)]
                num_augment += 1

            if 'scale' in augment: #scale
                scale = [1.2,0.90,]#[1.2,]
                for s in scale:
                    input_scale = F.interpolate(input, scale_factor=s,mode='bilinear',align_corners=False)
                    if s>1:
                        x = int((s-1.0)/2*236)
                        y = int((s-1.0)/2*137)
                        input_scale = input_scale[:,:,y:y+137,x:x+236]#137, 236
                    if s<1:
                        x = int((1.0-s)/2*236)
                        y = int((1.0-s)/2*137)
                        h,w = input_scale.shape[2:]
                        input_scale = F.pad(input_scale, pad=[x,236-w-x,y,137-h-y,], mode='reflect')

                    logit = data_parallel(net,input_scale)
                    prob  = logit_to_probability(logit)

                    probability = [p+q**0.5/len(scale) for p,q in zip(probability,prob)]
                    num_augment += 1/len(scale)

            #---
            probability = [p/num_augment for p in probability]

        #---
        batch_size  = len(infor)
        for i in range(NUM_TASK):
            test_probability[i].append(probability[i].data.cpu().numpy())
            test_truth[i].append(truth[i].data.cpu().numpy())

        test_id.extend([i.image_id for i in infor])
        test_num += batch_size

        #---
        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start_timer),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    for i in range(NUM_TASK):
        test_probability[i] = np.concatenate(test_probability[i])
        test_truth[i] = np.concatenate(test_truth[i])

    print(time_to_str((timer() - start_timer),'sec'))
    return test_id, test_truth, test_probability


######################################################################################
def run_submit():



    valid_split = 'balance2/valid_b_fold0_15985.npy'
    out_dir = 'result/run6/seresnext50-4cls-224-fold0'
    initial_checkpoint = out_dir + '/checkpoint/00118000_model.pth'
        #out_dir + '/checkpoint/swa_fold0_no_bn_model.pth'
        #out_dir + '/checkpoint/00154000_model.pth'

    valid_split = 'balance2/valid_b_fold0_15985.npy'
    out_dir = 'result/run6/seresnext50-4cls-224-fold0'
    initial_checkpoint = out_dir + '/checkpoint/00124000_model.pth'

    valid_split = 'balance2/valid_b_fold0_15985.npy'
    out_dir = 'result/run7/seresnext50-fold1'
    initial_checkpoint = out_dir + '/checkpoint/00069000_model.pth'

    #valid_split = 'balance2/valid_b_fold2_15985.npy'
    #out_dir = '/root/share/project/kaggle/2020/grapheme_classification/result/run6/seresnext50-4cls-224-fold2'
    #initial_checkpoint = out_dir + '/checkpoint/swa_fold2_no_bn_model.pth'

    #valid_split = 'balance2/valid_b_fold1_15985.npy'
    #out_dir = '/root/share/project/kaggle/2020/grapheme_classification/result/run6/seresnext50-4cls-224-fold1'
    #initial_checkpoint = out_dir + '/checkpoint/swa_fold1_no_bn_model.pth'



    ###############################################################
    augment = ['null', ]  #['null','scale'] #['null', 'scale' ]  #
    mode_folder = 'valid-swa-null' #tta  null

    #---

    ## setup
    os.makedirs(out_dir +'/submit/%s'%(mode_folder), exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## start testing here! ##############################################
    #
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('initial_checkpoint  = %s\n'%initial_checkpoint)
    log.write('\n')

    if 1: #save
        def valid_augment(image, label, infor):
            return image, label, infor

        log.write('** dataset setting **\n')
        test_dataset = KaggleDataset(
            mode    = 'train',
            csv     = 'train.csv',
            split   = valid_split,
            parquet = None,
            augment = valid_augment,
        )
        log.write('test_dataset : \n%s\n'%(test_dataset))
        log.write('\n')

        ## net
        log.write('** net setting **\n')
        net = Net().cuda()
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)

        image_id, truth, probability = do_evaluate(net, test_dataset, augment)


        if 1: #save
            write_list_to_file (out_dir + '/submit/%s/image_id.txt'%(mode_folder),image_id)
            write_pickle_to_file(out_dir + '/submit/%s/probability.pickle'%(mode_folder), probability)
            write_pickle_to_file(out_dir + '/submit/%s/truth.pickle'%(mode_folder), truth)

        #exit(0)

    if 1:
        image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt'%(mode_folder))
        probability = read_pickle_from_file(out_dir + '/submit/%s/probability.pickle'%(mode_folder))
        truth       = read_pickle_from_file(out_dir + '/submit/%s/truth.pickle'%(mode_folder))
    num_test= len(image_id)


    # inspect here !!!  ###################
    if 1:
        average, componet, recall = compute_kaggle_metric(probability, truth)
        log.write('avgerage recall : %f\n'%(average))

        for i,name in enumerate(list(TASK.keys())):
            log.write('%28s  %f\n'%(name, componet[i]))
        log.write('\n')


        if 0:
            #confusion matrix
            confusion = np.zeros((168,168), np.float32)
            for c in range(168):
                index = np.where(truth[0]==c)
                p = probability[0][index]
                confusion[c] = p.mean(0)

            df_confusion = pd.DataFrame(confusion, range(168), range(168))
            #sn.set(font_scale=1.0)#for label size
            #sn.heatmap(df_confusion, annot=True, annot_kws={"size": 8})# font size


            #plt.matshow(confusion)
            plt.matshow(np.log(confusion+1))
            plt.show()

'''
submitting .... @ ['null']
initial_checkpoint  = /root/share/project/kaggle/2020/grapheme_classification/result/run6/dense121-4cls-256-fold0/checkpoint/swa_fold0_no_bn_model.pth

avgerage recall : 0.976437
               grapheme_root  0.966045
             vowel_diacritic  0.989336
         consonant_diacritic  0.984321
                    grapheme  0.964135 


submitting .... @ ['null']
initial_checkpoint  = /root/share/project/kaggle/2020/grapheme_classification/result/run6/dense121-4cls-256-fold0/checkpoint/00129000_model.pth
** net setting **
 15985 / 15985   0 hr 00 min
 0 min 39 sec
avgerage recall : 0.976314
               grapheme_root  0.965552
             vowel_diacritic  0.989474
         consonant_diacritic  0.984676
                    grapheme  0.964580


submitting .... @ ['null']
initial_checkpoint  = /root/share/project/kaggle/2020/grapheme_classification/result/run6/dense121-4cls-256-fold0/checkpoint/00127000_model.pth

 15985 / 15985   0 hr 00 min
 0 min 39 sec
avgerage recall : 0.976666
               grapheme_root  0.965906
             vowel_diacritic  0.989157
         consonant_diacritic  0.985693
                    grapheme  0.963676

'''



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_submit()
  
