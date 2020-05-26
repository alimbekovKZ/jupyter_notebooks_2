import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *
from train   import *

######################################################################################

def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict


def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    return predict




def compute_metric(truth, predict):

    num = len(truth)
    t = truth.reshape(num*4,-1).astype(np.float32)
    p = predict.reshape(num*4,-1).astype(np.float32)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    h_neg = (p_sum == 0).astype(np.float32)
    h_pos = (p_sum >  0).astype(np.float32)
    d_pos = 2* (p*t).sum(-1)/((p+t).sum(-1)+1e-12)

    t_sum = t_sum.reshape(num,4)
    p_sum = p_sum.reshape(num,4)
    h_neg = h_neg.reshape(num,4)
    h_pos = h_pos.reshape(num,4)
    d_pos = d_pos.reshape(num,4)

    #for each class
    hit_neg = []
    hit_pos = []
    dice_pos = []
    for c in range(4):
        neg_index = np.where(t_sum[:,c]==0)[0]
        pos_index = np.where(t_sum[:,c]>=1)[0]
        hit_neg.append(h_neg[:,c][neg_index])
        hit_pos.append(h_pos[:,c][pos_index])
        dice_pos.append(d_pos[:,c][pos_index])

    ##
    hit_neg_all = np.concatenate(hit_neg).mean()
    hit_pos_all = np.concatenate(hit_pos).mean()
    hit_neg  = [np.nan_to_num(h.mean(),0) for h in hit_neg]
    hit_pos  = [np.nan_to_num(h.mean(),0) for h in hit_pos]
    dice_pos = [np.nan_to_num(d.mean(),0) for d in dice_pos]

    ## from kaggle probing ...
    kaggle_pos = np.array([ 128,43,741,120 ])
    kaggle_neg_all = 6172
    kaggle_all     = 1801*4
    kaggle = (hit_neg_all*kaggle_neg_all + sum(dice_pos*kaggle_pos))/kaggle_all


    #confusion matrix
    t = truth.transpose(1,0,2,3).reshape(4,-1)
    t = np.vstack([t.sum(0,keepdims=True)==0,t])
    p = predict.transpose(0,2,3,1).reshape(-1,4)
    p = np.hstack([p.sum(1,keepdims=True)==0,p])

    confusion = np.zeros((5,5), np.float32)
    for c in range(5):
        index = np.where(t[c]==1)[0]
        confusion[c] = p[index].sum(0)/len(index)


    #print (np.array_str(confusion, precision=3, suppress_small=True))
    return kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion



###################################################################################


def do_evaluate(net, test_dataset, augment=[], out_dir=None):

    test_loader = DataLoader(
            test_dataset,
            sampler     = SequentialSampler(test_dataset),
            batch_size  = 8,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
    )
    #----

    #def sharpen(p,t=0):
    def sharpen(p,t=0):
        if t!=0:
            return p**t
        else:
            return p


    test_num  = 0
    test_id   = []
    #test_image = []
    test_probability_label = []
    test_probability_mask  = []
    test_probability       = []
    test_truth_label = []
    test_truth_mask  = []

    start = timer()
    for t, (input, truth_mask, truth_label, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  data_parallel(net,input)  #net(input)
                probability = torch.softmax(logit,1)

                probability_mask = sharpen(probability,0)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            #5 224 crop
            # if '5crop' in augment:
            #     raise NotImplementedError
            #     for sx, sy in[ (16,16),(0,0),(32,0),(32,0),(32,32) ]:
            #         crop = input[:,:,sy:sy+224,sx:sx+1568]
            #         #print(crop.shape)
            #         logit = data_parallel(net,crop)
            #         probability = torch.sigmoid(logit)
            #
            #         probability_mask += sharpen(probability)
            #         num_augment+=1

            # if 0: #scale /shift
            #     raise NotImplementedError
            #     input_pad = F.pad(input0,[8,8,8,8], mode='constant',value=0)
            #     for sx, sy in[ (8,0),(8,16),(0,8),(16,8) ]:
            #         logit =  data_parallel(net,input_pad[:,:,sy:sy+H,sx:sx+W])
            #         probability = torch.sigmoid(logit)
            #
            #         probability_mask += sharpen(probability)
            #         num_augment+=1

            probability_mask = probability_mask/num_augment
            probability = probability_mask.clone()

            probability_mask = one_hot_encode_predict(probability_mask)
            truth_mask  = one_hot_encode_truth(truth_mask)
        #---
        batch_size = len(infor)
        #image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
        truth_mask  = truth_mask.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        probability_mask = probability_mask.data.cpu().numpy()
        probability = probability.data.cpu().numpy()

        test_id.extend([i.image_id for i in infor])
        #test_image.append(image)
        test_probability.append(probability)
        test_probability_mask.append(probability_mask)
        test_truth_mask.append(truth_mask)
        test_truth_label.append(truth_label)
        test_num += batch_size

       # debug-----------------------------
        if out_dir is not None:
            probability = torch.softmax(logit,1)
            image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

            probability = one_hot_encode_predict(probability)
            truth_mask  = one_hot_encode_truth(truth_mask)

            probability_mask = probability.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = truth_mask.data.cpu().numpy()

            for b in range(0, batch_size, 4):
                result = draw_predict_result(image[b], truth_mask[b], truth_label[b], probability_mask[b], stack='vertical')
                image_show('result',result,resize=1)
                cv2.imwrite(out_dir +'/valid/%s.png'%(infor[b].image_id[:-4]), result)
                cv2.waitKey(1)
                pass
        # debug-----------------------------

        #---
        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')


    #test_image = np.concatenate(test_image)
    test_probability = np.concatenate(test_probability)
    test_probability_mask = np.concatenate(test_probability_mask)
    test_truth_mask  = np.concatenate(test_truth_mask)
    test_truth_label = np.concatenate(test_truth_label)

    return test_id, test_probability, test_probability_label, test_probability_mask, test_truth_label, test_truth_mask




#################################################################################################################
def run_submit():

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result1/resnet18-seg-full-softmax-foldb1-1-4balance'

    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result1/resnet18-seg-full-softmax-foldb1-1-4balance/checkpoint/00114000_model.pth'


    mode = 'test' #'train' # 'test'
    augment = ['null', 'flip_lr','flip_ud'] #['null'] # ['null, 'flip_lr','flip_ud','5crop']

    ## setup  -----------------------------------------------------------------------------

    os.makedirs(out_dir +'/submit/%s'%(mode), exist_ok=True)
    os.makedirs(out_dir +'/submit/%s/dump'%(mode), exist_ok=True)

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


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'train':
        test_dataset = SteelDataset(
            mode    = 'train',
            csv     = ['train.csv',],
            #split   = ['valid0_500.npy',],
            split   = ['valid_b1_1000.npy',],
            augment = None,
        )

    if mode == 'test':
        test_dataset = SteelDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_1801.npy',],
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)

    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net().cuda()
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n'%(type(net)))
    log.write('\n')

    ## start testing here! ##############################################
    #
    # test_id, test_image, test_probability_label, test_truth_mask, test_truth_label

    image_id, probability, probability_label, probability_mask, truth_label, truth_mask =\
        do_evaluate(net, test_dataset, augment)
        #do_evaluate(net, test_dataset, [])

    # if 0:
    #     # save for ensembling
    #     probability_mask = (probability_mask*255).astype(np.uint8)
    #     probability_mask[probability_mask<32]=0
    #
    #     probability = (probability*255).astype(np.uint8)
    #     probability[probability<32]=0
    #
    #     truth_mask = (truth_mask*255).astype(np.uint8)
    #
    #     write_list_to_file(out_dir + '/submit/%s/image_id.txt'%(mode),image_id)
    #     np.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode), truth_mask)
    #     np.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode), probability_mask)
    #     np.savez_compressed(out_dir + '/submit/%s/probability.uint8.npz'%(mode), probability)
    #     #exit(0)



    #---
    threshold_pixel = [0.50,0.50,0.50,0.50,]
    threshold_size  = [200,1500,1500,2000,]


    # inspect here !!!  ###################
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('threshold_pixel = %s\n'%str(threshold_pixel))
    log.write('threshold_size  = %s\n'%str(threshold_size))
    log.write('\n')

    if mode == 'train':

        #-----
        def log_train_metric():
            log.write('\n')
            log.write('kaggle      = %f\n'%kaggle)
            log.write('hit_neg_all = %f\n'%hit_neg_all)
            log.write('hit_pos_all = %f\n'%hit_pos_all)
            log.write('\n')

            log.write('* image level metric *\n')
            for c in range(4):
                log.write('dice_pos[%d], hit_pos[%d], hit_neg[%d] = %0.5f,  %0.5f,  %0.5f\n'%(
                    c+1,c+1,c+1,dice_pos[c],hit_pos[c],hit_neg[c]
                ))
            log.write('\n')

            log.write('confusion\n')
            log.write('%s\n'%(np.array_str(confusion, precision=3, suppress_small=True)))
            log.write('\n')
        #-----

        log.write('** after threshold_pixel **\n')

        predict_mask = probability_mask>np.array(threshold_pixel).reshape(1,4,1,1)

        kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion =\
            compute_metric(truth_mask, predict_mask)

        log_train_metric()

        #-----

        log.write('** after threshold_size **\n')

        predict_mask = remove_small(predict_mask, threshold_size)
        kaggle,hit_neg_all,hit_pos_all,hit_neg,hit_pos,dice_pos,confusion =\
            compute_metric(truth_mask, predict_mask)

        log_train_metric()


    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        csv_file = out_dir +'/submit/%s/resnet18-softmax-tta-0.50.csv'%(mode)


        predict_mask  = probability_mask>np.array(threshold_pixel).reshape(1,4,1,1)
        predict_mask  = remove_small(predict_mask, threshold_size)
        predict_label = ((predict_mask.sum(-1).sum(-1))>0).astype(np.int32)


        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(4):
                image_id_class_id.append(image_id[b]+'_%d'%(c+1))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)

        ## print statistics ----
        text = print_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))


    exit(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_submit()

    #merge_csv()