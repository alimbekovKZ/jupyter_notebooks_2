import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *




def valid_augment(image, mask, infor):
    return image, mask, infor

def train_augment(image, mask, infor):

    u=np.random.choice(3)
    if u==0:
        pass
    elif u==1:
        image, mask = do_random_crop_rescale(image, mask, 1600-(256-224), 224)
    elif u==2:
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600-(256-224), 224)

    if np.random.rand()>0.5:
        image = do_random_log_contast(image)

    if np.random.rand()>0.5:
        image, mask = do_flip_lr(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_flip_ud(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_noise(image, mask)

    return image, mask, infor



def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_mask  = []
    truth_label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_mask.append(batch[b][1])
        infor.append(batch[b][2])

        label = (batch[b][1].reshape(4,-1).sum(1)>8).astype(np.int32)
        truth_label.append(label)


    input = np.stack(input)
    input = image_to_input(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
    input = torch.from_numpy(input).float()

    truth_mask = np.stack(truth_mask)
    truth_mask = (truth_mask>0.5).astype(np.float32)
    truth_mask = torch.from_numpy(truth_mask).float()

    truth_label = np.array(truth_label)
    truth_label = torch.from_numpy(truth_label).float()

    return input, truth_mask, truth_label, infor

#------------------------------------

def do_valid(net, valid_loader, out_dir=None):

    valid_num  = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)

    for t, (input, truth_mask, truth_label, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        input = input.cuda()
        truth_mask  = truth_mask.cuda()
        truth_label = truth_label.cuda()

        with torch.no_grad():
            logit = data_parallel(net, input)  #net(input)
            loss  = criterion(logit, truth_label)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)


            #zz=0
        #---
        batch_size = len(infor)
        l = np.array([ loss.item(), tn,*tp])
        n = np.array([ batch_size, num_neg,*num_pos])
        valid_loss += l*n
        valid_num  += n

        #debug-----------------------------
        if out_dir is not None:
            probability = torch.sigmoid(logit)
            image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

            probability_label = probability.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = truth_mask.data.cpu().numpy()

            for b in range(0, batch_size, 4):
                image_id = infor[b].image_id[:-4]
                result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_label[b], stack='vertical')
                draw_shadow_text(result,'%05d    %s.jpg'%(valid_num[0]-batch_size+b, image_id),(5,24),0.75,[255,255,255],1)
                image_show('result',result,resize=1)
                cv2.imwrite(out_dir +'/valid/%s.png'%(image_id), result)
                cv2.waitKey(1)
                pass
        #debug-----------------------------

        #print(valid_loss)
        print('\r %8d /%8d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/valid_num

    return valid_loss



def run_train():
    out_dir = \
        '/root/share/project/kaggle/2019/steel/result1/resnet34-cls-full-foldb0-0'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result1/resnet34-cls-full-foldb0-0/checkpoint/00007500_model.pth'




    schduler = NullScheduler(lr=0.001)
    batch_size = 20 #8
    iter_accum =  1


    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b0_11568.npy',],
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        #sampler     = BalanceClassSampler(train_dataset, 3*len(train_dataset)),
        #sampler    = SequentialSampler(train_dataset),
        sampler    = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    valid_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid_b0_1000.npy',],
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler    = SequentialSampler(valid_dataset),
        #sampler     = RandomSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        #for k in ['logit.weight','logit.bias']: state_dict.pop(k, None)

        net.load_state_dict(state_dict,strict=False)
    else:
        load_pretrain(net.e, skip=['logit'], is_print=False)


    log.write('%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 1500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass



    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('   experiment  = %s\n' % __file__.split('/')[-2])
    log.write('                      |--------------- VALID-----------------|---------------------- TRAIN/BATCH ------------------\n')
    log.write('rate     iter   epoch |  loss    tn, [tp1,tp2,tp3,tp4]       |  loss    tn, [tp1,tp2,tp3,tp4]       | time        \n')
    log.write('--------------------------------------------------------------------------------------------------------------------\n')
              #0.00000   40.5*  26.8 |  0.124   0.98 [0.77,0.40,0.93,0.91]  |  0.000   0.00 [0.00,0.00,0.00,0.00]  |  0 hr 00 min


    train_loss = np.zeros(20,np.float32)
    valid_loss = np.zeros(20,np.float32)
    batch_loss = np.zeros(20,np.float32)
    iter = 0
    i    = 0


    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = np.zeros(20,np.float32)

        optimizer.zero_grad()
        for t, (input, truth_mask, truth_label, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


            #if 0:
            if (iter % iter_valid==0):
                valid_loss = do_valid(net, valid_loader, out_dir) #
                #pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:6],
                         *train_loss[:6],
                         time_to_str((timer() - start),'min'))
                )
                log.write('\n')




            #if 0:
            if iter in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                pass




            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask  = truth_mask.cuda()

            logit =  data_parallel(net,input)  #net(input)
            loss = criterion(logit, truth_label)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)


            (loss/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            l = np.array([ loss.item(), tn,*tp ])
            n = np.array([ batch_size, num_neg,*num_pos ])

            batch_loss[:6] = l
            sum_train_loss[:6] += l*n
            sum[:6] += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum+1e-12)
                sum_train_loss[...] = 0
                sum[...]            = 0


            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:6],
                         *batch_loss[:6],
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)
            i=i+1


            # debug-----------------------------
            if 1:
                for di in range(3):
                    if (iter+di)%1000==0:

                        probability = torch.sigmoid(logit)
                        image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

                        probability_label = probability.data.cpu().numpy()
                        truth_label = truth_label.data.cpu().numpy()
                        truth_mask  = truth_mask.data.cpu().numpy()


                        for b in range(batch_size):
                            result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_label[b], stack='vertical')

                            image_show('result',result,resize=1)
                            cv2.imwrite(out_dir +'/train/%05d.png'%(di*100+b), result)
                            cv2.waitKey(1)
                            pass



        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()



#####
#
#  03002.png
#
#  12
# 111111111111