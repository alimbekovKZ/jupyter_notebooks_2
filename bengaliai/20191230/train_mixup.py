import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *


################################################################################################
def do_mixup(input, onehot):
    batch_size = len(input)

    #print(len(onehot))
    #print(onehot[0].shape)

    alpha = 0.4  #0.2,0.4
    gamma = np.random.beta(alpha, alpha)
    gamma = max(1-gamma,gamma)

    # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
    perm = torch.randperm(batch_size).to(input.device)

    truth = onehot

    perm_input  = input[perm]
    perm_onehot = [t[perm] for t in onehot]
    #print(perm_onehot)
    #perm_target = [t[perm] for t in onehot]
    mix_input  = gamma*input + (1-gamma)*perm_input
    #mix_onehot = [gamma*t    + (1-gamma)*perm_t for t,perm_t in zip(truth, perm_onehot)]
    mix_onehot = [gamma*t    + (1-gamma)*perm_t for t,perm_t in zip(truth, perm_onehot)]
    return mix_input, mix_onehot

'''
def mixup(input, target):
    perm_target = target[perm]
    return  target.mul_(gamma).add_(1 - gamma, perm_target)
'''

def train_augment(image, label, infor):
    if np.random.rand()<0.5:
        image = do_grid_distortion(image, distort=0.25, num_step=5)
    if np.random.rand()<0.5:
        image = do_random_crop_rotate_rescale(image, mode={'rotate':15,'scale':0.25,'shift':0.05})
    return image, label, infor


def valid_augment(image, label, infor):
    return image, label, infor

#------------------------------------
def do_valid(net, valid_loader, out_dir=None):

    valid_loss = np.zeros(8, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    valid_probability = [[],[],[],[],]
    valid_truth = [[],[],[],[],]

    for t, (input, truth, infor) in enumerate(valid_loader):

        #if b==5: break
        batch_size = len(infor)

        net.eval()
        input  = input.cuda()
        truth  = [t.cuda() for t in truth]
        onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, onehot)
            correct = metric(probability, truth)

        #---
        loss = [l.item() for l in loss]
        l = np.array([ *loss, *correct, ])*batch_size
        n = np.array([ 1, 1, 1, 1, 1, 1, 1, 1  ])*batch_size
        valid_loss += l
        valid_num  += n

        #---
        for i in range(4):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        #print(valid_loss)
        print('\r %8d / %d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/(valid_num+1e-8)

    #------
    for i in range(4):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    average, componet, recall = compute_kaggle_metric(valid_probability, valid_truth)


    return valid_loss, (average, componet, recall)





def run_train():


    fold = 1
    out_dir = \
        'result/run6/seresnext50-4cls-mixup-224-fold%d'%fold
    initial_checkpoint = \
        out_dir + '/checkpoint/00081000_model.pth'
        #'/root/share/project/kaggle/2020/grapheme_classification/result/run6/seresnext50-4cls-224-fold1/checkpoint/00108000_model.pth'
        #None #
    
    initial_checkpoint = None

    schduler = NullScheduler(lr=0.005)
    iter_accum = 1
    batch_size = 40 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

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

    train_dataset = KaggleDataset(
        mode    = 'train',
        csv     = 'train.csv',
        split   = 'balance2/train_b_fold%d_184855.npy'%fold,
        # parquet = ['train_image_data_0.parquet',
        #            'train_image_data_1.parquet',
        #            'train_image_data_2.parquet',
        #            'train_image_data_3.parquet',
        #            ],
        parquet = None,
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        #sampler     = RandomSampler(train_dataset),
        sampler     = BalanceSampler(train_dataset,184855),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    valid_dataset = KaggleDataset(
        mode    = 'train',
        csv     = 'train.csv',
        split   = 'balance2/valid_b_fold%d_15985.npy'%fold,
        parquet = None,
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = 64,
        drop_last   = False,
        num_workers = 4,
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
        # for k in list(state_dict.keys()):
        #      if any(s in k for s in ['logit',]): state_dict.pop(k, None)
        # net.load_state_dict(state_dict,strict=False)

        net.load_state_dict(state_dict,strict=True)  #True
    else:
        net.load_pretrain(is_print=False)


    log.write('net=%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay=0.0)

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 250
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1000))#1*1000

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
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                    |------------------------------------- VALID-----------------------------------------|------- TRAIN/BATCH ------------------\n')
    log.write('rate    iter  epoch | kaggle                           | loss                     acc                    | loss                   | time        \n')
    log.write('------------------------------------------------------------------------------------------------------------------------------------------------\n')
              #0.01000   0.5   0.2 | 0.648 : 0.508 0.830 0.745  0.204 | 1.11, 0.29, 0.19, 5.31 : 0.72, 0.91, 0.94, 0.20 | 1.41, 0.39, 0.27, 5.88 | 0 hr 05 min

    def message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f %5.1f%s %4.1f | '%(rate, iter/1000, asterisk, epoch,) +\
            '%0.3f : %0.3f %0.3f %0.3f  %0.3f | '%(kaggle[0],*kaggle[1]) +\
            '%4.2f, %4.2f, %4.2f, %4.2f : %4.2f, %4.2f, %4.2f, %4.2f | '%(*valid_loss,) +\
            '%4.2f, %4.2f, %4.2f, %4.2f |'%(*loss,) +\
            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    kaggle = (0,0,0,0)
    valid_loss = np.zeros(8,np.float32)
    train_loss = np.zeros(4,np.float32)
    batch_loss = np.zeros_like(train_loss)
    iter = 0
    i    = 0



    start_timer = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


            #if 0:
            if (iter % iter_valid==0):
                valid_loss, kaggle = do_valid(net, valid_loader, out_dir) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='log'))
                log.write('\n')

            #if 0:
            if iter in iter_save:
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
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
            truth = [t.cuda() for t in truth]
            onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]

            with torch.no_grad():
                input, onehot = do_mixup(input, onehot)

                
            logit = data_parallel(net, input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, onehot)

            (( 2*loss[0]+loss[1]+loss[2]+0.1*loss[3] )/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            loss = [l.item() for l in loss]
            l = np.array([ *loss, ])*batch_size
            n = np.array([ 1, 1, 1, 1 ])*batch_size
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0


            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
