#https://github.com/junfu1115/DANet

from common  import *
from dataset import *
from resnet  import *


####################################################################################################

class Net(nn.Module):

    def load_pretrain(self, skip, is_print=True):
        conversion=copy.copy(CONVERSION)
        for i in range(0,len(conversion)-8,4):
            conversion[i] = 'block.' + conversion[i][5:]
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=conversion, is_print=is_print)

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit


#########################################################################


#def criterion(logit, truth, weight=[0.75,0.25]):
def criterion(logit, truth, weight=None):
    batch_size,num_class, H,W = logit.shape
    logit = logit.view(batch_size,num_class)
    truth = truth.view(batch_size,num_class)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()
        #raise NotImplementedError

    return loss



#----

def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives

        tp = tp.sum(dim=[0,2])
        tn = tn.sum(dim=[0,2])
        num_pos = t.sum(dim=[0,2])
        num_neg = batch_size*H*W - num_pos

        tp = tp.data.cpu().numpy()
        tn = tn.data.cpu().numpy().sum()
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().sum()

        tp = np.nan_to_num(tp/(num_pos+1e-12),0)
        tn = np.nan_to_num(tn/(num_neg+1e-12),0)

        tp = list(tp)
        num_pos = list(num_pos)

    return tn,tp, num_neg,num_pos




##############################################################################################
def make_dummy_data(folder='256x256', batch_size=8):

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/dump/%s/image/*.png'%folder) #32
    image_file = sorted(image_file)

    input=[]
    truth_mask =[]
    truth_label=[]
    for b in range(0, batch_size):
        i = b%len(image_file)
        image = cv2.imread(image_file[i], cv2.IMREAD_COLOR)
        mask  = np.load(image_file[i].replace('/image/','/mask/').replace('.png','.npy'))
        label = (mask.reshape(4,-1).sum(1)>0).astype(np.int32)

        input.append(image)
        truth_mask.append(mask)
        truth_label.append(label)

    input = np.array(input)
    input = image_to_input(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

    truth_mask = np.array(truth_mask)
    truth_mask = (truth_mask>0).astype(np.float32)

    truth_label = np.array(truth_label)

    infor= None

    return input, truth_mask, truth_label, infor








#########################################################################
def run_check_basenet():
    net = Net()
    print(net)
    #---
    if 1:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')

    net.load_pretrain(skip=['logit'])



def run_check_net():

    batch_size = 1
    C, H, W    = 3, 256, 1600
    num_class  = 4

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net(num_class=num_class).cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ',input.shape)
    print('logit: ',logit.shape)
    #print(net)


def run_check_train():


    if 1:
        input, truth_mask, truth_label, infor = make_dummy_data(folder='256x256', batch_size=20)
        batch_size, C, H, W  = input.shape

        print(input.shape)
        print(truth_label.shape)
        print(truth_mask.shape)
        print(truth_label.sum(0))

    #---
    truth_mask = torch.from_numpy(truth_mask).float().cuda()
    truth_label = torch.from_numpy(truth_label).float().cuda()
    input = torch.from_numpy(input).float().cuda()


    net = Net(drop_connect_rate=0.1).cuda()
    net.load_pretrain(skip=['logit'],is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit = net(input)
        loss = criterion(logit, truth_label)
        tn,tp, num_neg,num_pos = metric_hit (logit, truth_label)

        print('loss = %0.5f'%loss.item())
        print('tn,tp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] '%(tn,tp[0],tp[1],tp[2],tp[3]))
        print('num_pos,num_neg = %d, [%d,%d,%d,%d] '%(num_neg,num_pos[0],num_pos[1],num_pos[2],num_pos[3]))
        print('')


    #exit(0)
    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('--------------------------------------------')
    print('[iter ]  loss     |  tn, [tp1,tp2,tp3,tp4]  ')
    print('--------------------------------------------')
          #[00000]  0.70383  | 0.00000, 0.46449


    i=0
    optimizer.zero_grad()
    while i<=50:

        net.train()
        optimizer.zero_grad()

        logit = net(input)
        loss = criterion(logit, truth_label)
        tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)

        (loss).backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f]  '%(
                i,
                loss.item(),
                tn,tp[0],tp[1],tp[2],tp[3],
            ))
        i = i+1
    print('')


    if 1:
        #net.eval()
        logit = net(input)
        probability = torch.sigmoid(logit)

        probability_label = probability.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask  = truth_mask.data.cpu().numpy()

        image = input_to_image(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)

        for b in range(batch_size):
            print('%d ------ '%(b))
            result = draw_predict_result_label(image[b], truth_mask[b], truth_label[b], probability_label[b])
            image_show('result',result, resize=0.5)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


