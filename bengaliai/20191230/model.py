from common  import *
from dataset import *
from seresnext import *


class Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=NUM_CLASS):
        super(Net, self).__init__()
        e = ResNext50()
        self.rgb = e.rgb
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped


        self.logit = nn.ModuleList(
            [ nn.Linear(2048,c) for c in num_class ]
        )


    def forward(self, x):
        batch_size,C,H,W = x.shape
        if (H,W) !=(224,224):
            x = F.interpolate(x,size=(224,224), mode='bilinear',align_corners=False)

        x = self.rgb(x.repeat(1,3,1,1))
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x,0.2,self.training)

        logit = [l(x) for l in self.logit]
        return logit



def logit_to_probability(logit):
    probability = [ F.softmax(l,1) for l in logit ]
    return probability


#########################################################################
def to_onehot(truth, num_class):
    batch_size = len(truth)
    onehot = torch.zeros(batch_size,num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1,1),value=1)
    return onehot

def cross_entropy_onehot_loss(logit, onehot):
    #print(logit)
    #batch_size,num_class = logit.shape
    log_probability = -F.log_softmax(logit,1)
    loss = (log_probability*onehot)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss


def criterion(logit, truth):
    loss = []
    for l,t in zip(logit,truth):
        #e = F.cross_entropy(l, t)
        e = cross_entropy_onehot_loss(l, t)
        loss.append(e)

    return loss


def criterion_n(logit, truth):
    loss = []
    for l,t in zip(logit,truth):
        #e = F.cross_entropy(l, t)
        e = cross_entropy_onehot_loss(l, t)
        #loss.append(e)

    return e

#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(probability, truth):

    correct = []
    for p,t in zip(probability,truth):
        p = p.data.cpu().numpy()
        t = t.data.cpu().numpy()
        y = p.argmax(-1)
        c = np.mean(y==t)
        correct.append(c)

    return correct

##############################################################################################
def make_dummy_data(batch_size=128):

    data_dir = '/root/share/project/kaggle/2020/grapheme_classification/data/image/debug'
    image_file = glob.glob(data_dir+'/*.png')
    image = []
    for f in image_file:
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        m = m.astype(np.float32)/255
        image.append(m)

    num_image = len(image)
    infor = read_pickle_from_file(data_dir+'.infor.pickle')
    label = read_pickle_from_file(data_dir+'.label.pickle')

    batch = []
    for b in range(0, batch_size):
        i = b%num_image
        batch.append([image[i],label[i],infor[i]])

    input, truth, infor = null_collate(batch)
    input = input.cuda()
    truth = [t.cuda() for t in truth]

    return input, truth, infor





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

    net.load_pretrain()



def run_check_net():

    batch_size = 10
    C, H, W    = 1, 137, 236

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net().cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ',input.shape)
    for t in range(4):
        print('logit[%d]: '%t,logit[t].shape)

    #print(net)



def run_check_train():


    if 1:
        input, truth, infor = make_dummy_data(batch_size=40)
        onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]
        batch_size, C, H, W  = input.shape

        print('input: ',input.shape)
        for t in range(4):
            print('truth[%d]: '%t,truth[t].shape)
            print('onehot[%d]: '%t,onehot[t].shape)
        print('')

    #---

    net = Net().cuda()
    net.load_pretrain(is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit = net(input)
        probability = logit_to_probability(logit)

        print('input: ',input.shape)
        for t in range(4):
            print('logit[%d]: '%t,logit[t].shape)
        print('')

        loss = criterion(logit, onehot)
        correct = metric(probability, truth)

        print('loss    = %0.5f, %0.5f, %0.5f, %0.5f'%(*[l.item() for l in loss],))
        print('correct = %0.5f, %0.5f, %0.5f, %0.5f'%(*[c for c in correct],))
        print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('-------------------------------------------------------------------------------')
    print('[iter ]     loss                        |   correct                       |    ')
    print('-------------------------------------------------------------------------------')
          #[00050]   3.94932,  1.98113,  1.14171   |   0.05055,  0.10606,  0.19048   |
    i=0
    optimizer.zero_grad()
    while i<=250: #100

        net.train()
        optimizer.zero_grad()

        logit = net(input)
        probability = logit_to_probability(logit)
        loss = criterion(logit, onehot)
        correct = metric(probability, truth)

        (loss[0]+loss[1]+loss[2]+loss[3]).backward()
        optimizer.step()

        #----
        loss = [ l.item() for l in loss]
        if i%50==0:
            print(
                '[%05d] '%(i),
                '%8.5f, %8.5f, %8.5f, %8.5f   | '%(*loss,),
                '%8.5f, %8.5f, %8.5f, %8.5f   | '%(*correct,),
            )
        i = i+1
    print('')


    #exit(0)
    if 1:
        image = input_to_image(input)
        image = (image*255).astype(np.uint8)

        truth = [t.data.cpu().numpy() for t in truth]
        probability = [p.data.cpu().numpy() for p in probability]

        for b in range(batch_size):
            print('%2d --------------------------- '%(b))
            print('%s'%(str(infor[b])))
            for i,name in enumerate(list(TASK.keys())):
                t   = truth[i][b]
                p_t = probability[i][b,t]
                m   = probability[i][b].argmax()
                p_m = probability[i][b,m]
                print('%28s  |  truth : %4d, %0.5f  |  max : %4d, %0.5f'%(name,t,p_t,m,p_m))
            print('')
            image_show('image',image[b])
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


