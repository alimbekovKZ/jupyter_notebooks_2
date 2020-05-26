from common  import *
from dataset import *
from seresnext import *


class Net(nn.Module):
    def load_pretrain(self, skip=['block0.','logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=NUM_CLASS):
        super(Net, self).__init__()
        e = ResNext50()

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
        if (H,W) !=(64,112):
             x = F.interpolate(x,size=(64,112), mode='bilinear',align_corners=False)

        x = x.repeat(1,3,1,1)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x,0.25,self.training)

        feature = None
        logit = [l(x) for l in self.logit]
        return logit, feature



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
    batch_size,num_class = logit.shape
    log_probability = -F.log_softmax(logit,1)
    loss = (log_probability*onehot)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss

def focal_onehot_loss(logit, onehot):
    alpha = 2.0

    batch_size,num_class = logit.shape
    probability = F.softmax(logit,1)
    weight = (1-probability)**alpha
    weight = weight/(onehot*weight).sum().item()*batch_size

    log_probability = -F.log_softmax(logit,1)
    loss = (log_probability*onehot*weight)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss


def criterion(logit, truth):

    loss = []
    for i in range(4):
        #e = F.cross_entropy(logit[i], truth[i])
        e = cross_entropy_onehot_loss(logit[i], truth[i])
        #e = focal_onehot_loss(logit[i], truth[i])
        loss.append(e)

    return loss

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
    infor = read_pickle_from_file(data_dir+'.infor.pickle')
    label = read_pickle_from_file(data_dir+'.label.pickle')
    num_image = len(infor)
    image = []
    for b in range(num_image):
        image_file = data_dir + '/%s.png'%(infor[b].image_id)
        m = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        m = 1-m.astype(np.float32)/255
        m = to_64x112(m)
        image.append(m)


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
        logit, feature = net(input)

    print('')
    print('input: ',input.shape)
    for t in range(4):
        print('logit[%d]: '%t,logit[t].shape)

    #print(net)



def run_check_train():

    if 1:
        input, truth, infor = make_dummy_data(batch_size=64)
        onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]
        batch_size, C, H, W  = input.shape

        print('input: ',input.shape)
        for t in range(4):
            print('truth [%d]: '%t,truth[t].shape)
            print('onehot[%d]: '%t,onehot[t].shape)
            print('')
        print('')

    #---

    net = Net().cuda()
    net.load_pretrain(is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit, feature = net(input)
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
    print('-----------------------------------------------------------------------------------------------')
    print('[iter ]     loss                                  |   correct                                 |')
    print('-----------------------------------------------------------------------------------------------')
          #[00000]   5.19262,  2.49344,  1.92460,  7.18336   |   0.00000,  0.05000,  0.30000,  0.00000   |
    i=0
    optimizer.zero_grad()
    while i<=250: #100

        net.train()
        optimizer.zero_grad()

        logit, feature = net(input)
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

            overlay1 = draw_grapheme_compose(truth[-1][b])

            for i,name in enumerate(list(TASK.keys())):
                t   = truth[i][b]
                p_t = probability[i][b,t]
                m   = probability[i][b].argmax()
                p_m = probability[i][b,m]
                print('%28s  |  truth : %4d, %0.5f  |  max : %4d, %0.5f'%(name,t,p_t,m,p_m))
            print('')
            image_show('overlay1',overlay1)
            image_show('image',image[b])
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


