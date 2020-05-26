from common import *

from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
#BatchNorm2d = nn.BatchNorm2d


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]



###############################################################################
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),
 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),
 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),
 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),
 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),
 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),
 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),
 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),
 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),
 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),
 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),
 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),
 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),
 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),
 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),
 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),
 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),
 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),
 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),
 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),
 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),
 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),
 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),
 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),
 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),
 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),
 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),
 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),
 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),
 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),
 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),
 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),
 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),
 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),
 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),
 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),
 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),
 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),
 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),
 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),
 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),
 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),
 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),
 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),
 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),
 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),
 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),
 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),
 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),
 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),
 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),
 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),
 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),
 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),
 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),
 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),
 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),
 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),
 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),
 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),
 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),
 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),
 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),
 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),
 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),
 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),
 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),
 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),
 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),
 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),
 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),
 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),
 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),
 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),
 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),
 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),
 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),
 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),
 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),
 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),
 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),
 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),
 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),
 'logit.bias',	(1000,),	 'fc.bias',	(1000,),

]
PRETRAIN_FILE = \
    '/root/share/project/kaggle/2019/steel/data/pretrain_model/resnet18-5c106cde.pth'

def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')



###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x




#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z



#resnet18
class ResNet18(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet18, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,2)],
        )
        self.logit = nn.Linear(512,num_class)



    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = F.max_pool2d(x,kernel_size=3, padding=1, stride=2, ceil_mode=False)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    net = ResNet18()
    load_pretrain(net, is_print=True)

    #---
    if 0:
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
        exit(0)

    #---
    if 1:
        #IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
        #IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

        net = net.cuda().eval()

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.resize(image,dsize=(224,224))
            #image = image[16:16+224,16:16+224]

            image = image[:,:,::-1]
            image = (image.astype(np.float32)/255 -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
            input = image.transpose(2,0,1)

            input = torch.from_numpy(input).float().cuda().unsqueeze(0)

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(f, image.shape)
            print(probability[:5])
            for t in range(5):
                print(t, '%5d'%argsort[t], probability[argsort[t]])
            print('')

            pass

    print('\nsucess!')

##############################################################################3
'''

great_white_shark (224, 224, 3)
[9.4850137e-09 1.0241954e-10 6.3554895e-01 3.6209035e-01 7.0319895e-04]
0     2 0.63554895
1     3 0.36209035
2     4 0.00070319895
3   391 0.0006877669
4   394 0.0005033317

screwdriver (224, 224, 3)
[7.5342734e-07 8.0605433e-10 3.7223426e-08 3.4161702e-09 3.0139031e-09]
0   784 0.97836363
1   845 0.017194362
2   418 0.0009959626
3   744 0.0008223861
4   657 0.0006719876

ostrich (224, 224, 3)
[2.2530783e-11 6.3119941e-13 1.6578347e-10 7.2425117e-12 1.0529154e-11]
0     9 0.6178599
1   135 0.37783644
2   127 0.0022111784
3   128 0.00069762324
4   138 0.00046368755

blad_eagle (224, 224, 3)
[4.1533657e-14 1.6096601e-12 7.0071298e-16 9.8058961e-16 9.9926985e-17]
0    22 0.9502171
1    21 0.049710903
2    93 5.8040747e-05
3   135 1.1641874e-05
4   146 8.518575e-07

english_foxhound (224, 224, 3)
[3.3760391e-17 1.9117241e-11 2.1513479e-12 2.5913074e-15 4.0416227e-13]
0   167 0.8951152
1   166 0.07485257
2   215 0.014488099
3   218 0.011066306
4   162 0.004281139

goldfish (224, 224, 3)
[1.8774564e-13 9.9999976e-01 8.1598264e-17 2.6006566e-16 2.0891524e-16]
0     1 0.99999976
1    90 1.4076785e-07
2    88 1.1558041e-07
3   130 9.544082e-09
4    94 4.6506425e-09
'''