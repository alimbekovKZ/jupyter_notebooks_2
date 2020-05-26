from common import *


BatchNorm2d = nn.BatchNorm2d


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]


###############################################################################
PRETRAIN_FILE = \
    '/home/renat/.torch/models/densenet121-a639ec97.pth'
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'features.conv0.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'features.norm0.weight',	(64,),
 'block0.1.bias',	(64,),	 'features.norm0.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'features.norm0.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'features.norm0.running_var',	(64,),
 'block1.0.denselayer1.norm1.weight',	(64,),	 'features.denseblock1.denselayer1.norm1.weight',	(64,),
 'block1.0.denselayer1.norm1.bias',	(64,),	 'features.denseblock1.denselayer1.norm1.bias',	(64,),
 'block1.0.denselayer1.norm1.running_mean',	(64,),	 'features.denseblock1.denselayer1.norm1.running_mean',	(64,),
 'block1.0.denselayer1.norm1.running_var',	(64,),	 'features.denseblock1.denselayer1.norm1.running_var',	(64,),
 'block1.0.denselayer1.conv1.weight',	(128, 64, 1, 1),	 'features.denseblock1.denselayer1.conv1.weight',	(128, 64, 1, 1),
 'block1.0.denselayer1.norm2.weight',	(128,),	 'features.denseblock1.denselayer1.norm2.weight',	(128,),
 'block1.0.denselayer1.norm2.bias',	(128,),	 'features.denseblock1.denselayer1.norm2.bias',	(128,),
 'block1.0.denselayer1.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer1.norm2.running_mean',	(128,),
 'block1.0.denselayer1.norm2.running_var',	(128,),	 'features.denseblock1.denselayer1.norm2.running_var',	(128,),
 'block1.0.denselayer1.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer1.conv2.weight',	(32, 128, 3, 3),
 'block1.0.denselayer2.norm1.weight',	(96,),	 'features.denseblock1.denselayer2.norm1.weight',	(96,),
 'block1.0.denselayer2.norm1.bias',	(96,),	 'features.denseblock1.denselayer2.norm1.bias',	(96,),
 'block1.0.denselayer2.norm1.running_mean',	(96,),	 'features.denseblock1.denselayer2.norm1.running_mean',	(96,),
 'block1.0.denselayer2.norm1.running_var',	(96,),	 'features.denseblock1.denselayer2.norm1.running_var',	(96,),
 'block1.0.denselayer2.conv1.weight',	(128, 96, 1, 1),	 'features.denseblock1.denselayer2.conv1.weight',	(128, 96, 1, 1),
 'block1.0.denselayer2.norm2.weight',	(128,),	 'features.denseblock1.denselayer2.norm2.weight',	(128,),
 'block1.0.denselayer2.norm2.bias',	(128,),	 'features.denseblock1.denselayer2.norm2.bias',	(128,),
 'block1.0.denselayer2.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer2.norm2.running_mean',	(128,),
 'block1.0.denselayer2.norm2.running_var',	(128,),	 'features.denseblock1.denselayer2.norm2.running_var',	(128,),
 'block1.0.denselayer2.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer2.conv2.weight',	(32, 128, 3, 3),
 'block1.0.denselayer3.norm1.weight',	(128,),	 'features.denseblock1.denselayer3.norm1.weight',	(128,),
 'block1.0.denselayer3.norm1.bias',	(128,),	 'features.denseblock1.denselayer3.norm1.bias',	(128,),
 'block1.0.denselayer3.norm1.running_mean',	(128,),	 'features.denseblock1.denselayer3.norm1.running_mean',	(128,),
 'block1.0.denselayer3.norm1.running_var',	(128,),	 'features.denseblock1.denselayer3.norm1.running_var',	(128,),
 'block1.0.denselayer3.conv1.weight',	(128, 128, 1, 1),	 'features.denseblock1.denselayer3.conv1.weight',	(128, 128, 1, 1),
 'block1.0.denselayer3.norm2.weight',	(128,),	 'features.denseblock1.denselayer3.norm2.weight',	(128,),
 'block1.0.denselayer3.norm2.bias',	(128,),	 'features.denseblock1.denselayer3.norm2.bias',	(128,),
 'block1.0.denselayer3.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer3.norm2.running_mean',	(128,),
 'block1.0.denselayer3.norm2.running_var',	(128,),	 'features.denseblock1.denselayer3.norm2.running_var',	(128,),
 'block1.0.denselayer3.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer3.conv2.weight',	(32, 128, 3, 3),
 'block1.0.denselayer4.norm1.weight',	(160,),	 'features.denseblock1.denselayer4.norm1.weight',	(160,),
 'block1.0.denselayer4.norm1.bias',	(160,),	 'features.denseblock1.denselayer4.norm1.bias',	(160,),
 'block1.0.denselayer4.norm1.running_mean',	(160,),	 'features.denseblock1.denselayer4.norm1.running_mean',	(160,),
 'block1.0.denselayer4.norm1.running_var',	(160,),	 'features.denseblock1.denselayer4.norm1.running_var',	(160,),
 'block1.0.denselayer4.conv1.weight',	(128, 160, 1, 1),	 'features.denseblock1.denselayer4.conv1.weight',	(128, 160, 1, 1),
 'block1.0.denselayer4.norm2.weight',	(128,),	 'features.denseblock1.denselayer4.norm2.weight',	(128,),
 'block1.0.denselayer4.norm2.bias',	(128,),	 'features.denseblock1.denselayer4.norm2.bias',	(128,),
 'block1.0.denselayer4.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer4.norm2.running_mean',	(128,),
 'block1.0.denselayer4.norm2.running_var',	(128,),	 'features.denseblock1.denselayer4.norm2.running_var',	(128,),
 'block1.0.denselayer4.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer4.conv2.weight',	(32, 128, 3, 3),
 'block1.0.denselayer5.norm1.weight',	(192,),	 'features.denseblock1.denselayer5.norm1.weight',	(192,),
 'block1.0.denselayer5.norm1.bias',	(192,),	 'features.denseblock1.denselayer5.norm1.bias',	(192,),
 'block1.0.denselayer5.norm1.running_mean',	(192,),	 'features.denseblock1.denselayer5.norm1.running_mean',	(192,),
 'block1.0.denselayer5.norm1.running_var',	(192,),	 'features.denseblock1.denselayer5.norm1.running_var',	(192,),
 'block1.0.denselayer5.conv1.weight',	(128, 192, 1, 1),	 'features.denseblock1.denselayer5.conv1.weight',	(128, 192, 1, 1),
 'block1.0.denselayer5.norm2.weight',	(128,),	 'features.denseblock1.denselayer5.norm2.weight',	(128,),
 'block1.0.denselayer5.norm2.bias',	(128,),	 'features.denseblock1.denselayer5.norm2.bias',	(128,),
 'block1.0.denselayer5.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer5.norm2.running_mean',	(128,),
 'block1.0.denselayer5.norm2.running_var',	(128,),	 'features.denseblock1.denselayer5.norm2.running_var',	(128,),
 'block1.0.denselayer5.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer5.conv2.weight',	(32, 128, 3, 3),
 'block1.0.denselayer6.norm1.weight',	(224,),	 'features.denseblock1.denselayer6.norm1.weight',	(224,),
 'block1.0.denselayer6.norm1.bias',	(224,),	 'features.denseblock1.denselayer6.norm1.bias',	(224,),
 'block1.0.denselayer6.norm1.running_mean',	(224,),	 'features.denseblock1.denselayer6.norm1.running_mean',	(224,),
 'block1.0.denselayer6.norm1.running_var',	(224,),	 'features.denseblock1.denselayer6.norm1.running_var',	(224,),
 'block1.0.denselayer6.conv1.weight',	(128, 224, 1, 1),	 'features.denseblock1.denselayer6.conv1.weight',	(128, 224, 1, 1),
 'block1.0.denselayer6.norm2.weight',	(128,),	 'features.denseblock1.denselayer6.norm2.weight',	(128,),
 'block1.0.denselayer6.norm2.bias',	(128,),	 'features.denseblock1.denselayer6.norm2.bias',	(128,),
 'block1.0.denselayer6.norm2.running_mean',	(128,),	 'features.denseblock1.denselayer6.norm2.running_mean',	(128,),
 'block1.0.denselayer6.norm2.running_var',	(128,),	 'features.denseblock1.denselayer6.norm2.running_var',	(128,),
 'block1.0.denselayer6.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock1.denselayer6.conv2.weight',	(32, 128, 3, 3),
 'block1.1.norm.weight',	(256,),	 'features.transition1.norm.weight',	(256,),
 'block1.1.norm.bias',	(256,),	 'features.transition1.norm.bias',	(256,),
 'block1.1.norm.running_mean',	(256,),	 'features.transition1.norm.running_mean',	(256,),
 'block1.1.norm.running_var',	(256,),	 'features.transition1.norm.running_var',	(256,),
 'block1.1.conv.weight',	(128, 256, 1, 1),	 'features.transition1.conv.weight',	(128, 256, 1, 1),
 'block2.0.denselayer1.norm1.weight',	(128,),	 'features.denseblock2.denselayer1.norm1.weight',	(128,),
 'block2.0.denselayer1.norm1.bias',	(128,),	 'features.denseblock2.denselayer1.norm1.bias',	(128,),
 'block2.0.denselayer1.norm1.running_mean',	(128,),	 'features.denseblock2.denselayer1.norm1.running_mean',	(128,),
 'block2.0.denselayer1.norm1.running_var',	(128,),	 'features.denseblock2.denselayer1.norm1.running_var',	(128,),
 'block2.0.denselayer1.conv1.weight',	(128, 128, 1, 1),	 'features.denseblock2.denselayer1.conv1.weight',	(128, 128, 1, 1),
 'block2.0.denselayer1.norm2.weight',	(128,),	 'features.denseblock2.denselayer1.norm2.weight',	(128,),
 'block2.0.denselayer1.norm2.bias',	(128,),	 'features.denseblock2.denselayer1.norm2.bias',	(128,),
 'block2.0.denselayer1.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer1.norm2.running_mean',	(128,),
 'block2.0.denselayer1.norm2.running_var',	(128,),	 'features.denseblock2.denselayer1.norm2.running_var',	(128,),
 'block2.0.denselayer1.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer1.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer2.norm1.weight',	(160,),	 'features.denseblock2.denselayer2.norm1.weight',	(160,),
 'block2.0.denselayer2.norm1.bias',	(160,),	 'features.denseblock2.denselayer2.norm1.bias',	(160,),
 'block2.0.denselayer2.norm1.running_mean',	(160,),	 'features.denseblock2.denselayer2.norm1.running_mean',	(160,),
 'block2.0.denselayer2.norm1.running_var',	(160,),	 'features.denseblock2.denselayer2.norm1.running_var',	(160,),
 'block2.0.denselayer2.conv1.weight',	(128, 160, 1, 1),	 'features.denseblock2.denselayer2.conv1.weight',	(128, 160, 1, 1),
 'block2.0.denselayer2.norm2.weight',	(128,),	 'features.denseblock2.denselayer2.norm2.weight',	(128,),
 'block2.0.denselayer2.norm2.bias',	(128,),	 'features.denseblock2.denselayer2.norm2.bias',	(128,),
 'block2.0.denselayer2.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer2.norm2.running_mean',	(128,),
 'block2.0.denselayer2.norm2.running_var',	(128,),	 'features.denseblock2.denselayer2.norm2.running_var',	(128,),
 'block2.0.denselayer2.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer2.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer3.norm1.weight',	(192,),	 'features.denseblock2.denselayer3.norm1.weight',	(192,),
 'block2.0.denselayer3.norm1.bias',	(192,),	 'features.denseblock2.denselayer3.norm1.bias',	(192,),
 'block2.0.denselayer3.norm1.running_mean',	(192,),	 'features.denseblock2.denselayer3.norm1.running_mean',	(192,),
 'block2.0.denselayer3.norm1.running_var',	(192,),	 'features.denseblock2.denselayer3.norm1.running_var',	(192,),
 'block2.0.denselayer3.conv1.weight',	(128, 192, 1, 1),	 'features.denseblock2.denselayer3.conv1.weight',	(128, 192, 1, 1),
 'block2.0.denselayer3.norm2.weight',	(128,),	 'features.denseblock2.denselayer3.norm2.weight',	(128,),
 'block2.0.denselayer3.norm2.bias',	(128,),	 'features.denseblock2.denselayer3.norm2.bias',	(128,),
 'block2.0.denselayer3.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer3.norm2.running_mean',	(128,),
 'block2.0.denselayer3.norm2.running_var',	(128,),	 'features.denseblock2.denselayer3.norm2.running_var',	(128,),
 'block2.0.denselayer3.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer3.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer4.norm1.weight',	(224,),	 'features.denseblock2.denselayer4.norm1.weight',	(224,),
 'block2.0.denselayer4.norm1.bias',	(224,),	 'features.denseblock2.denselayer4.norm1.bias',	(224,),
 'block2.0.denselayer4.norm1.running_mean',	(224,),	 'features.denseblock2.denselayer4.norm1.running_mean',	(224,),
 'block2.0.denselayer4.norm1.running_var',	(224,),	 'features.denseblock2.denselayer4.norm1.running_var',	(224,),
 'block2.0.denselayer4.conv1.weight',	(128, 224, 1, 1),	 'features.denseblock2.denselayer4.conv1.weight',	(128, 224, 1, 1),
 'block2.0.denselayer4.norm2.weight',	(128,),	 'features.denseblock2.denselayer4.norm2.weight',	(128,),
 'block2.0.denselayer4.norm2.bias',	(128,),	 'features.denseblock2.denselayer4.norm2.bias',	(128,),
 'block2.0.denselayer4.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer4.norm2.running_mean',	(128,),
 'block2.0.denselayer4.norm2.running_var',	(128,),	 'features.denseblock2.denselayer4.norm2.running_var',	(128,),
 'block2.0.denselayer4.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer4.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer5.norm1.weight',	(256,),	 'features.denseblock2.denselayer5.norm1.weight',	(256,),
 'block2.0.denselayer5.norm1.bias',	(256,),	 'features.denseblock2.denselayer5.norm1.bias',	(256,),
 'block2.0.denselayer5.norm1.running_mean',	(256,),	 'features.denseblock2.denselayer5.norm1.running_mean',	(256,),
 'block2.0.denselayer5.norm1.running_var',	(256,),	 'features.denseblock2.denselayer5.norm1.running_var',	(256,),
 'block2.0.denselayer5.conv1.weight',	(128, 256, 1, 1),	 'features.denseblock2.denselayer5.conv1.weight',	(128, 256, 1, 1),
 'block2.0.denselayer5.norm2.weight',	(128,),	 'features.denseblock2.denselayer5.norm2.weight',	(128,),
 'block2.0.denselayer5.norm2.bias',	(128,),	 'features.denseblock2.denselayer5.norm2.bias',	(128,),
 'block2.0.denselayer5.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer5.norm2.running_mean',	(128,),
 'block2.0.denselayer5.norm2.running_var',	(128,),	 'features.denseblock2.denselayer5.norm2.running_var',	(128,),
 'block2.0.denselayer5.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer5.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer6.norm1.weight',	(288,),	 'features.denseblock2.denselayer6.norm1.weight',	(288,),
 'block2.0.denselayer6.norm1.bias',	(288,),	 'features.denseblock2.denselayer6.norm1.bias',	(288,),
 'block2.0.denselayer6.norm1.running_mean',	(288,),	 'features.denseblock2.denselayer6.norm1.running_mean',	(288,),
 'block2.0.denselayer6.norm1.running_var',	(288,),	 'features.denseblock2.denselayer6.norm1.running_var',	(288,),
 'block2.0.denselayer6.conv1.weight',	(128, 288, 1, 1),	 'features.denseblock2.denselayer6.conv1.weight',	(128, 288, 1, 1),
 'block2.0.denselayer6.norm2.weight',	(128,),	 'features.denseblock2.denselayer6.norm2.weight',	(128,),
 'block2.0.denselayer6.norm2.bias',	(128,),	 'features.denseblock2.denselayer6.norm2.bias',	(128,),
 'block2.0.denselayer6.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer6.norm2.running_mean',	(128,),
 'block2.0.denselayer6.norm2.running_var',	(128,),	 'features.denseblock2.denselayer6.norm2.running_var',	(128,),
 'block2.0.denselayer6.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer6.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer7.norm1.weight',	(320,),	 'features.denseblock2.denselayer7.norm1.weight',	(320,),
 'block2.0.denselayer7.norm1.bias',	(320,),	 'features.denseblock2.denselayer7.norm1.bias',	(320,),
 'block2.0.denselayer7.norm1.running_mean',	(320,),	 'features.denseblock2.denselayer7.norm1.running_mean',	(320,),
 'block2.0.denselayer7.norm1.running_var',	(320,),	 'features.denseblock2.denselayer7.norm1.running_var',	(320,),
 'block2.0.denselayer7.conv1.weight',	(128, 320, 1, 1),	 'features.denseblock2.denselayer7.conv1.weight',	(128, 320, 1, 1),
 'block2.0.denselayer7.norm2.weight',	(128,),	 'features.denseblock2.denselayer7.norm2.weight',	(128,),
 'block2.0.denselayer7.norm2.bias',	(128,),	 'features.denseblock2.denselayer7.norm2.bias',	(128,),
 'block2.0.denselayer7.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer7.norm2.running_mean',	(128,),
 'block2.0.denselayer7.norm2.running_var',	(128,),	 'features.denseblock2.denselayer7.norm2.running_var',	(128,),
 'block2.0.denselayer7.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer7.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer8.norm1.weight',	(352,),	 'features.denseblock2.denselayer8.norm1.weight',	(352,),
 'block2.0.denselayer8.norm1.bias',	(352,),	 'features.denseblock2.denselayer8.norm1.bias',	(352,),
 'block2.0.denselayer8.norm1.running_mean',	(352,),	 'features.denseblock2.denselayer8.norm1.running_mean',	(352,),
 'block2.0.denselayer8.norm1.running_var',	(352,),	 'features.denseblock2.denselayer8.norm1.running_var',	(352,),
 'block2.0.denselayer8.conv1.weight',	(128, 352, 1, 1),	 'features.denseblock2.denselayer8.conv1.weight',	(128, 352, 1, 1),
 'block2.0.denselayer8.norm2.weight',	(128,),	 'features.denseblock2.denselayer8.norm2.weight',	(128,),
 'block2.0.denselayer8.norm2.bias',	(128,),	 'features.denseblock2.denselayer8.norm2.bias',	(128,),
 'block2.0.denselayer8.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer8.norm2.running_mean',	(128,),
 'block2.0.denselayer8.norm2.running_var',	(128,),	 'features.denseblock2.denselayer8.norm2.running_var',	(128,),
 'block2.0.denselayer8.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer8.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer9.norm1.weight',	(384,),	 'features.denseblock2.denselayer9.norm1.weight',	(384,),
 'block2.0.denselayer9.norm1.bias',	(384,),	 'features.denseblock2.denselayer9.norm1.bias',	(384,),
 'block2.0.denselayer9.norm1.running_mean',	(384,),	 'features.denseblock2.denselayer9.norm1.running_mean',	(384,),
 'block2.0.denselayer9.norm1.running_var',	(384,),	 'features.denseblock2.denselayer9.norm1.running_var',	(384,),
 'block2.0.denselayer9.conv1.weight',	(128, 384, 1, 1),	 'features.denseblock2.denselayer9.conv1.weight',	(128, 384, 1, 1),
 'block2.0.denselayer9.norm2.weight',	(128,),	 'features.denseblock2.denselayer9.norm2.weight',	(128,),
 'block2.0.denselayer9.norm2.bias',	(128,),	 'features.denseblock2.denselayer9.norm2.bias',	(128,),
 'block2.0.denselayer9.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer9.norm2.running_mean',	(128,),
 'block2.0.denselayer9.norm2.running_var',	(128,),	 'features.denseblock2.denselayer9.norm2.running_var',	(128,),
 'block2.0.denselayer9.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer9.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer10.norm1.weight',	(416,),	 'features.denseblock2.denselayer10.norm1.weight',	(416,),
 'block2.0.denselayer10.norm1.bias',	(416,),	 'features.denseblock2.denselayer10.norm1.bias',	(416,),
 'block2.0.denselayer10.norm1.running_mean',	(416,),	 'features.denseblock2.denselayer10.norm1.running_mean',	(416,),
 'block2.0.denselayer10.norm1.running_var',	(416,),	 'features.denseblock2.denselayer10.norm1.running_var',	(416,),
 'block2.0.denselayer10.conv1.weight',	(128, 416, 1, 1),	 'features.denseblock2.denselayer10.conv1.weight',	(128, 416, 1, 1),
 'block2.0.denselayer10.norm2.weight',	(128,),	 'features.denseblock2.denselayer10.norm2.weight',	(128,),
 'block2.0.denselayer10.norm2.bias',	(128,),	 'features.denseblock2.denselayer10.norm2.bias',	(128,),
 'block2.0.denselayer10.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer10.norm2.running_mean',	(128,),
 'block2.0.denselayer10.norm2.running_var',	(128,),	 'features.denseblock2.denselayer10.norm2.running_var',	(128,),
 'block2.0.denselayer10.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer10.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer11.norm1.weight',	(448,),	 'features.denseblock2.denselayer11.norm1.weight',	(448,),
 'block2.0.denselayer11.norm1.bias',	(448,),	 'features.denseblock2.denselayer11.norm1.bias',	(448,),
 'block2.0.denselayer11.norm1.running_mean',	(448,),	 'features.denseblock2.denselayer11.norm1.running_mean',	(448,),
 'block2.0.denselayer11.norm1.running_var',	(448,),	 'features.denseblock2.denselayer11.norm1.running_var',	(448,),
 'block2.0.denselayer11.conv1.weight',	(128, 448, 1, 1),	 'features.denseblock2.denselayer11.conv1.weight',	(128, 448, 1, 1),
 'block2.0.denselayer11.norm2.weight',	(128,),	 'features.denseblock2.denselayer11.norm2.weight',	(128,),
 'block2.0.denselayer11.norm2.bias',	(128,),	 'features.denseblock2.denselayer11.norm2.bias',	(128,),
 'block2.0.denselayer11.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer11.norm2.running_mean',	(128,),
 'block2.0.denselayer11.norm2.running_var',	(128,),	 'features.denseblock2.denselayer11.norm2.running_var',	(128,),
 'block2.0.denselayer11.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer11.conv2.weight',	(32, 128, 3, 3),
 'block2.0.denselayer12.norm1.weight',	(480,),	 'features.denseblock2.denselayer12.norm1.weight',	(480,),
 'block2.0.denselayer12.norm1.bias',	(480,),	 'features.denseblock2.denselayer12.norm1.bias',	(480,),
 'block2.0.denselayer12.norm1.running_mean',	(480,),	 'features.denseblock2.denselayer12.norm1.running_mean',	(480,),
 'block2.0.denselayer12.norm1.running_var',	(480,),	 'features.denseblock2.denselayer12.norm1.running_var',	(480,),
 'block2.0.denselayer12.conv1.weight',	(128, 480, 1, 1),	 'features.denseblock2.denselayer12.conv1.weight',	(128, 480, 1, 1),
 'block2.0.denselayer12.norm2.weight',	(128,),	 'features.denseblock2.denselayer12.norm2.weight',	(128,),
 'block2.0.denselayer12.norm2.bias',	(128,),	 'features.denseblock2.denselayer12.norm2.bias',	(128,),
 'block2.0.denselayer12.norm2.running_mean',	(128,),	 'features.denseblock2.denselayer12.norm2.running_mean',	(128,),
 'block2.0.denselayer12.norm2.running_var',	(128,),	 'features.denseblock2.denselayer12.norm2.running_var',	(128,),
 'block2.0.denselayer12.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock2.denselayer12.conv2.weight',	(32, 128, 3, 3),
 'block2.1.norm.weight',	(512,),	 'features.transition2.norm.weight',	(512,),
 'block2.1.norm.bias',	(512,),	 'features.transition2.norm.bias',	(512,),
 'block2.1.norm.running_mean',	(512,),	 'features.transition2.norm.running_mean',	(512,),
 'block2.1.norm.running_var',	(512,),	 'features.transition2.norm.running_var',	(512,),
 'block2.1.conv.weight',	(256, 512, 1, 1),	 'features.transition2.conv.weight',	(256, 512, 1, 1),
 'block3.0.denselayer1.norm1.weight',	(256,),	 'features.denseblock3.denselayer1.norm1.weight',	(256,),
 'block3.0.denselayer1.norm1.bias',	(256,),	 'features.denseblock3.denselayer1.norm1.bias',	(256,),
 'block3.0.denselayer1.norm1.running_mean',	(256,),	 'features.denseblock3.denselayer1.norm1.running_mean',	(256,),
 'block3.0.denselayer1.norm1.running_var',	(256,),	 'features.denseblock3.denselayer1.norm1.running_var',	(256,),
 'block3.0.denselayer1.conv1.weight',	(128, 256, 1, 1),	 'features.denseblock3.denselayer1.conv1.weight',	(128, 256, 1, 1),
 'block3.0.denselayer1.norm2.weight',	(128,),	 'features.denseblock3.denselayer1.norm2.weight',	(128,),
 'block3.0.denselayer1.norm2.bias',	(128,),	 'features.denseblock3.denselayer1.norm2.bias',	(128,),
 'block3.0.denselayer1.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer1.norm2.running_mean',	(128,),
 'block3.0.denselayer1.norm2.running_var',	(128,),	 'features.denseblock3.denselayer1.norm2.running_var',	(128,),
 'block3.0.denselayer1.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer1.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer2.norm1.weight',	(288,),	 'features.denseblock3.denselayer2.norm1.weight',	(288,),
 'block3.0.denselayer2.norm1.bias',	(288,),	 'features.denseblock3.denselayer2.norm1.bias',	(288,),
 'block3.0.denselayer2.norm1.running_mean',	(288,),	 'features.denseblock3.denselayer2.norm1.running_mean',	(288,),
 'block3.0.denselayer2.norm1.running_var',	(288,),	 'features.denseblock3.denselayer2.norm1.running_var',	(288,),
 'block3.0.denselayer2.conv1.weight',	(128, 288, 1, 1),	 'features.denseblock3.denselayer2.conv1.weight',	(128, 288, 1, 1),
 'block3.0.denselayer2.norm2.weight',	(128,),	 'features.denseblock3.denselayer2.norm2.weight',	(128,),
 'block3.0.denselayer2.norm2.bias',	(128,),	 'features.denseblock3.denselayer2.norm2.bias',	(128,),
 'block3.0.denselayer2.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer2.norm2.running_mean',	(128,),
 'block3.0.denselayer2.norm2.running_var',	(128,),	 'features.denseblock3.denselayer2.norm2.running_var',	(128,),
 'block3.0.denselayer2.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer2.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer3.norm1.weight',	(320,),	 'features.denseblock3.denselayer3.norm1.weight',	(320,),
 'block3.0.denselayer3.norm1.bias',	(320,),	 'features.denseblock3.denselayer3.norm1.bias',	(320,),
 'block3.0.denselayer3.norm1.running_mean',	(320,),	 'features.denseblock3.denselayer3.norm1.running_mean',	(320,),
 'block3.0.denselayer3.norm1.running_var',	(320,),	 'features.denseblock3.denselayer3.norm1.running_var',	(320,),
 'block3.0.denselayer3.conv1.weight',	(128, 320, 1, 1),	 'features.denseblock3.denselayer3.conv1.weight',	(128, 320, 1, 1),
 'block3.0.denselayer3.norm2.weight',	(128,),	 'features.denseblock3.denselayer3.norm2.weight',	(128,),
 'block3.0.denselayer3.norm2.bias',	(128,),	 'features.denseblock3.denselayer3.norm2.bias',	(128,),
 'block3.0.denselayer3.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer3.norm2.running_mean',	(128,),
 'block3.0.denselayer3.norm2.running_var',	(128,),	 'features.denseblock3.denselayer3.norm2.running_var',	(128,),
 'block3.0.denselayer3.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer3.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer4.norm1.weight',	(352,),	 'features.denseblock3.denselayer4.norm1.weight',	(352,),
 'block3.0.denselayer4.norm1.bias',	(352,),	 'features.denseblock3.denselayer4.norm1.bias',	(352,),
 'block3.0.denselayer4.norm1.running_mean',	(352,),	 'features.denseblock3.denselayer4.norm1.running_mean',	(352,),
 'block3.0.denselayer4.norm1.running_var',	(352,),	 'features.denseblock3.denselayer4.norm1.running_var',	(352,),
 'block3.0.denselayer4.conv1.weight',	(128, 352, 1, 1),	 'features.denseblock3.denselayer4.conv1.weight',	(128, 352, 1, 1),
 'block3.0.denselayer4.norm2.weight',	(128,),	 'features.denseblock3.denselayer4.norm2.weight',	(128,),
 'block3.0.denselayer4.norm2.bias',	(128,),	 'features.denseblock3.denselayer4.norm2.bias',	(128,),
 'block3.0.denselayer4.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer4.norm2.running_mean',	(128,),
 'block3.0.denselayer4.norm2.running_var',	(128,),	 'features.denseblock3.denselayer4.norm2.running_var',	(128,),
 'block3.0.denselayer4.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer4.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer5.norm1.weight',	(384,),	 'features.denseblock3.denselayer5.norm1.weight',	(384,),
 'block3.0.denselayer5.norm1.bias',	(384,),	 'features.denseblock3.denselayer5.norm1.bias',	(384,),
 'block3.0.denselayer5.norm1.running_mean',	(384,),	 'features.denseblock3.denselayer5.norm1.running_mean',	(384,),
 'block3.0.denselayer5.norm1.running_var',	(384,),	 'features.denseblock3.denselayer5.norm1.running_var',	(384,),
 'block3.0.denselayer5.conv1.weight',	(128, 384, 1, 1),	 'features.denseblock3.denselayer5.conv1.weight',	(128, 384, 1, 1),
 'block3.0.denselayer5.norm2.weight',	(128,),	 'features.denseblock3.denselayer5.norm2.weight',	(128,),
 'block3.0.denselayer5.norm2.bias',	(128,),	 'features.denseblock3.denselayer5.norm2.bias',	(128,),
 'block3.0.denselayer5.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer5.norm2.running_mean',	(128,),
 'block3.0.denselayer5.norm2.running_var',	(128,),	 'features.denseblock3.denselayer5.norm2.running_var',	(128,),
 'block3.0.denselayer5.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer5.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer6.norm1.weight',	(416,),	 'features.denseblock3.denselayer6.norm1.weight',	(416,),
 'block3.0.denselayer6.norm1.bias',	(416,),	 'features.denseblock3.denselayer6.norm1.bias',	(416,),
 'block3.0.denselayer6.norm1.running_mean',	(416,),	 'features.denseblock3.denselayer6.norm1.running_mean',	(416,),
 'block3.0.denselayer6.norm1.running_var',	(416,),	 'features.denseblock3.denselayer6.norm1.running_var',	(416,),
 'block3.0.denselayer6.conv1.weight',	(128, 416, 1, 1),	 'features.denseblock3.denselayer6.conv1.weight',	(128, 416, 1, 1),
 'block3.0.denselayer6.norm2.weight',	(128,),	 'features.denseblock3.denselayer6.norm2.weight',	(128,),
 'block3.0.denselayer6.norm2.bias',	(128,),	 'features.denseblock3.denselayer6.norm2.bias',	(128,),
 'block3.0.denselayer6.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer6.norm2.running_mean',	(128,),
 'block3.0.denselayer6.norm2.running_var',	(128,),	 'features.denseblock3.denselayer6.norm2.running_var',	(128,),
 'block3.0.denselayer6.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer6.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer7.norm1.weight',	(448,),	 'features.denseblock3.denselayer7.norm1.weight',	(448,),
 'block3.0.denselayer7.norm1.bias',	(448,),	 'features.denseblock3.denselayer7.norm1.bias',	(448,),
 'block3.0.denselayer7.norm1.running_mean',	(448,),	 'features.denseblock3.denselayer7.norm1.running_mean',	(448,),
 'block3.0.denselayer7.norm1.running_var',	(448,),	 'features.denseblock3.denselayer7.norm1.running_var',	(448,),
 'block3.0.denselayer7.conv1.weight',	(128, 448, 1, 1),	 'features.denseblock3.denselayer7.conv1.weight',	(128, 448, 1, 1),
 'block3.0.denselayer7.norm2.weight',	(128,),	 'features.denseblock3.denselayer7.norm2.weight',	(128,),
 'block3.0.denselayer7.norm2.bias',	(128,),	 'features.denseblock3.denselayer7.norm2.bias',	(128,),
 'block3.0.denselayer7.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer7.norm2.running_mean',	(128,),
 'block3.0.denselayer7.norm2.running_var',	(128,),	 'features.denseblock3.denselayer7.norm2.running_var',	(128,),
 'block3.0.denselayer7.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer7.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer8.norm1.weight',	(480,),	 'features.denseblock3.denselayer8.norm1.weight',	(480,),
 'block3.0.denselayer8.norm1.bias',	(480,),	 'features.denseblock3.denselayer8.norm1.bias',	(480,),
 'block3.0.denselayer8.norm1.running_mean',	(480,),	 'features.denseblock3.denselayer8.norm1.running_mean',	(480,),
 'block3.0.denselayer8.norm1.running_var',	(480,),	 'features.denseblock3.denselayer8.norm1.running_var',	(480,),
 'block3.0.denselayer8.conv1.weight',	(128, 480, 1, 1),	 'features.denseblock3.denselayer8.conv1.weight',	(128, 480, 1, 1),
 'block3.0.denselayer8.norm2.weight',	(128,),	 'features.denseblock3.denselayer8.norm2.weight',	(128,),
 'block3.0.denselayer8.norm2.bias',	(128,),	 'features.denseblock3.denselayer8.norm2.bias',	(128,),
 'block3.0.denselayer8.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer8.norm2.running_mean',	(128,),
 'block3.0.denselayer8.norm2.running_var',	(128,),	 'features.denseblock3.denselayer8.norm2.running_var',	(128,),
 'block3.0.denselayer8.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer8.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer9.norm1.weight',	(512,),	 'features.denseblock3.denselayer9.norm1.weight',	(512,),
 'block3.0.denselayer9.norm1.bias',	(512,),	 'features.denseblock3.denselayer9.norm1.bias',	(512,),
 'block3.0.denselayer9.norm1.running_mean',	(512,),	 'features.denseblock3.denselayer9.norm1.running_mean',	(512,),
 'block3.0.denselayer9.norm1.running_var',	(512,),	 'features.denseblock3.denselayer9.norm1.running_var',	(512,),
 'block3.0.denselayer9.conv1.weight',	(128, 512, 1, 1),	 'features.denseblock3.denselayer9.conv1.weight',	(128, 512, 1, 1),
 'block3.0.denselayer9.norm2.weight',	(128,),	 'features.denseblock3.denselayer9.norm2.weight',	(128,),
 'block3.0.denselayer9.norm2.bias',	(128,),	 'features.denseblock3.denselayer9.norm2.bias',	(128,),
 'block3.0.denselayer9.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer9.norm2.running_mean',	(128,),
 'block3.0.denselayer9.norm2.running_var',	(128,),	 'features.denseblock3.denselayer9.norm2.running_var',	(128,),
 'block3.0.denselayer9.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer9.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer10.norm1.weight',	(544,),	 'features.denseblock3.denselayer10.norm1.weight',	(544,),
 'block3.0.denselayer10.norm1.bias',	(544,),	 'features.denseblock3.denselayer10.norm1.bias',	(544,),
 'block3.0.denselayer10.norm1.running_mean',	(544,),	 'features.denseblock3.denselayer10.norm1.running_mean',	(544,),
 'block3.0.denselayer10.norm1.running_var',	(544,),	 'features.denseblock3.denselayer10.norm1.running_var',	(544,),
 'block3.0.denselayer10.conv1.weight',	(128, 544, 1, 1),	 'features.denseblock3.denselayer10.conv1.weight',	(128, 544, 1, 1),
 'block3.0.denselayer10.norm2.weight',	(128,),	 'features.denseblock3.denselayer10.norm2.weight',	(128,),
 'block3.0.denselayer10.norm2.bias',	(128,),	 'features.denseblock3.denselayer10.norm2.bias',	(128,),
 'block3.0.denselayer10.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer10.norm2.running_mean',	(128,),
 'block3.0.denselayer10.norm2.running_var',	(128,),	 'features.denseblock3.denselayer10.norm2.running_var',	(128,),
 'block3.0.denselayer10.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer10.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer11.norm1.weight',	(576,),	 'features.denseblock3.denselayer11.norm1.weight',	(576,),
 'block3.0.denselayer11.norm1.bias',	(576,),	 'features.denseblock3.denselayer11.norm1.bias',	(576,),
 'block3.0.denselayer11.norm1.running_mean',	(576,),	 'features.denseblock3.denselayer11.norm1.running_mean',	(576,),
 'block3.0.denselayer11.norm1.running_var',	(576,),	 'features.denseblock3.denselayer11.norm1.running_var',	(576,),
 'block3.0.denselayer11.conv1.weight',	(128, 576, 1, 1),	 'features.denseblock3.denselayer11.conv1.weight',	(128, 576, 1, 1),
 'block3.0.denselayer11.norm2.weight',	(128,),	 'features.denseblock3.denselayer11.norm2.weight',	(128,),
 'block3.0.denselayer11.norm2.bias',	(128,),	 'features.denseblock3.denselayer11.norm2.bias',	(128,),
 'block3.0.denselayer11.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer11.norm2.running_mean',	(128,),
 'block3.0.denselayer11.norm2.running_var',	(128,),	 'features.denseblock3.denselayer11.norm2.running_var',	(128,),
 'block3.0.denselayer11.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer11.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer12.norm1.weight',	(608,),	 'features.denseblock3.denselayer12.norm1.weight',	(608,),
 'block3.0.denselayer12.norm1.bias',	(608,),	 'features.denseblock3.denselayer12.norm1.bias',	(608,),
 'block3.0.denselayer12.norm1.running_mean',	(608,),	 'features.denseblock3.denselayer12.norm1.running_mean',	(608,),
 'block3.0.denselayer12.norm1.running_var',	(608,),	 'features.denseblock3.denselayer12.norm1.running_var',	(608,),
 'block3.0.denselayer12.conv1.weight',	(128, 608, 1, 1),	 'features.denseblock3.denselayer12.conv1.weight',	(128, 608, 1, 1),
 'block3.0.denselayer12.norm2.weight',	(128,),	 'features.denseblock3.denselayer12.norm2.weight',	(128,),
 'block3.0.denselayer12.norm2.bias',	(128,),	 'features.denseblock3.denselayer12.norm2.bias',	(128,),
 'block3.0.denselayer12.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer12.norm2.running_mean',	(128,),
 'block3.0.denselayer12.norm2.running_var',	(128,),	 'features.denseblock3.denselayer12.norm2.running_var',	(128,),
 'block3.0.denselayer12.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer12.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer13.norm1.weight',	(640,),	 'features.denseblock3.denselayer13.norm1.weight',	(640,),
 'block3.0.denselayer13.norm1.bias',	(640,),	 'features.denseblock3.denselayer13.norm1.bias',	(640,),
 'block3.0.denselayer13.norm1.running_mean',	(640,),	 'features.denseblock3.denselayer13.norm1.running_mean',	(640,),
 'block3.0.denselayer13.norm1.running_var',	(640,),	 'features.denseblock3.denselayer13.norm1.running_var',	(640,),
 'block3.0.denselayer13.conv1.weight',	(128, 640, 1, 1),	 'features.denseblock3.denselayer13.conv1.weight',	(128, 640, 1, 1),
 'block3.0.denselayer13.norm2.weight',	(128,),	 'features.denseblock3.denselayer13.norm2.weight',	(128,),
 'block3.0.denselayer13.norm2.bias',	(128,),	 'features.denseblock3.denselayer13.norm2.bias',	(128,),
 'block3.0.denselayer13.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer13.norm2.running_mean',	(128,),
 'block3.0.denselayer13.norm2.running_var',	(128,),	 'features.denseblock3.denselayer13.norm2.running_var',	(128,),
 'block3.0.denselayer13.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer13.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer14.norm1.weight',	(672,),	 'features.denseblock3.denselayer14.norm1.weight',	(672,),
 'block3.0.denselayer14.norm1.bias',	(672,),	 'features.denseblock3.denselayer14.norm1.bias',	(672,),
 'block3.0.denselayer14.norm1.running_mean',	(672,),	 'features.denseblock3.denselayer14.norm1.running_mean',	(672,),
 'block3.0.denselayer14.norm1.running_var',	(672,),	 'features.denseblock3.denselayer14.norm1.running_var',	(672,),
 'block3.0.denselayer14.conv1.weight',	(128, 672, 1, 1),	 'features.denseblock3.denselayer14.conv1.weight',	(128, 672, 1, 1),
 'block3.0.denselayer14.norm2.weight',	(128,),	 'features.denseblock3.denselayer14.norm2.weight',	(128,),
 'block3.0.denselayer14.norm2.bias',	(128,),	 'features.denseblock3.denselayer14.norm2.bias',	(128,),
 'block3.0.denselayer14.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer14.norm2.running_mean',	(128,),
 'block3.0.denselayer14.norm2.running_var',	(128,),	 'features.denseblock3.denselayer14.norm2.running_var',	(128,),
 'block3.0.denselayer14.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer14.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer15.norm1.weight',	(704,),	 'features.denseblock3.denselayer15.norm1.weight',	(704,),
 'block3.0.denselayer15.norm1.bias',	(704,),	 'features.denseblock3.denselayer15.norm1.bias',	(704,),
 'block3.0.denselayer15.norm1.running_mean',	(704,),	 'features.denseblock3.denselayer15.norm1.running_mean',	(704,),
 'block3.0.denselayer15.norm1.running_var',	(704,),	 'features.denseblock3.denselayer15.norm1.running_var',	(704,),
 'block3.0.denselayer15.conv1.weight',	(128, 704, 1, 1),	 'features.denseblock3.denselayer15.conv1.weight',	(128, 704, 1, 1),
 'block3.0.denselayer15.norm2.weight',	(128,),	 'features.denseblock3.denselayer15.norm2.weight',	(128,),
 'block3.0.denselayer15.norm2.bias',	(128,),	 'features.denseblock3.denselayer15.norm2.bias',	(128,),
 'block3.0.denselayer15.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer15.norm2.running_mean',	(128,),
 'block3.0.denselayer15.norm2.running_var',	(128,),	 'features.denseblock3.denselayer15.norm2.running_var',	(128,),
 'block3.0.denselayer15.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer15.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer16.norm1.weight',	(736,),	 'features.denseblock3.denselayer16.norm1.weight',	(736,),
 'block3.0.denselayer16.norm1.bias',	(736,),	 'features.denseblock3.denselayer16.norm1.bias',	(736,),
 'block3.0.denselayer16.norm1.running_mean',	(736,),	 'features.denseblock3.denselayer16.norm1.running_mean',	(736,),
 'block3.0.denselayer16.norm1.running_var',	(736,),	 'features.denseblock3.denselayer16.norm1.running_var',	(736,),
 'block3.0.denselayer16.conv1.weight',	(128, 736, 1, 1),	 'features.denseblock3.denselayer16.conv1.weight',	(128, 736, 1, 1),
 'block3.0.denselayer16.norm2.weight',	(128,),	 'features.denseblock3.denselayer16.norm2.weight',	(128,),
 'block3.0.denselayer16.norm2.bias',	(128,),	 'features.denseblock3.denselayer16.norm2.bias',	(128,),
 'block3.0.denselayer16.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer16.norm2.running_mean',	(128,),
 'block3.0.denselayer16.norm2.running_var',	(128,),	 'features.denseblock3.denselayer16.norm2.running_var',	(128,),
 'block3.0.denselayer16.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer16.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer17.norm1.weight',	(768,),	 'features.denseblock3.denselayer17.norm1.weight',	(768,),
 'block3.0.denselayer17.norm1.bias',	(768,),	 'features.denseblock3.denselayer17.norm1.bias',	(768,),
 'block3.0.denselayer17.norm1.running_mean',	(768,),	 'features.denseblock3.denselayer17.norm1.running_mean',	(768,),
 'block3.0.denselayer17.norm1.running_var',	(768,),	 'features.denseblock3.denselayer17.norm1.running_var',	(768,),
 'block3.0.denselayer17.conv1.weight',	(128, 768, 1, 1),	 'features.denseblock3.denselayer17.conv1.weight',	(128, 768, 1, 1),
 'block3.0.denselayer17.norm2.weight',	(128,),	 'features.denseblock3.denselayer17.norm2.weight',	(128,),
 'block3.0.denselayer17.norm2.bias',	(128,),	 'features.denseblock3.denselayer17.norm2.bias',	(128,),
 'block3.0.denselayer17.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer17.norm2.running_mean',	(128,),
 'block3.0.denselayer17.norm2.running_var',	(128,),	 'features.denseblock3.denselayer17.norm2.running_var',	(128,),
 'block3.0.denselayer17.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer17.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer18.norm1.weight',	(800,),	 'features.denseblock3.denselayer18.norm1.weight',	(800,),
 'block3.0.denselayer18.norm1.bias',	(800,),	 'features.denseblock3.denselayer18.norm1.bias',	(800,),
 'block3.0.denselayer18.norm1.running_mean',	(800,),	 'features.denseblock3.denselayer18.norm1.running_mean',	(800,),
 'block3.0.denselayer18.norm1.running_var',	(800,),	 'features.denseblock3.denselayer18.norm1.running_var',	(800,),
 'block3.0.denselayer18.conv1.weight',	(128, 800, 1, 1),	 'features.denseblock3.denselayer18.conv1.weight',	(128, 800, 1, 1),
 'block3.0.denselayer18.norm2.weight',	(128,),	 'features.denseblock3.denselayer18.norm2.weight',	(128,),
 'block3.0.denselayer18.norm2.bias',	(128,),	 'features.denseblock3.denselayer18.norm2.bias',	(128,),
 'block3.0.denselayer18.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer18.norm2.running_mean',	(128,),
 'block3.0.denselayer18.norm2.running_var',	(128,),	 'features.denseblock3.denselayer18.norm2.running_var',	(128,),
 'block3.0.denselayer18.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer18.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer19.norm1.weight',	(832,),	 'features.denseblock3.denselayer19.norm1.weight',	(832,),
 'block3.0.denselayer19.norm1.bias',	(832,),	 'features.denseblock3.denselayer19.norm1.bias',	(832,),
 'block3.0.denselayer19.norm1.running_mean',	(832,),	 'features.denseblock3.denselayer19.norm1.running_mean',	(832,),
 'block3.0.denselayer19.norm1.running_var',	(832,),	 'features.denseblock3.denselayer19.norm1.running_var',	(832,),
 'block3.0.denselayer19.conv1.weight',	(128, 832, 1, 1),	 'features.denseblock3.denselayer19.conv1.weight',	(128, 832, 1, 1),
 'block3.0.denselayer19.norm2.weight',	(128,),	 'features.denseblock3.denselayer19.norm2.weight',	(128,),
 'block3.0.denselayer19.norm2.bias',	(128,),	 'features.denseblock3.denselayer19.norm2.bias',	(128,),
 'block3.0.denselayer19.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer19.norm2.running_mean',	(128,),
 'block3.0.denselayer19.norm2.running_var',	(128,),	 'features.denseblock3.denselayer19.norm2.running_var',	(128,),
 'block3.0.denselayer19.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer19.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer20.norm1.weight',	(864,),	 'features.denseblock3.denselayer20.norm1.weight',	(864,),
 'block3.0.denselayer20.norm1.bias',	(864,),	 'features.denseblock3.denselayer20.norm1.bias',	(864,),
 'block3.0.denselayer20.norm1.running_mean',	(864,),	 'features.denseblock3.denselayer20.norm1.running_mean',	(864,),
 'block3.0.denselayer20.norm1.running_var',	(864,),	 'features.denseblock3.denselayer20.norm1.running_var',	(864,),
 'block3.0.denselayer20.conv1.weight',	(128, 864, 1, 1),	 'features.denseblock3.denselayer20.conv1.weight',	(128, 864, 1, 1),
 'block3.0.denselayer20.norm2.weight',	(128,),	 'features.denseblock3.denselayer20.norm2.weight',	(128,),
 'block3.0.denselayer20.norm2.bias',	(128,),	 'features.denseblock3.denselayer20.norm2.bias',	(128,),
 'block3.0.denselayer20.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer20.norm2.running_mean',	(128,),
 'block3.0.denselayer20.norm2.running_var',	(128,),	 'features.denseblock3.denselayer20.norm2.running_var',	(128,),
 'block3.0.denselayer20.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer20.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer21.norm1.weight',	(896,),	 'features.denseblock3.denselayer21.norm1.weight',	(896,),
 'block3.0.denselayer21.norm1.bias',	(896,),	 'features.denseblock3.denselayer21.norm1.bias',	(896,),
 'block3.0.denselayer21.norm1.running_mean',	(896,),	 'features.denseblock3.denselayer21.norm1.running_mean',	(896,),
 'block3.0.denselayer21.norm1.running_var',	(896,),	 'features.denseblock3.denselayer21.norm1.running_var',	(896,),
 'block3.0.denselayer21.conv1.weight',	(128, 896, 1, 1),	 'features.denseblock3.denselayer21.conv1.weight',	(128, 896, 1, 1),
 'block3.0.denselayer21.norm2.weight',	(128,),	 'features.denseblock3.denselayer21.norm2.weight',	(128,),
 'block3.0.denselayer21.norm2.bias',	(128,),	 'features.denseblock3.denselayer21.norm2.bias',	(128,),
 'block3.0.denselayer21.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer21.norm2.running_mean',	(128,),
 'block3.0.denselayer21.norm2.running_var',	(128,),	 'features.denseblock3.denselayer21.norm2.running_var',	(128,),
 'block3.0.denselayer21.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer21.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer22.norm1.weight',	(928,),	 'features.denseblock3.denselayer22.norm1.weight',	(928,),
 'block3.0.denselayer22.norm1.bias',	(928,),	 'features.denseblock3.denselayer22.norm1.bias',	(928,),
 'block3.0.denselayer22.norm1.running_mean',	(928,),	 'features.denseblock3.denselayer22.norm1.running_mean',	(928,),
 'block3.0.denselayer22.norm1.running_var',	(928,),	 'features.denseblock3.denselayer22.norm1.running_var',	(928,),
 'block3.0.denselayer22.conv1.weight',	(128, 928, 1, 1),	 'features.denseblock3.denselayer22.conv1.weight',	(128, 928, 1, 1),
 'block3.0.denselayer22.norm2.weight',	(128,),	 'features.denseblock3.denselayer22.norm2.weight',	(128,),
 'block3.0.denselayer22.norm2.bias',	(128,),	 'features.denseblock3.denselayer22.norm2.bias',	(128,),
 'block3.0.denselayer22.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer22.norm2.running_mean',	(128,),
 'block3.0.denselayer22.norm2.running_var',	(128,),	 'features.denseblock3.denselayer22.norm2.running_var',	(128,),
 'block3.0.denselayer22.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer22.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer23.norm1.weight',	(960,),	 'features.denseblock3.denselayer23.norm1.weight',	(960,),
 'block3.0.denselayer23.norm1.bias',	(960,),	 'features.denseblock3.denselayer23.norm1.bias',	(960,),
 'block3.0.denselayer23.norm1.running_mean',	(960,),	 'features.denseblock3.denselayer23.norm1.running_mean',	(960,),
 'block3.0.denselayer23.norm1.running_var',	(960,),	 'features.denseblock3.denselayer23.norm1.running_var',	(960,),
 'block3.0.denselayer23.conv1.weight',	(128, 960, 1, 1),	 'features.denseblock3.denselayer23.conv1.weight',	(128, 960, 1, 1),
 'block3.0.denselayer23.norm2.weight',	(128,),	 'features.denseblock3.denselayer23.norm2.weight',	(128,),
 'block3.0.denselayer23.norm2.bias',	(128,),	 'features.denseblock3.denselayer23.norm2.bias',	(128,),
 'block3.0.denselayer23.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer23.norm2.running_mean',	(128,),
 'block3.0.denselayer23.norm2.running_var',	(128,),	 'features.denseblock3.denselayer23.norm2.running_var',	(128,),
 'block3.0.denselayer23.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer23.conv2.weight',	(32, 128, 3, 3),
 'block3.0.denselayer24.norm1.weight',	(992,),	 'features.denseblock3.denselayer24.norm1.weight',	(992,),
 'block3.0.denselayer24.norm1.bias',	(992,),	 'features.denseblock3.denselayer24.norm1.bias',	(992,),
 'block3.0.denselayer24.norm1.running_mean',	(992,),	 'features.denseblock3.denselayer24.norm1.running_mean',	(992,),
 'block3.0.denselayer24.norm1.running_var',	(992,),	 'features.denseblock3.denselayer24.norm1.running_var',	(992,),
 'block3.0.denselayer24.conv1.weight',	(128, 992, 1, 1),	 'features.denseblock3.denselayer24.conv1.weight',	(128, 992, 1, 1),
 'block3.0.denselayer24.norm2.weight',	(128,),	 'features.denseblock3.denselayer24.norm2.weight',	(128,),
 'block3.0.denselayer24.norm2.bias',	(128,),	 'features.denseblock3.denselayer24.norm2.bias',	(128,),
 'block3.0.denselayer24.norm2.running_mean',	(128,),	 'features.denseblock3.denselayer24.norm2.running_mean',	(128,),
 'block3.0.denselayer24.norm2.running_var',	(128,),	 'features.denseblock3.denselayer24.norm2.running_var',	(128,),
 'block3.0.denselayer24.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock3.denselayer24.conv2.weight',	(32, 128, 3, 3),
 'block3.1.norm.weight',	(1024,),	 'features.transition3.norm.weight',	(1024,),
 'block3.1.norm.bias',	(1024,),	 'features.transition3.norm.bias',	(1024,),
 'block3.1.norm.running_mean',	(1024,),	 'features.transition3.norm.running_mean',	(1024,),
 'block3.1.norm.running_var',	(1024,),	 'features.transition3.norm.running_var',	(1024,),
 'block3.1.conv.weight',	(512, 1024, 1, 1),	 'features.transition3.conv.weight',	(512, 1024, 1, 1),
 'block4.0.denselayer1.norm1.weight',	(512,),	 'features.denseblock4.denselayer1.norm1.weight',	(512,),
 'block4.0.denselayer1.norm1.bias',	(512,),	 'features.denseblock4.denselayer1.norm1.bias',	(512,),
 'block4.0.denselayer1.norm1.running_mean',	(512,),	 'features.denseblock4.denselayer1.norm1.running_mean',	(512,),
 'block4.0.denselayer1.norm1.running_var',	(512,),	 'features.denseblock4.denselayer1.norm1.running_var',	(512,),
 'block4.0.denselayer1.conv1.weight',	(128, 512, 1, 1),	 'features.denseblock4.denselayer1.conv1.weight',	(128, 512, 1, 1),
 'block4.0.denselayer1.norm2.weight',	(128,),	 'features.denseblock4.denselayer1.norm2.weight',	(128,),
 'block4.0.denselayer1.norm2.bias',	(128,),	 'features.denseblock4.denselayer1.norm2.bias',	(128,),
 'block4.0.denselayer1.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer1.norm2.running_mean',	(128,),
 'block4.0.denselayer1.norm2.running_var',	(128,),	 'features.denseblock4.denselayer1.norm2.running_var',	(128,),
 'block4.0.denselayer1.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer1.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer2.norm1.weight',	(544,),	 'features.denseblock4.denselayer2.norm1.weight',	(544,),
 'block4.0.denselayer2.norm1.bias',	(544,),	 'features.denseblock4.denselayer2.norm1.bias',	(544,),
 'block4.0.denselayer2.norm1.running_mean',	(544,),	 'features.denseblock4.denselayer2.norm1.running_mean',	(544,),
 'block4.0.denselayer2.norm1.running_var',	(544,),	 'features.denseblock4.denselayer2.norm1.running_var',	(544,),
 'block4.0.denselayer2.conv1.weight',	(128, 544, 1, 1),	 'features.denseblock4.denselayer2.conv1.weight',	(128, 544, 1, 1),
 'block4.0.denselayer2.norm2.weight',	(128,),	 'features.denseblock4.denselayer2.norm2.weight',	(128,),
 'block4.0.denselayer2.norm2.bias',	(128,),	 'features.denseblock4.denselayer2.norm2.bias',	(128,),
 'block4.0.denselayer2.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer2.norm2.running_mean',	(128,),
 'block4.0.denselayer2.norm2.running_var',	(128,),	 'features.denseblock4.denselayer2.norm2.running_var',	(128,),
 'block4.0.denselayer2.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer2.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer3.norm1.weight',	(576,),	 'features.denseblock4.denselayer3.norm1.weight',	(576,),
 'block4.0.denselayer3.norm1.bias',	(576,),	 'features.denseblock4.denselayer3.norm1.bias',	(576,),
 'block4.0.denselayer3.norm1.running_mean',	(576,),	 'features.denseblock4.denselayer3.norm1.running_mean',	(576,),
 'block4.0.denselayer3.norm1.running_var',	(576,),	 'features.denseblock4.denselayer3.norm1.running_var',	(576,),
 'block4.0.denselayer3.conv1.weight',	(128, 576, 1, 1),	 'features.denseblock4.denselayer3.conv1.weight',	(128, 576, 1, 1),
 'block4.0.denselayer3.norm2.weight',	(128,),	 'features.denseblock4.denselayer3.norm2.weight',	(128,),
 'block4.0.denselayer3.norm2.bias',	(128,),	 'features.denseblock4.denselayer3.norm2.bias',	(128,),
 'block4.0.denselayer3.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer3.norm2.running_mean',	(128,),
 'block4.0.denselayer3.norm2.running_var',	(128,),	 'features.denseblock4.denselayer3.norm2.running_var',	(128,),
 'block4.0.denselayer3.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer3.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer4.norm1.weight',	(608,),	 'features.denseblock4.denselayer4.norm1.weight',	(608,),
 'block4.0.denselayer4.norm1.bias',	(608,),	 'features.denseblock4.denselayer4.norm1.bias',	(608,),
 'block4.0.denselayer4.norm1.running_mean',	(608,),	 'features.denseblock4.denselayer4.norm1.running_mean',	(608,),
 'block4.0.denselayer4.norm1.running_var',	(608,),	 'features.denseblock4.denselayer4.norm1.running_var',	(608,),
 'block4.0.denselayer4.conv1.weight',	(128, 608, 1, 1),	 'features.denseblock4.denselayer4.conv1.weight',	(128, 608, 1, 1),
 'block4.0.denselayer4.norm2.weight',	(128,),	 'features.denseblock4.denselayer4.norm2.weight',	(128,),
 'block4.0.denselayer4.norm2.bias',	(128,),	 'features.denseblock4.denselayer4.norm2.bias',	(128,),
 'block4.0.denselayer4.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer4.norm2.running_mean',	(128,),
 'block4.0.denselayer4.norm2.running_var',	(128,),	 'features.denseblock4.denselayer4.norm2.running_var',	(128,),
 'block4.0.denselayer4.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer4.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer5.norm1.weight',	(640,),	 'features.denseblock4.denselayer5.norm1.weight',	(640,),
 'block4.0.denselayer5.norm1.bias',	(640,),	 'features.denseblock4.denselayer5.norm1.bias',	(640,),
 'block4.0.denselayer5.norm1.running_mean',	(640,),	 'features.denseblock4.denselayer5.norm1.running_mean',	(640,),
 'block4.0.denselayer5.norm1.running_var',	(640,),	 'features.denseblock4.denselayer5.norm1.running_var',	(640,),
 'block4.0.denselayer5.conv1.weight',	(128, 640, 1, 1),	 'features.denseblock4.denselayer5.conv1.weight',	(128, 640, 1, 1),
 'block4.0.denselayer5.norm2.weight',	(128,),	 'features.denseblock4.denselayer5.norm2.weight',	(128,),
 'block4.0.denselayer5.norm2.bias',	(128,),	 'features.denseblock4.denselayer5.norm2.bias',	(128,),
 'block4.0.denselayer5.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer5.norm2.running_mean',	(128,),
 'block4.0.denselayer5.norm2.running_var',	(128,),	 'features.denseblock4.denselayer5.norm2.running_var',	(128,),
 'block4.0.denselayer5.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer5.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer6.norm1.weight',	(672,),	 'features.denseblock4.denselayer6.norm1.weight',	(672,),
 'block4.0.denselayer6.norm1.bias',	(672,),	 'features.denseblock4.denselayer6.norm1.bias',	(672,),
 'block4.0.denselayer6.norm1.running_mean',	(672,),	 'features.denseblock4.denselayer6.norm1.running_mean',	(672,),
 'block4.0.denselayer6.norm1.running_var',	(672,),	 'features.denseblock4.denselayer6.norm1.running_var',	(672,),
 'block4.0.denselayer6.conv1.weight',	(128, 672, 1, 1),	 'features.denseblock4.denselayer6.conv1.weight',	(128, 672, 1, 1),
 'block4.0.denselayer6.norm2.weight',	(128,),	 'features.denseblock4.denselayer6.norm2.weight',	(128,),
 'block4.0.denselayer6.norm2.bias',	(128,),	 'features.denseblock4.denselayer6.norm2.bias',	(128,),
 'block4.0.denselayer6.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer6.norm2.running_mean',	(128,),
 'block4.0.denselayer6.norm2.running_var',	(128,),	 'features.denseblock4.denselayer6.norm2.running_var',	(128,),
 'block4.0.denselayer6.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer6.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer7.norm1.weight',	(704,),	 'features.denseblock4.denselayer7.norm1.weight',	(704,),
 'block4.0.denselayer7.norm1.bias',	(704,),	 'features.denseblock4.denselayer7.norm1.bias',	(704,),
 'block4.0.denselayer7.norm1.running_mean',	(704,),	 'features.denseblock4.denselayer7.norm1.running_mean',	(704,),
 'block4.0.denselayer7.norm1.running_var',	(704,),	 'features.denseblock4.denselayer7.norm1.running_var',	(704,),
 'block4.0.denselayer7.conv1.weight',	(128, 704, 1, 1),	 'features.denseblock4.denselayer7.conv1.weight',	(128, 704, 1, 1),
 'block4.0.denselayer7.norm2.weight',	(128,),	 'features.denseblock4.denselayer7.norm2.weight',	(128,),
 'block4.0.denselayer7.norm2.bias',	(128,),	 'features.denseblock4.denselayer7.norm2.bias',	(128,),
 'block4.0.denselayer7.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer7.norm2.running_mean',	(128,),
 'block4.0.denselayer7.norm2.running_var',	(128,),	 'features.denseblock4.denselayer7.norm2.running_var',	(128,),
 'block4.0.denselayer7.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer7.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer8.norm1.weight',	(736,),	 'features.denseblock4.denselayer8.norm1.weight',	(736,),
 'block4.0.denselayer8.norm1.bias',	(736,),	 'features.denseblock4.denselayer8.norm1.bias',	(736,),
 'block4.0.denselayer8.norm1.running_mean',	(736,),	 'features.denseblock4.denselayer8.norm1.running_mean',	(736,),
 'block4.0.denselayer8.norm1.running_var',	(736,),	 'features.denseblock4.denselayer8.norm1.running_var',	(736,),
 'block4.0.denselayer8.conv1.weight',	(128, 736, 1, 1),	 'features.denseblock4.denselayer8.conv1.weight',	(128, 736, 1, 1),
 'block4.0.denselayer8.norm2.weight',	(128,),	 'features.denseblock4.denselayer8.norm2.weight',	(128,),
 'block4.0.denselayer8.norm2.bias',	(128,),	 'features.denseblock4.denselayer8.norm2.bias',	(128,),
 'block4.0.denselayer8.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer8.norm2.running_mean',	(128,),
 'block4.0.denselayer8.norm2.running_var',	(128,),	 'features.denseblock4.denselayer8.norm2.running_var',	(128,),
 'block4.0.denselayer8.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer8.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer9.norm1.weight',	(768,),	 'features.denseblock4.denselayer9.norm1.weight',	(768,),
 'block4.0.denselayer9.norm1.bias',	(768,),	 'features.denseblock4.denselayer9.norm1.bias',	(768,),
 'block4.0.denselayer9.norm1.running_mean',	(768,),	 'features.denseblock4.denselayer9.norm1.running_mean',	(768,),
 'block4.0.denselayer9.norm1.running_var',	(768,),	 'features.denseblock4.denselayer9.norm1.running_var',	(768,),
 'block4.0.denselayer9.conv1.weight',	(128, 768, 1, 1),	 'features.denseblock4.denselayer9.conv1.weight',	(128, 768, 1, 1),
 'block4.0.denselayer9.norm2.weight',	(128,),	 'features.denseblock4.denselayer9.norm2.weight',	(128,),
 'block4.0.denselayer9.norm2.bias',	(128,),	 'features.denseblock4.denselayer9.norm2.bias',	(128,),
 'block4.0.denselayer9.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer9.norm2.running_mean',	(128,),
 'block4.0.denselayer9.norm2.running_var',	(128,),	 'features.denseblock4.denselayer9.norm2.running_var',	(128,),
 'block4.0.denselayer9.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer9.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer10.norm1.weight',	(800,),	 'features.denseblock4.denselayer10.norm1.weight',	(800,),
 'block4.0.denselayer10.norm1.bias',	(800,),	 'features.denseblock4.denselayer10.norm1.bias',	(800,),
 'block4.0.denselayer10.norm1.running_mean',	(800,),	 'features.denseblock4.denselayer10.norm1.running_mean',	(800,),
 'block4.0.denselayer10.norm1.running_var',	(800,),	 'features.denseblock4.denselayer10.norm1.running_var',	(800,),
 'block4.0.denselayer10.conv1.weight',	(128, 800, 1, 1),	 'features.denseblock4.denselayer10.conv1.weight',	(128, 800, 1, 1),
 'block4.0.denselayer10.norm2.weight',	(128,),	 'features.denseblock4.denselayer10.norm2.weight',	(128,),
 'block4.0.denselayer10.norm2.bias',	(128,),	 'features.denseblock4.denselayer10.norm2.bias',	(128,),
 'block4.0.denselayer10.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer10.norm2.running_mean',	(128,),
 'block4.0.denselayer10.norm2.running_var',	(128,),	 'features.denseblock4.denselayer10.norm2.running_var',	(128,),
 'block4.0.denselayer10.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer10.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer11.norm1.weight',	(832,),	 'features.denseblock4.denselayer11.norm1.weight',	(832,),
 'block4.0.denselayer11.norm1.bias',	(832,),	 'features.denseblock4.denselayer11.norm1.bias',	(832,),
 'block4.0.denselayer11.norm1.running_mean',	(832,),	 'features.denseblock4.denselayer11.norm1.running_mean',	(832,),
 'block4.0.denselayer11.norm1.running_var',	(832,),	 'features.denseblock4.denselayer11.norm1.running_var',	(832,),
 'block4.0.denselayer11.conv1.weight',	(128, 832, 1, 1),	 'features.denseblock4.denselayer11.conv1.weight',	(128, 832, 1, 1),
 'block4.0.denselayer11.norm2.weight',	(128,),	 'features.denseblock4.denselayer11.norm2.weight',	(128,),
 'block4.0.denselayer11.norm2.bias',	(128,),	 'features.denseblock4.denselayer11.norm2.bias',	(128,),
 'block4.0.denselayer11.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer11.norm2.running_mean',	(128,),
 'block4.0.denselayer11.norm2.running_var',	(128,),	 'features.denseblock4.denselayer11.norm2.running_var',	(128,),
 'block4.0.denselayer11.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer11.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer12.norm1.weight',	(864,),	 'features.denseblock4.denselayer12.norm1.weight',	(864,),
 'block4.0.denselayer12.norm1.bias',	(864,),	 'features.denseblock4.denselayer12.norm1.bias',	(864,),
 'block4.0.denselayer12.norm1.running_mean',	(864,),	 'features.denseblock4.denselayer12.norm1.running_mean',	(864,),
 'block4.0.denselayer12.norm1.running_var',	(864,),	 'features.denseblock4.denselayer12.norm1.running_var',	(864,),
 'block4.0.denselayer12.conv1.weight',	(128, 864, 1, 1),	 'features.denseblock4.denselayer12.conv1.weight',	(128, 864, 1, 1),
 'block4.0.denselayer12.norm2.weight',	(128,),	 'features.denseblock4.denselayer12.norm2.weight',	(128,),
 'block4.0.denselayer12.norm2.bias',	(128,),	 'features.denseblock4.denselayer12.norm2.bias',	(128,),
 'block4.0.denselayer12.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer12.norm2.running_mean',	(128,),
 'block4.0.denselayer12.norm2.running_var',	(128,),	 'features.denseblock4.denselayer12.norm2.running_var',	(128,),
 'block4.0.denselayer12.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer12.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer13.norm1.weight',	(896,),	 'features.denseblock4.denselayer13.norm1.weight',	(896,),
 'block4.0.denselayer13.norm1.bias',	(896,),	 'features.denseblock4.denselayer13.norm1.bias',	(896,),
 'block4.0.denselayer13.norm1.running_mean',	(896,),	 'features.denseblock4.denselayer13.norm1.running_mean',	(896,),
 'block4.0.denselayer13.norm1.running_var',	(896,),	 'features.denseblock4.denselayer13.norm1.running_var',	(896,),
 'block4.0.denselayer13.conv1.weight',	(128, 896, 1, 1),	 'features.denseblock4.denselayer13.conv1.weight',	(128, 896, 1, 1),
 'block4.0.denselayer13.norm2.weight',	(128,),	 'features.denseblock4.denselayer13.norm2.weight',	(128,),
 'block4.0.denselayer13.norm2.bias',	(128,),	 'features.denseblock4.denselayer13.norm2.bias',	(128,),
 'block4.0.denselayer13.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer13.norm2.running_mean',	(128,),
 'block4.0.denselayer13.norm2.running_var',	(128,),	 'features.denseblock4.denselayer13.norm2.running_var',	(128,),
 'block4.0.denselayer13.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer13.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer14.norm1.weight',	(928,),	 'features.denseblock4.denselayer14.norm1.weight',	(928,),
 'block4.0.denselayer14.norm1.bias',	(928,),	 'features.denseblock4.denselayer14.norm1.bias',	(928,),
 'block4.0.denselayer14.norm1.running_mean',	(928,),	 'features.denseblock4.denselayer14.norm1.running_mean',	(928,),
 'block4.0.denselayer14.norm1.running_var',	(928,),	 'features.denseblock4.denselayer14.norm1.running_var',	(928,),
 'block4.0.denselayer14.conv1.weight',	(128, 928, 1, 1),	 'features.denseblock4.denselayer14.conv1.weight',	(128, 928, 1, 1),
 'block4.0.denselayer14.norm2.weight',	(128,),	 'features.denseblock4.denselayer14.norm2.weight',	(128,),
 'block4.0.denselayer14.norm2.bias',	(128,),	 'features.denseblock4.denselayer14.norm2.bias',	(128,),
 'block4.0.denselayer14.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer14.norm2.running_mean',	(128,),
 'block4.0.denselayer14.norm2.running_var',	(128,),	 'features.denseblock4.denselayer14.norm2.running_var',	(128,),
 'block4.0.denselayer14.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer14.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer15.norm1.weight',	(960,),	 'features.denseblock4.denselayer15.norm1.weight',	(960,),
 'block4.0.denselayer15.norm1.bias',	(960,),	 'features.denseblock4.denselayer15.norm1.bias',	(960,),
 'block4.0.denselayer15.norm1.running_mean',	(960,),	 'features.denseblock4.denselayer15.norm1.running_mean',	(960,),
 'block4.0.denselayer15.norm1.running_var',	(960,),	 'features.denseblock4.denselayer15.norm1.running_var',	(960,),
 'block4.0.denselayer15.conv1.weight',	(128, 960, 1, 1),	 'features.denseblock4.denselayer15.conv1.weight',	(128, 960, 1, 1),
 'block4.0.denselayer15.norm2.weight',	(128,),	 'features.denseblock4.denselayer15.norm2.weight',	(128,),
 'block4.0.denselayer15.norm2.bias',	(128,),	 'features.denseblock4.denselayer15.norm2.bias',	(128,),
 'block4.0.denselayer15.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer15.norm2.running_mean',	(128,),
 'block4.0.denselayer15.norm2.running_var',	(128,),	 'features.denseblock4.denselayer15.norm2.running_var',	(128,),
 'block4.0.denselayer15.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer15.conv2.weight',	(32, 128, 3, 3),
 'block4.0.denselayer16.norm1.weight',	(992,),	 'features.denseblock4.denselayer16.norm1.weight',	(992,),
 'block4.0.denselayer16.norm1.bias',	(992,),	 'features.denseblock4.denselayer16.norm1.bias',	(992,),
 'block4.0.denselayer16.norm1.running_mean',	(992,),	 'features.denseblock4.denselayer16.norm1.running_mean',	(992,),
 'block4.0.denselayer16.norm1.running_var',	(992,),	 'features.denseblock4.denselayer16.norm1.running_var',	(992,),
 'block4.0.denselayer16.conv1.weight',	(128, 992, 1, 1),	 'features.denseblock4.denselayer16.conv1.weight',	(128, 992, 1, 1),
 'block4.0.denselayer16.norm2.weight',	(128,),	 'features.denseblock4.denselayer16.norm2.weight',	(128,),
 'block4.0.denselayer16.norm2.bias',	(128,),	 'features.denseblock4.denselayer16.norm2.bias',	(128,),
 'block4.0.denselayer16.norm2.running_mean',	(128,),	 'features.denseblock4.denselayer16.norm2.running_mean',	(128,),
 'block4.0.denselayer16.norm2.running_var',	(128,),	 'features.denseblock4.denselayer16.norm2.running_var',	(128,),
 'block4.0.denselayer16.conv2.weight',	(32, 128, 3, 3),	 'features.denseblock4.denselayer16.conv2.weight',	(32, 128, 3, 3),
 'block4.1.weight',	(1024,),	 'features.norm5.weight',	(1024,),
 'block4.1.bias',	(1024,),	 'features.norm5.bias',	(1024,),
 'block4.1.running_mean',	(1024,),	 'features.norm5.running_mean',	(1024,),
 'block4.1.running_var',	(1024,),	 'features.norm5.running_var',	(1024,),
 'logit.weight',	(1000, 1024),	 'classifier.weight',	(1000, 1024),
 'logit.bias',	(1000,),	 'classifier.bias',	(1000,),

]

def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    def recompile_pretrain_state_dict(pretrain_state_dict):
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        for key in list(pretrain_state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                pretrain_state_dict[new_key] = pretrain_state_dict[key]
                del pretrain_state_dict[key]

        return pretrain_state_dict

    #-----

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    pretrain_state_dict = recompile_pretrain_state_dict(pretrain_state_dict)

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

    #---
    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')


class RGB(nn.Module):
    def __init__(self,):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

###############################################################################

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - set to True to use checkpointing. Much more memory efficient,
          but slower. Default: *False*
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out






#--------------
class DenseNet121(nn.Module):
    def __init__(self, num_class=1000 ):
        super(DenseNet121, self).__init__()
        self.rgb = RGB()

        #model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
        e = DenseNet(32, (6, 12, 24, 16), 64)
        self.block0  = nn.Sequential(
            e.features.conv0,
            e.features.norm0,
            e.features.relu0,
            e.features.pool0,
        )
        #self.block0[0].bias.data.fill_(0.0)
        self.block1  = nn.Sequential(
            e.features.denseblock1,
            e.features.transition1,
        )
        self.block2  = nn.Sequential(
            e.features.denseblock2,
            e.features.transition2,
        )
        self.block3  = nn.Sequential(
            e.features.denseblock3,
            e.features.transition3,
        )
        self.block4  = nn.Sequential(
            e.features.denseblock4,
            e.features.norm5,
            nn.ReLU(inplace=True),
        )

        self.logit = e.classifier
        e=None


    def forward(self, x):
        batch_size = len(x)
        x = self.rgb(x)

        x = self.block0(x)
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

    net = DenseNet121()
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

        net = net.cuda().eval()

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #image = cv2.resize(image,dsize=(224,224))
            #image = image[16:16+224,16:16+224]

            image = image[:,:,::-1]
            image = image.astype(np.float32)/255
            #image = (image -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
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
