from common import *
from kaggle import *


DATA_DIR = '/root/share/project/kaggle/2019/steel/data'



class SteelDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f) for f in csv])
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df

    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()

        length = len(self)
        num = len(self)*4
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        #---

        string  = ''
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\t\tlen   = %5d\n'%len(self)
        if self.mode == 'train':
            string += '\t\tnum   = %5d\n'%num
            string += '\t\tneg   = %5d  %0.3f\n'%(neg,neg/num)
            string += '\t\tpos   = %5d  %0.3f\n'%(pos,pos/num)
            string += '\t\tpos1  = %5d  %0.3f  %0.3f\n'%(pos1,pos1/length,pos1/pos)
            string += '\t\tpos2  = %5d  %0.3f  %0.3f\n'%(pos2,pos2/length,pos2/pos)
            string += '\t\tpos3  = %5d  %0.3f  %0.3f\n'%(pos3,pos3/length,pos3/pos)
            string += '\t\tpos4  = %5d  %0.3f  %0.3f\n'%(pos4,pos4/length,pos4/pos)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)

'''
test_dataset : 
	mode    = train
	split   = ['valid0_500.npy']
	csv     = ['train.csv']
		len   =   500
		neg   =   212  0.424
		pos   =   288  0.576
		pos1  =    35  0.070  0.122
		pos2  =     5  0.010  0.017
		pos3  =   213  0.426  0.740
		pos4  =    35  0.070  0.122
		

train_dataset : 
	mode    = train
	split   = ['train0_12068.npy']
	csv     = ['train.csv']
		len   = 12068
		neg   =  5261  0.436
		pos   =  6807  0.564
		pos1  =   862  0.071  0.127
		pos2  =   242  0.020  0.036
		pos3  =  4937  0.409  0.725
		pos4  =   766  0.063  0.113

		
'''

def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        infor.append(batch[b][2])

    input = np.stack(input).astype(np.float32)/255
    input = input.transpose(0,3,1,2)
    truth = np.stack(truth)
    truth = (truth>0.5).astype(np.float32)

    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(truth).float()

    return input, truth, infor


class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4*num_neg


    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length

class FixedSampler(Sampler):

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index   = index
        self.length  = len(index)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return self.length


##############################################################
#
# class BalanceClassSampler(Sampler):
#
#     def __init__(self, dataset, length=None):
#         self.dataset = dataset
#
#         if length is None:
#             length = len(self.dataset)
#
#         self.length = length
#
#
#
#     def __iter__(self):
#
#         df = self.dataset.df
#         df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
#         df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
#
#
#         label = df['Label'].values*df['Class'].values
#         unique, count = np.unique(label, return_counts=True)
#         L = len(label)//5
#
#
#
#
#         pos_index = np.where(self.dataset.label==1)[0]
#         neg_index = np.where(self.dataset.label==0)[0]
#         half = self.length//2 + 1
#
#
#         neg  = np.random.choice(label==0, [L,6], replace=True)
#         pos1 = np.random.choice(label==1, L, replace=True)
#         pos2 = np.random.choice(label==2, L, replace=True)
#         pos3 = np.random.choice(label==3, L, replace=True)
#         pos4 = np.random.choice(label==4, L, replace=True)
#
#
#         l = np.stack([neg.reshape,pos1,pos2,pos3,pos3,pos4]).T
#         l = l.reshape(-1)
#         l = l[:self.length]
#         return iter(l)
#
#     def __len__(self):
#         return self.length


##############################################################

def image_to_input(image,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-rbg_mean[0])/rbg_std[0]
    input[:,1] = (input[:,1]-rbg_mean[1])/rbg_std[1]
    input[:,2] = (input[:,2]-rbg_mean[2])/rbg_std[2]
    return input


def input_to_image(input,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:,0] = (input[:,0]*rbg_std[0]+rbg_mean[0])
    input[:,1] = (input[:,1]*rbg_std[1]+rbg_mean[1])
    input[:,2] = (input[:,2]*rbg_std[2]+rbg_mean[2])
    input = input.transpose(0,2,3,1)
    input = input[...,::-1]
    image = (input*255).astype(np.uint8)
    return image



##############################################################

def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]
    return image, mask

def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

        mask = mask.transpose(1,2,0)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2,0,1)

    return image, mask




def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = mask[:,:,::-1]
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = mask[:,::-1,:]
    return image, mask




def do_random_scale_rotate(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-5, 5)
    dscale = np.random.uniform(-0.15,0.15,2)
    dshift = np.random.uniform(0,1,2)
    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale #1,1 #
    tx,ty = dshift

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)
    y = (src*[sin, cos]).sum(1)
    x = x-x.min()
    y = y-y.min()
    x = x + (W-x.max())*tx
    y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = mask.transpose(1,2,0)
    mask = cv2.warpPerspective( mask, transform, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = mask.transpose(2,0,1)
    mask = (mask>0.5).astype(np.float32)

    return image, mask


def do_random_crop_rotate_rescale(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1,0.1,2)

    dscale_x = np.random.uniform(-0.00075,0.00075)
    dscale_y = np.random.uniform(-0.25,0.25)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift*min(H,W)

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+W/2
    y = (src*[sin, cos]).sum(1)+H/2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = mask.transpose(1,2,0)
    mask = cv2.warpPerspective( mask, transform, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = mask.transpose(2,0,1)


    return image, mask
def do_random_log_contast(image):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image


def do_noise(image, mask, noise=8):
    H,W = image.shape[:2]
    image = image + np.random.uniform(-1,1,(H,W,1))*noise
    image = np.clip(image,0,255).astype(np.uint8)
    return image, mask


##############################################################

def run_check_train_dataset():

    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train0_12068.npy',],
        augment = None, #
    )
    print(dataset)
    #exit(0)

    for n in range(0,len(dataset)):
        i = n #i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        overlay = np.vstack([m for m in mask])

        #----
        print('%05d : %s'%(i, infor.image_id))
        image_show('image',image,0.5)
        image_show_norm('mask',overlay,0,1,0.5)
        cv2.waitKey(0)


def run_check_test_dataset():

    dataset = SteelDataset(
        mode    = 'test',
        csv     = ['sample_submission.csv',],
        split   = ['test_1801.npy',],
        augment = None, #
    )
    print(dataset)
    #exit(0)

    for n in range(0,len(dataset)):
        i = n #i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        overlay = np.vstack([m for m in mask])

        #----
        print('%05d : %s'%(i, infor.image_id))
        image_show('image',image,0.5)
        image_show_norm('mask',overlay,0,1,0.5)
        cv2.waitKey(0)


def run_check_data_loader():

    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train0_12068.npy',],
        augment = None, #
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = BalanceClassSampler(dataset),
        #sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 32,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    for t,(input, truth, infor) in enumerate(loader):

        print('----t=%d---'%t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth', truth.shape)
        print('')

        if 1:
            batch_size= len(infor)
            input = input.data.cpu().numpy()
            input = (input*255).astype(np.uint8)
            input = input.transpose(0,2,3,1)
            #input = 255-(input*255).astype(np.uint8)

            truth = truth.data.cpu().numpy()
            for b in range(batch_size):
                print(infor[b].image_id)

                image = input[b]
                mask  = truth[b]
                overlay = np.vstack([m for m in mask])

                image_show('image',image,0.5)
                image_show_norm('mask',overlay,0,1,0.5)
                cv2.waitKey(0)



def run_check_augment():

    def augment(image, mask, infor):
        #image, mask = do_random_scale_rotate(image, mask)
        #image = do_random_log_contast(image)

        # if np.random.rand()<0.5:
        #     image, mask = do_flip_ud(image, mask)



        #image, mask = do_noise(image, mask, noise=8)
        #image, mask = do_random_crop_rescale(image,mask,1600-(256-224),224)
        image, mask = do_random_crop_rotate_rescale(image,mask,1600-(256-224),224)


        #image, mask = do_random_scale_rotate(image, mask, 224*2, 224)
        return image, mask, infor


    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train0_12068.npy',],
        augment = None,  #None
    )
    print(dataset)


    for t in range(len(dataset)):
        image, mask, infor = dataset[t]

        overlay = image.copy()
        overlay = draw_contour_overlay(overlay, mask[0], (0,0,255), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[1], (0,255,0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[2], (255,0,0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[3], (0,255,255), thickness=2)

        print('----t=%d---'%t)
        print('')
        print('infor\n',infor)
        print(image.shape)
        print(mask.shape)
        print('')


        #image_show('original_mask',mask,  resize=0.25)
        image_show('original_image',image,resize=0.5)
        image_show('original_overlay',overlay,resize=0.5)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, mask1, infor1 =  augment(image.copy(), mask.copy(), infor)

                overlay1 = image1.copy()
                overlay1 = draw_contour_overlay(overlay1, mask1[0], (0,0,255), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[1], (0,255,0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[2], (255,0,0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[3], (0,255,255), thickness=2)

                #image_show_norm('mask',mask1,  resize=0.25)
                image_show('image1',image1,resize=0.5)
                image_show('overlay1',overlay1,resize=0.5)
                cv2.waitKey(1)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_train_dataset()
    #run_check_test_dataset()

    #run_check_data_loader()
    run_check_augment()



