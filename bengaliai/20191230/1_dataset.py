from common import *
from kaggle import *



TRAIN_NUM = 200840
TRAIN_PARQUET = None

DATA_DIR = '../'
TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]
NUM_TASK = len(TASK_NAME)


CLASS_MAP = pd.read_csv(DATA_DIR + '/class_map.csv')
CLASS_MAP=[
    CLASS_MAP[:168]['component'].values,
    CLASS_MAP[168:179]['component'].values,
    CLASS_MAP[179:]['component'].values,
]

class KaggleDataset(Dataset):
    def __init__(self, split, mode, csv, parquet, augment=None):
        global TRAIN_PARQUET

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.parquet  = parquet
        self.augment = augment

        #'image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'
        df = pd.read_csv(DATA_DIR + '/%s'%csv) #.fillna('')

        if parquet is not None:
            uid   = []
            image = []
            for f in parquet:
                d = pd.read_parquet(DATA_DIR + '/%s'%f , engine='pyarrow')
                uid.append(d['image_id'].values)
                image.append(d.drop('image_id', axis=1).values.astype(np.uint8))
            uid   = np.concatenate(uid)
            image = np.concatenate(image)

        else:
            if 0==0:
                if TRAIN_PARQUET is None:
                    d = pd.concat(
                        pd.read_parquet(DATA_DIR + 'train/%s'%f , engine='pyarrow')
                        for f in ['train_image_data_0.parquet',
                                  'train_image_data_1.parquet',
                                  'train_image_data_2.parquet',
                                  'train_image_data_3.parquet',]
                    )
                    TRAIN_PARQUET = d
                else:
                    d = TRAIN_PARQUET
                print(sys.getsizeof(d)) # 6508912474 bytes, 6 GB

                uid   = d['image_id'].values
                image = d.drop('image_id', axis=1).values.astype(np.uint8)
                np.save('uid.npy',uid)
                np.save('image.npy',image)

            TRAIN_PARQUET = None
            if 1: #use this !!! faster load
                if TRAIN_PARQUET is None:
                    uid   = np.load('uid.npy',allow_pickle=True)
                    image = np.load('image.npy',allow_pickle=True)
                    TRAIN_PARQUET = (uid, image)
                else:
                    uid, image = TRAIN_PARQUET


        #---
        if split is not None:
            uid = np.load('data/split/%s'%split, allow_pickle=True)
            df  = df_loc_by_list(df, 'image_id', uid)


        index = df.index.values
        self.df = df.set_index('image_id', drop = True)
        self.uid = uid
        self.image = image[index]
        self.num_image = len(uid)

        assert np.all(uid == df['image_id'].values)


    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tmode     = %s\n'%self.mode
        string += '\tsplit    = %s\n'%self.split
        string += '\tcsv      = %s\n'%str(self.csv)
        string += '\tparquet  = %s\n'%self.parquet
        string += '\tnum_image = %d\n'%self.num_image

        return string


    def __len__(self):
        return self.num_image


    def __getitem__(self, index):
        # print(index)
        image_id = self.uid[index]
        grapheme_root, vowel_diacritic, consonant_diacritic, grapheme  =  self.df.loc[image_id].values

        image = self.image[index].reshape(137, 236)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image = image.astype(np.float32)/255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic]

        infor = Struct(
            index    = index,
            image_id = image_id,
            grapheme = grapheme,
        )

        if self.augment is None:
            return image, label, infor
        else:
            return self.augment(image, label, infor)




def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    input = np.stack(input)
    #input = input[...,::-1].copy()
    input = input.transpose(0,3,1,2)

    label = np.stack(label)

    #----
    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(label).long()
    truth0, truth1, truth2 = truth[:,0],truth[:,1],truth[:,2]
    truth = [truth0, truth1, truth2]
    return input, truth, infor


##############################################################

def tensor_to_image(tensor):
    image = tensor.data.cpu().numpy()
    image = image.transpose(0,2,3,1)
    #image = image[...,::-1]
    return image


##############################################################

def do_random_crop_rotate_rescale(
    image,
    mode={'rotate': 10,'scale': 0.1,'shift': 0.1}
):

    dangle = 0
    dscale_x, dscale_y = 0,0
    dshift_x, dshift_y = 0,0

    for k,v in mode.items():
        if   'rotate'== k:
            dangle = np.random.uniform(-v, v)
        elif 'scale' == k:
            dscale_x, dscale_y = np.random.uniform(-1, 1, 2)*v
        elif 'shift' == k:
            dshift_x, dshift_y = np.random.uniform(-1, 1, 2)*v
        else:
            raise NotImplementedError

    #----

    height, width = image.shape[:2]

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift_x*width, dshift_y*height

    src = np.array([[-width/2,-height/2],[ width/2,-height/2],[ width/2, height/2],[-width/2, height/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+width/2 +tx
    y = (src*[sin, cos]).sum(1)+height/2+ty
    src = np.column_stack([x,y])

    dst = np.array([[0,0],[width,0],[width,height],[0,height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)
    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image


def do_random_log_contast(image, gain=[0.70, 1.30] ):
    gain = np.random.uniform(gain[0],gain[1],1)
    inverse = np.random.choice(2,1)

    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image,0,1)
    return image


#https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py
def do_grid_distortion(image, distort=0.25, num_step = 10):

    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]

    #---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        start = x
        end   = x + step_x
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + step_x * distort_x[i]
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur


    yy = np.zeros(height, np.float32)
    step_y = height // num_step

    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        start = y
        end = y + step_y
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + step_y * distort_y[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image



# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contast(image, alpha=[0,1]):
    beta  = 0
    alpha = random.uniform(*alpha) + 1
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image,0,1)
    return image



##############################################################

def run_check_dataset():

    dump_label = []
    dump_infor = []
    if 0:
        dataset = KaggleDataset(
            mode    = 'train',
            csv     = 'train.csv',
            #split   = 'random1/valid_small_a_fold0_1000.npy',
            split   = 'random0/valid_small_a_fold0_1000.npy',
            parquet = ['train_image_data_0.parquet',
                       #'train_image_data_1.parquet',
                       #'train_image_data_2.parquet',
                       #'train_image_data_3.parquet',
                       ],
            augment = None,
        )


    if 0:
        dataset = KaggleDataset(
            mode    = 'train',
            csv     = 'train.csv',
            split   = 'random1/valid_small_a_fold0_1000.npy',
            parquet = None,
            augment = None,
        )


    if 1:
        dataset = KaggleDataset(
            mode    = 'train',
            csv     = 'debug_image_data_0.csv',
            split   = None,
            parquet = ['debug_image_data_0.parquet',],
            augment = None,
        )

    print(dataset)
    for n in range(0,128):
    #for n in range(0,len(dataset)):
        i = n #i = np.random.choice(len(dataset))

        image, label, infor = dataset[i]
        #overlay = draw_truth(image, label, infor, threshold=0.3)

        overlay = (image*255).astype(np.uint8)
        #----
        print('%05d :\n%s'%(i, str(infor)))
        print('label = %s'%str(label))
        image_show('overlay',overlay)
        cv2.waitKey(0)

        if 0:
            dump_dir = '/root/share/project/kaggle/2020/grapheme_classification/data/image/debug'
            cv2.imwrite(dump_dir+'/%s.png'%infor.image_id,overlay)
            dump_label.append(label)
            dump_infor.append(infor)
    if 0:
        write_pickle_to_file(dump_dir+'.label.pickle', dump_label)
        write_pickle_to_file(dump_dir+'.infor.pickle', dump_infor)






def run_check_dataloader():

    dataset = KaggleDataset(
        mode    = 'train',
        csv     = 'train.csv',
        #split   = 'random1/valid_small_a_fold0_1000.npy',
        split   = 'random0/valid_small_a_fold0_1000.npy',
        parquet = ['train_image_data_0.parquet', ],
        augment = None,
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 5,
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
        print('truth[0]: ',truth[0].shape)
        print('truth[1]: ',truth[1].shape)
        print('truth[2]: ',truth[2].shape)
        print('')

        if 1:
            batch_size = len(infor)

            image = tensor_to_image(input)
            truth = [t.data.cpu().numpy() for t in truth]

            for b in range(batch_size):
                print('%2d --------------------------- '%(b))
                print('%s'%(str(infor[b])))

                for i,name in enumerate(TASK_NAME):
                    t   = truth[i][b]
                    print('%28s  |  truth : %3d '%(name,t))
                print('')

                overlay = (image[b]*255).astype(np.uint8)
                image_show('image',overlay)
                cv2.waitKey(0)



def run_check_augment():
    def augment(image, label, infor):

        if 0:
            image = do_grid_distortion(image, distort=0.25, num_step = 10)
        if 0:
            image = do_random_crop_rotate_rescale(image, mode={'rotate': 20,'scale': 0.2,'shift': 0.08})
        if 1:
            image = do_random_contast(image, alpha=[0,1])
        return image, label, infor



    def rand_augment(image, mask, infor):
        ops=np.random.choice([
            lambda image : image,
            lambda image : do_random_crop_rotate_rescale(image, mode={'rotate': 20,'scale': 0.2,'shift': 0.08}),
            lambda image : do_grid_distortion(image, distort=0.25, num_step = 10),
        ],3)
        for op in ops:
            image = op(image)
        return image, label, infor


    dataset = KaggleDataset(
        mode    = 'train',
        csv     = 'debug_image_data_0.csv',
        split   = None,
        parquet = ['debug_image_data_0.parquet',],
        augment = None,
    )
    print(dataset)


    for b in range(len(dataset)):
        image, label, infor = dataset[b]

        print('%2d --------------------------- '%(b))
        print('%s'%(str(infor)))
        print('%s'%(str(label)))
        print('')

        overlay = (image*255).astype(np.uint8)
        image_show('image',overlay)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, label1, infor1 =  augment(image.copy(), label.copy(), infor)
                #image1, label1, infor1 =  rand_augment(image.copy(), label.copy(), infor)

                overlay1 = (image1*255).astype(np.uint8)
                image_show('image1',overlay1)
                cv2.waitKey(0)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_dataset()
    #run_check_dataloader()
    run_check_augment()





