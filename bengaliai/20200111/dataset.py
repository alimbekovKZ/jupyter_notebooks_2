from common import *
from kaggle import *

#--------------
DATA_DIR  = '/root/share/project/kaggle/2020/grapheme_classification/data'
IMAGE_HEIGHT, IMAGE_WIDTH = 137, 236


TASK = {
   'grapheme_root': Struct(
       num_class = 168,
   ),
   'vowel_diacritic': Struct(
       num_class = 11,
   ),
   'consonant_diacritic': Struct(
       num_class = 7,
   ),
   'grapheme': Struct(
       num_class = 1295,
       class_map = dict(pd.read_csv(DATA_DIR + '/grapheme_1295.csv')[['grapheme','label']].values),
       #freqency  = None,
   ),
}
NUM_TASK  = len(TASK)
NUM_CLASS = [TASK[k].num_class for k in ['grapheme_root','vowel_diacritic','consonant_diacritic','grapheme']]


#---
TRAIN_NUM     = 200840
TRAIN_PARQUET = None


class KaggleDataset(Dataset):
    def __init__(self, split, mode, csv, parquet, augment=None):
        global TRAIN_PARQUET

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.parquet  = parquet
        self.augment = augment

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
            if 0:
                if TRAIN_PARQUET is None:
                    d = pd.concat(
                        pd.read_parquet(DATA_DIR + '/%s'%f , engine='pyarrow')
                        for f in ['train_image_data_0.parquet',
                                  'train_image_data_1.parquet',
                                  'train_image_data_2.parquet',
                                  'train_image_data_3.parquet',]
                    )
                    TRAIN_PARQUET = d
                else:
                    d = TRAIN_PARQUET
                #print(sys.getsizeof(d)) # 6508912474 bytes, 6 GB

                uid   = d['image_id'].values
                image = d.drop('image_id', axis=1).values.astype(np.uint8)
                #np.save('/media/ssd/data/kaggle/grapheme_classification/uid.npy',uid)
                #np.save('/media/ssd/data/kaggle/grapheme_classification/image.npy',image)

            if 1:
                if TRAIN_PARQUET is None:
                    uid   = np.load('/media/ssd/data/kaggle/grapheme_classification/uid.npy',allow_pickle=True)
                    image = np.load('/media/ssd/data/kaggle/grapheme_classification/image.npy',allow_pickle=True)
                    TRAIN_PARQUET = (uid, image)
                else:
                    uid, image = TRAIN_PARQUET

        #---
        #'image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'
        if split is not None:
            s  = np.load(DATA_DIR + '/split/%s'%split, allow_pickle=True)
            df = df_loc_by_list(df, 'image_id', s)


        # index    = df.index.values
        # self.df  = df.set_index('image_id', drop=True)
        # self.uid   = uid[index]
        # self.image = image[index]
        # self.num_image = len(self.uid)
        # assert np.all(self.uid == self.df.index)

        df['i'] = df['image_id'].map(lambda x: int(x.split('_')[-1]))
        df = df[['i','image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme']]

        self.uid   = uid
        self.image = image
        self.df = df
        self.num_image = len(self.df)



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

        i, image_id, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme  =  self.df.values[index]
        grapheme = TASK['grapheme'].class_map[grapheme]


        image = self.image[i].copy().reshape(137, 236)
        image = image.astype(np.float32)/255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic, grapheme]

        infor = Struct(
            index    = index,
            image_id = image_id,
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

    label = np.stack(label)

    input = np.stack(input)
    #input = input[...,::-1].copy()
    #input = input.transpose(0,3,1,2)

    #----
    input = torch.from_numpy(input).float()
    input = input.unsqueeze(1) #.repeat(1,3,1,1) # 3 common

    truth = torch.from_numpy(label).long()
    truth = torch.unbind(truth,1)
    return input, truth, infor

#---
# see trorch/utils/data/sampler.py
class BalanceSampler(Sampler):
    def __init__(self, dataset, length):
        self.length = length

        df = dataset.df.reset_index()

        group = []
        grapheme_gb = df.groupby(['grapheme'])
        for k,i in TASK['grapheme'].class_map.items():
            g = grapheme_gb.get_group(k).index
            group.append(list(g))
            assert(len(g)>0)

        self.group=group

    def __iter__(self):
        #l = iter(range(self.num_samples))
        #return l

        # for i in range(self.num_sample):
        #     yield i


        index = []
        n = 0

        is_loop = True
        while is_loop:
            num_class = TASK['grapheme'].num_class #1295
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n+=1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)





    def __len__(self):
        return self.length




##############################################################

def input_to_image(input):
    image = input.data.cpu().numpy()
    image = image.squeeze(1)
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
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1)

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
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=1)

    return image


# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contast(image, alpha=[0,1]):
    beta  = 0
    alpha = random.uniform(*alpha) + 1
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image,0,1)
    return image

def do_random_erase(image, size=[0.1, 0.3]):
    height,width = image.shape

    #get bounding box
    m = image.copy()
    cv2.rectangle(m,(0,0),(height,width),1,5)
    m = image<0.5
    if m.sum()==0: return image

    m = np.where(m)
    y0,y1,x0,x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
    w = x1-x0
    h = y1-y0
    if w*h<10: return image

    ew, eh = np.random.uniform(*size,2)
    ew = int(ew*w)
    eh = int(eh*h)

    ex = np.random.randint(0,w-ew)+x0
    ey = np.random.randint(0,h-eh)+y0

    image[ey:ey+eh, ex:ex+ew] = 1 #0.5
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
            csv     = 'debug_image_data_0.csv',
            split   = None,
            parquet = ['debug_image_data_0.parquet',],
            augment = None,
        )

    if 1:
        dataset = KaggleDataset(
            mode    = 'train',
            csv     = 'train.csv',
            split   = 'random1/valid_small_a_fold0_1000.npy',
            parquet = None,
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

            image = input_to_image(input)
            truth = [t.data.cpu().numpy() for t in truth]

            for b in range(batch_size):
                print('%2d --------------------------- '%(b))
                print('%s'%(str(infor[b])))

                for i,name in enumerate(list(TASK.keys())):
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
        if 0:
            image = do_random_contast(image, alpha=[0,1])
        if 1:
            image = do_random_erase(image, size=[0,1])


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


def run_tune_agument():

    def show_overlay(overlay, S, resize):
        overlay = np.vstack([
            np.hstack(overlay[i,j] for i in range(S))
        for j in range(S)])
        image_show_norm('overlay',overlay, resize=resize)


    #image_file = '/root/share/project/kaggle/2020/grapheme_classification/data/image/debug/Train_6427.png'
    image_file = '/root/share/project/kaggle/2020/grapheme_classification/data/image/debug/Train_6463.png'
    image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)

    gh,gw = int(IMAGE_HEIGHT//5), int(IMAGE_WIDTH//5)
    #image = draw_grid(image, grid_size=[gw,gh], color=[0,0,255], thickness=1)


    S=10
    overlay = np.zeros((S,S,IMAGE_HEIGHT,IMAGE_WIDTH), np.float32)
    image = image.astype(np.float32)/255

    for i in range(S):
        for j in range(S):
            image1 = image.copy()


            # image1 = do_grid_distortion(image1, distort=0.50, num_step = 5)
            # image1 = do_random_crop_rotate_rescale(image1, mode={'rotate': 15,'scale': 0.25,'shift': 0.08})
            # image1 = do_random_contast(image1, alpha=[0,1])
            image1 = do_random_erase(image1, size=[0.1, 0.3])

            overlay[i,j] = image1
            #image_show_norm('image',image)
            #image_show_norm('image1',image1)
            #cv2.waitKey(0)

    image_show_norm('image',image)
    show_overlay(overlay, S, resize=0.5)
    cv2.waitKey(0)


def run_check_sampler():

    dataset = KaggleDataset(
        mode    = 'train',
        csv     = 'train.csv',
        split   = 'balance2/train_b_fold0_184855.npy',
        parquet = None,
        augment = None,
    )

    occurence = np.zeros(1295)
    sampler = BalanceSampler(dataset,3*1295)
    for n,i in enumerate(sampler):
        image, label, infor = dataset[i]
        print('%5d, %5d, label = %s'%(n,i,str(label)))

        occurence[label[-1]]+=1
        # plt.bar(range(1295), occurence, color='green')
        # plt.pause(1)


    print(occurence)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_dataset()
    #run_check_dataloader()
    #run_check_augment()
    #run_tune_agument()

    run_check_sampler()




