from common import *
#from kaggle import *


DATA_DIR = '/root/share/project/kaggle/2020/grapheme_classification/data'
TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]



def run_ex_0():
    csv = 'train.csv'
    df = pd.read_csv(DATA_DIR + '/%s'%csv) #.fillna('')
    image_id = df['image_id'].values

    np.random.shuffle(image_id)
    num = len(image_id)

    num_valid = int(0.10*num)
    train = image_id[num_valid:]
    valid = image_id[:num_valid]
    valid_small = image_id[:1000]

    np.save(DATA_DIR + '/split/train_a_fold0_%d.npy'%(len(train)),train)
    np.save(DATA_DIR + '/split/valid_a_fold0_%d.npy'%(len(valid)),valid)
    np.save(DATA_DIR + '/split/valid_small_a_fold0_%d.npy'%(len(valid_small)),valid_small)


    zz=0

def run_ex_1():
    csv = 'train.csv'
    df = pd.read_csv(DATA_DIR + '/%s'%csv) #.fillna('')
    image_id = df['image_id'].values[:200840//4] #200840

    np.random.shuffle(image_id)
    num = len(image_id)

    num_valid = int(0.10*num)
    train = image_id[num_valid:]
    valid = image_id[:num_valid]
    valid_small = image_id[:1000]

    np.save(DATA_DIR + '/split/train_a_fold0_%d.npy'%(len(train)),train)
    np.save(DATA_DIR + '/split/valid_a_fold0_%d.npy'%(len(valid)),valid)
    np.save(DATA_DIR + '/split/valid_small_a_fold0_%d.npy'%(len(valid_small)),valid_small)



def run_ex_2():
    import pyarrow as pa

    df = pd.read_csv(DATA_DIR + '/train.csv')
    d  = pd.read_parquet(DATA_DIR + '/train_image_data_%d.parquet'%0, engine='pyarrow')

    df = df[:50]
    d  = d[:50]

    #---
    df.to_csv(DATA_DIR + '/debug_image_data_0.csv')

    pyarrow_table = pa.Table.from_pandas(d)
    pa.parquet.write_table(pyarrow_table, DATA_DIR + '/debug_image_data_0.parquet',compression='snappy',)

    #d.to_parquet(DATA_DIR + '/debug_image_data_0.parquet', engine='pyarrow', compression='snappy')




def run_ex_3():

    data_dir = '/root/share/project/kaggle/2020/grapheme_classification/data/image/debug'
    image_file = glob.glob(data_dir+'/*.png')
    image = []


    box=[]
    for f in image_file:
        image = cv2.imread(f, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #m = m.astype(np.float32)/255
        #image.append(m)


        t = np.where(gray < 160)
        y0,y1,x0,x1 = np.min(t[0]), np.max(t[0]), np.min(t[1]), np.max(t[1])
        w = abs(x0-x1)
        h = abs(y0-y1)
        print(w,h)
        box.append((w,h))

        cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),3)




        image_show_norm('image', image)
        cv2.waitKey(1)

    box = np.array(box)
    print(np.mean(box,0))
    print(np.std(box,0))
    print(np.max(box,0))
'''
[107.2109375  82.3984375]
[30.71432187 19.01863389]
[191 136]
'''



#-----------------------------
def print_df(df):
    grapheme_root_count       = df['grapheme_root'].value_counts().sort_index()
    vowel_diacritic_count     = df['vowel_diacritic'].value_counts().sort_index()
    consonant_diacritic_count = df['consonant_diacritic'].value_counts().sort_index()

    print((grapheme_root_count))   # 168
    print((vowel_diacritic_count)) # 11
    print((consonant_diacritic_count)) #7



def run_ex_5():
    csv = 'train.csv'
    df = pd.read_csv(DATA_DIR + '/%s'%csv) #.fillna('')
    all = df['image_id'].values[:200840] #200840
    # ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'],

    grapheme_root_count       = df['grapheme_root'].value_counts().sort_index()
    vowel_diacritic_count     = df['vowel_diacritic'].value_counts().sort_index()
    consonant_diacritic_count = df['consonant_diacritic'].value_counts().sort_index()

    print(len(grapheme_root_count))   # 168
    print(len(vowel_diacritic_count)) # 11
    print(len(consonant_diacritic_count)) #7

    grapheme_root_gb = df.groupby(['grapheme_root'])
    vowel_diacritic_gb = df.groupby(['vowel_diacritic'])
    consonant_diacritic_gb = df.groupby(['consonant_diacritic'])

    grapheme_root = []
    for i in range(168):
        r = grapheme_root_gb.get_group(i)['image_id'].values
        np.random.shuffle(r)
        grapheme_root.append(r)

    vowel_diacritic = []
    for i in range(11):
        r = vowel_diacritic_gb.get_group(i)['image_id'].values
        np.random.shuffle(r)
        vowel_diacritic.append(r)

    consonant_diacritic = []
    for i in range(7):
        r = consonant_diacritic_gb.get_group(i)['image_id'].values
        np.random.shuffle(r)
        consonant_diacritic.append(list(r))

    num_split=3
    for s in range(num_split):

        valid = []
        for image_id in grapheme_root:
            N = len(image_id)
            n = int(0.08*N)
            valid.extend(image_id[i*n:(i+1)*n])
        train = list(set(all) - set(valid))
        pass

        ##----
        #df_valid = df_loc_by_list(df, 'image_id', valid)
        #print_df(df_valid)

        np.save(DATA_DIR + '/split/train_b_fold%d_%d.npy'%(s,len(train)),train)
        np.save(DATA_DIR + '/split/valid_b_fold%d_%d.npy'%(s,len(valid)),valid)

        zz=0


'''

>>> df['consonant_diacritic'].value_counts()
0    125278
2     23465
5     21397
4     21270
1      7424
6      1387
3       619

>>> df['vowel_diacritic'].value_counts()
0     41508
1     36886
7     28723
2     25967
4     18848
3     16152
9     16032
5      5297
6      4336
10     3563
8      3528


>>> df['grapheme_root'].value_counts()
72     5736
64     5596
13     5420
107    5321
23     5149
...
45      144
130     144
158     143
102     141
33      136
73      130

'''


    #

def check_split():

    df_all = pd.read_csv(DATA_DIR + '/train.csv')


    def split_to_df(split):
        uid = np.load(DATA_DIR + '/split/%s'%split, allow_pickle=True)
        df = df_loc_by_list(df_all, 'image_id', uid)

        df_count = df['grapheme_root'].value_counts().sort_index()
        grapheme_root = pd.DataFrame({'index':df_count.index, 'count':df_count.values})
        grapheme_root['percent']=grapheme_root['count']/grapheme_root['count'].sum()
        return grapheme_root

    #--------
    train_split = 'random1/train_a_fold0_180756.npy'
    valid_split = 'random1/valid_a_fold0_20084.npy'

    train_split = 'balance2/train_b_fold0_184855.npy'
    valid_split = 'balance2/valid_b_fold0_15985.npy'


    train_grapheme_root = split_to_df(train_split)
    valid_grapheme_root = split_to_df(valid_split)

    # side-by-side data distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x0 = np.arange(168)*3
    x1 = np.arange(168)*3+1
    y0 = train_grapheme_root['percent'].values
    y1 = valid_grapheme_root['percent'].values

    ax.bar(x0,y0,1,color='b',)
    ax.bar(x1,y1,1,color='r',)
    plt.xticks(np.arange(0,168,10)*3, ['%03d'%i for i in range(0,168,10)], fontsize=8)



    #ax.set_xlabel('Test histogram')
    plt.show()

    exit(0)



def check_cooccurence():

    df_all = pd.read_csv(DATA_DIR + '/train.csv')

    cooccurence = np.zeros((168,11))
    csv = 'train.csv'
    df = pd.read_csv(DATA_DIR + '/%s'%csv) #.fillna('')

    grapheme_root = df['grapheme_root'].values
    vowel_diacritic = df['vowel_diacritic'].values

    num = len(df)
    for n in range(num):
        j = grapheme_root[n]
        i = vowel_diacritic[n]

        cooccurence[j,i]+=1


    plt.matshow(cooccurence.T)
    plt.show()



def run_ex_9():
    fold=2
    train = np.load(DATA_DIR + '/split/balance2/train_b_fold%d_184855.npy'%fold, allow_pickle=True)
    valid = np.load(DATA_DIR + '/split/balance2/valid_b_fold%d_15985.npy'%fold, allow_pickle=True)

    train = set(train)
    valid = set(valid)

    print(len(train))
    print(len(valid))
    print(train.intersection(valid))

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_ex_5()
    run_ex_9()
    #check_cooccurence()




