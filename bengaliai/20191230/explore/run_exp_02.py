from common import *
from kaggle import *
from dataset import *


# DATA_DIR = '/root/share/project/kaggle/2020/grapheme_classification/data'
# TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]


# https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen
def run_ex_10():
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


    uid   = np.load('/media/ssd/data/kaggle/grapheme_classification/uid.npy',allow_pickle=True)
    image = np.load('/media/ssd/data/kaggle/grapheme_classification/image.npy',allow_pickle=True)




    grapheme_root = []
    for c in range(168):
        r = grapheme_root_gb.get_group(c)['image_id'].values
        np.random.shuffle(r)

        num_show = 11

        count=0
        overlay=[]
        for image_id in r:
            i = np.where(uid==image_id)[0][0]
            m = image[i].reshape(137, 236)
            overlay.append(m)
            print(i,c,CLASS_MAP[0][c])
            count+=1
            if count==num_show*num_show:
                break

            #image_show('%i'%i,m)

        overlay = np.vstack([
            np.hstack(overlay[i*num_show:(i+1)*num_show]) for i in range(num_show)
        ])
        image_show('overlay',overlay,0.5)
        cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_ex_10()




