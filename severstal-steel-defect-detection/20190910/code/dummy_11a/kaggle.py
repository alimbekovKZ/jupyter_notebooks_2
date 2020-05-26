from common  import *


# https://www.kaggle.com/iafoss/severstal-fast-ai-256x256-crops-sub
# https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88

DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask


def run_length_encode(mask):
    #possible bug for here
    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        m   = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
    return rle

# https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107053#latest-617549
DUPLICATE=np.array([
    'train_images/6eb8690cd.jpg', 'train_images/a67df9196.jpg',
    'train_images/24e125a16.jpg', 'train_images/4a80680e5.jpg',
    'train_images/a335fc5cc.jpg', 'train_images/fb352c185.jpg',
    'train_images/c35fa49e2.jpg', 'train_images/e4da37c1e.jpg',
    'train_images/877d319fd.jpg', 'train_images/e6042b9a7.jpg',
    'train_images/618f0ff16.jpg', 'train_images/ace59105f.jpg',
    'train_images/ae35b6067.jpg', 'train_images/fdb5ae9d4.jpg',
    'train_images/3de8f5d88.jpg', 'train_images/a5aa4829b.jpg',
    'train_images/3bd0fd84d.jpg', 'train_images/b719010ac.jpg',
    'train_images/24fce7ae0.jpg', 'train_images/edf12f5f1.jpg',
    'train_images/49e374bd3.jpg', 'train_images/6099f39dc.jpg',
    'train_images/9b2ed195e.jpg', 'train_images/c30ecf35c.jpg',
    'train_images/3a7f1857b.jpg', 'train_images/c37633c03.jpg',
    'train_images/8c2a5c8f7.jpg', 'train_images/abedd15e2.jpg',
    'train_images/b46dafae2.jpg', 'train_images/ce5f0cec3.jpg',
    'train_images/5b1c96f09.jpg', 'train_images/e054a983d.jpg',
    'train_images/3088a6a0d.jpg', 'train_images/7f3181e44.jpg',
    'train_images/dc0c6c0de.jpg', 'train_images/e4d9efbaa.jpg',
    'train_images/488c35cf9.jpg', 'train_images/845935465.jpg',
    'train_images/3b168b16e.jpg', 'train_images/c6af2acac.jpg',
    'train_images/05bc27672.jpg', 'train_images/dfefd11c4.jpg',
    'train_images/048d14d3f.jpg', 'train_images/7c8a469a4.jpg',
    'train_images/a1a0111dd.jpg', 'train_images/b30a3e3b6.jpg',
    'train_images/d8be02bfa.jpg', 'train_images/e45010a6a.jpg',
    'train_images/caf49d870.jpg', 'train_images/ef5c1b08e.jpg',
    'train_images/63c219c6f.jpg', 'train_images/b1096a78f.jpg',
    'train_images/76096b17b.jpg', 'train_images/d490180a3.jpg',
    'train_images/bd0e26062.jpg', 'train_images/e7d7c87e2.jpg',
    'train_images/600a81590.jpg', 'train_images/eb5aec756.jpg',
    'train_images/ad5a2ea44.jpg', 'train_images/e9fa75516.jpg',
    'train_images/6afa917f2.jpg', 'train_images/9fb53a74b.jpg',
    'train_images/59931eb56.jpg', 'train_images/e7ced5b76.jpg',
    'train_images/0bfe252d0.jpg', 'train_images/b4d0843ed.jpg',
    'train_images/67fc6eeb8.jpg', 'train_images/c04aa9618.jpg',
    'train_images/741a5c461.jpg', 'train_images/dae3c563a.jpg',
    'train_images/78416c3d0.jpg', 'train_images/e34f68168.jpg',
    'train_images/0d258e4ae.jpg', 'train_images/72322fc23.jpg',
    'train_images/0aafd7471.jpg', 'train_images/461f83c57.jpg',
    'train_images/38a1d7aab.jpg', 'train_images/8866a93f6.jpg',
    'train_images/7c5b834b7.jpg', 'train_images/dea514023.jpg',
    'train_images/32854e5bf.jpg', 'train_images/530227cd2.jpg',
    'train_images/1b7d7eec6.jpg', 'train_images/f801dd10b.jpg',
    'train_images/46ace1c15.jpg', 'train_images/876e74fd6.jpg',
    'train_images/578b43574.jpg', 'train_images/9c5884cdd.jpg',
]).reshape(-1,2).tolist()


def print_submission_csv(df):

    text = ''
    df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
    df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
    pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
    pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
    pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
    pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

    num_image = len(df)//4
    num = len(df)
    pos = (df['Label']==1).sum()
    neg = num-pos

    text += 'compare with LB probing ... \n'
    text += '\t\tnum_image = %5d(1801) \n'%num_image
    text += '\t\tnum  = %5d(7204) \n'%num
    text += '\t\tneg  = %5d(6172)  %0.3f \n'%(neg,neg/num)
    text += '\t\tpos  = %5d(1032)  %0.3f \n'%(pos,pos/num)
    text += '\t\tpos1 = %5d( 128)  %0.3f  %0.3f \n'%(pos1,pos1/num_image,pos1/pos)
    text += '\t\tpos2 = %5d(  43)  %0.3f  %0.3f \n'%(pos2,pos2/num_image,pos2/pos)
    text += '\t\tpos3 = %5d( 741)  %0.3f  %0.3f \n'%(pos3,pos3/num_image,pos3/pos)
    text += '\t\tpos4 = %5d( 120)  %0.3f  %0.3f \n'%(pos4,pos4/num_image,pos4/pos)
    text += ' \n'

    if 1:
        #compare with reference
        pass

    return text



### draw ###################################################################

def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness==1:
        image[contour] = color
    else:
        for y,x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x,y), thickness//2, color, lineType=cv2.LINE_4 )
    return image

def draw_mask_overlay(image, mask, color=(0,0,255), alpha=0.5):
    H,W,C = image.shape
    mask = (mask*alpha).reshape(H,W,1)
    overlay = image.astype(np.float32)
    overlay = np.maximum( overlay, mask*color )
    overlay = np.clip(overlay,0,255)
    overlay = overlay.astype(np.uint8)
    return overlay

def draw_grid(image, grid_size=[32,32], color=[64,64,64], thickness=1):
    H,W,C = image.shape
    dx,dy = grid_size

    for x in range(0,W,dx):
        cv2.line(image,(x,0),(x,H),color, thickness=thickness)
    for y in range(0,H,dy):
        cv2.line(image,(0,y),(W,y),color, thickness=thickness)
    return image


def draw_predict_result(image, truth_mask, truth_label, probability_mask, stack='horizontal', scale=-1):
    color = DEFECT_COLOR

    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    H,W,C   = image.shape
    overlay = image.copy()
    result  = []
    for c in range(4):
        r = np.zeros((H,W,3),np.uint8)

        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            p = cv2.resize(probability_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]
            p = probability_mask[c]

        #r = draw_mask_overlay(r, p, color[c+1], alpha=1)
        r = draw_mask_overlay(r, p, (255,255,255), alpha=1)
        r = draw_contour_overlay(r, t, color[c+1], thickness=2)
        draw_shadow_text(r,'predict%d'%(c+1),(5,30),1,color[c+1],2)
        overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=6)
        result.append(r)

    draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay,] + result
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result

def draw_predict_result_single(image, truth_mask, truth_label, probability_mask, stack='horizontal', scale=-1):
    color = DEFECT_COLOR


    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        p = cv2.resize(probability_mask[0], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        p = probability_mask[0]

    H,W,C   = image.shape
    r = np.zeros((H,W,3),np.uint8)
    r = draw_mask_overlay(r, p, (255,255,255), alpha=1)

    overlay = image.copy()
    for c in range(4):
        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]

        r = draw_contour_overlay(r, t, color[c+1], thickness=4)
        overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=4)


    draw_shadow_text(r,'predict(all)',(5,30),1,(255,255,255),2)
    draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay,r]
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result

def draw_predict_result_32x32(image, truth_mask, truth_label, probability_label):
    color = DEFECT_COLOR
    H,W,C = image.shape

    result  = []
    overlay = image.copy()
    for c in range(4):
        overlay = draw_contour_overlay(overlay, truth_mask[c], color[c+1], thickness=2)

        t = truth_label[c][...,np.newaxis]*color[c+1]
        p = probability_label[c][...,np.newaxis]*[255,255,255]
        t = t.astype(np.uint8)
        p = p.astype(np.uint8)
        r = np.hstack([t,p])

        result.append(r)

    result = np.vstack(result)
    result = cv2.resize(result, dsize=None, fx=32,fy=32, interpolation=cv2.INTER_NEAREST)
    assert(result.shape==(4*H,2*W,3))

    result  = draw_grid(result, grid_size=[32,32], color=[64,64,64], thickness=1)
    overlay = draw_grid(overlay, grid_size=[32,32], color=[255,255,255], thickness=1)


    result = np.vstack([
        np.hstack([overlay, image]),
        result
    ])
    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=3)
    return result


def draw_predict_result_label(image, truth_mask, truth_label, probability_label, stack='horizontal', scale=-1):
    color = DEFECT_COLOR

    if scale >0:
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    H,W,C   = image.shape
    overlay = image.copy()
    for c in range(4):
        if scale >0:
            t = cv2.resize(truth_mask[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            t = truth_mask[c]
        overlay = draw_contour_overlay(overlay, t, color[c+1], thickness=4)


    for c in range(4):
        draw_shadow_text(overlay,'pos%d %0.2f (%d)'%(c+1,probability_label[c],truth_label[c]),(5,(c+1)*24),0.75,color[c+1],1)


    #draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = [image,overlay]
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    result = draw_grid(result, grid_size=[W,H], color=[255,255,255], thickness=1)
    return result



### check ##############################################################

def run_check_rle():

    #https://www.kaggle.com/bigkizd/se-resnext50-89
    def ref_mask2rle(img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


    image = cv2.imread('/root/share/project/kaggle/2019/steel/data/train_images/002fc4e19.jpg',cv2.IMREAD_COLOR)
    value = [
        '002fc4e19.jpg_1','146021 3 146275 10 146529 40 146783 46 147038 52 147292 59 147546 65 147800 70 148055 71 148311 72 148566 73 148822 74 149077 75 149333 76 149588 77 149844 78 150100 78 150357 75 150614 72 150870 70 151127 67 151384 64 151641 59 151897 53 152154 46 152411 22',
        '002fc4e19.jpg_2','145658 7 145901 20 146144 33 146386 47 146629 60 146872 73 147115 86 147364 93 147620 93 147876 93 148132 93 148388 93 148644 93 148900 93 149156 93 149412 93 149668 46',
        '002fc4e19.jpg_3', '',
        '002fc4e19.jpg_4', '',
    ]
    rle = [value[i] for i in range(1,8,2)]


    mask = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])
    print(mask.shape)

    print('**run_length_encode**')
    rle1 = [ run_length_encode(m) for m in mask ]
    print('0',rle1[0])
    print('1',rle1[1])
    print('2',rle1[2])
    print('3',rle1[3])
    assert(rle1==rle)
    print('check ok!!!!')

    print('**ref_mask2rle**')
    rle2 = [ ref_mask2rle(m) for m in mask ]
    print('0',rle2[0])
    print('1',rle2[1])
    print('2',rle2[2])
    print('3',rle2[3])
    assert(rle2==rle)
    print('check ok!!!!')


    exit(0)

    image_show_norm('mask[0]',mask[0],0,1)
    image_show_norm('mask[1]',mask[1],0,1)
    image_show_norm('mask[2]',mask[2],0,1)
    image_show_norm('mask[3]',mask[3],0,1)
    image_show('image',image)

    #---
    mask0 = draw_mask_overlay(image, mask[0],color=(0,0,255))
    image_show('mask0',mask0)
    mask1 = draw_mask_overlay(image, mask[1],color=(0,0,255))
    image_show('mask1',mask1)

    cv2.waitKey(0)


def run_make_split():

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/train_images/*.jpg')
    image_file = ['train_images/'+i.split('/')[-1] for i in image_file]
    print(len(image_file))
    print(image_file[:10])

    random.shuffle(image_file)
    print(image_file[:10])

    #12568
    num_valid = 500
    num_all   = len(image_file)
    num_train = num_all-num_valid

    train=np.array(image_file[num_valid:])
    valid=np.array(image_file[:num_valid])

    raise NotImplementedError
    np.save('/root/share/project/kaggle/2019/steel/data/split/train0_%d.npy'%len(train),train)
    np.save('/root/share/project/kaggle/2019/steel/data/split/valid0_%d.npy'%len(valid),valid)



def run_make_train_split():

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/train_images/*.jpg')
    image_file = ['train_images/'+i.split('/')[-1] for i in image_file]
    print(len(image_file)) #12568
    #print(image_file[:10])

    #without duplicate
    duplicate = np.array(DUPLICATE).reshape(-1).tolist() #88
    non_duplicate = list(set(image_file)-set(duplicate)) #12480
    random.shuffle(non_duplicate)


    #12568
    num_fold  = 2
    num_valid = 500

    for n in range(num_fold):
        valid = non_duplicate[n*num_valid:(n+1)*num_valid]
        train = list(set(image_file)-set(valid))

        print(set(train).intersection(valid))
        np.save('/root/share/project/kaggle/2019/steel/data/split/train_a%d_%d.npy'%(n,len(train)),train)
        np.save('/root/share/project/kaggle/2019/steel/data/split/valid_a%d_%d.npy'%(n,len(valid)),valid)


def run_make_test_split():

    df =  pd.read_csv('/root/share/project/kaggle/2019/steel/data/sample_submission.csv')
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    uid = df['ImageId'].unique().tolist()

    test = ['test_images/'+i for i in uid]
    np.save('/root/share/project/kaggle/2019/steel/data/split/test_%d.npy'%len(test),test)


def run_make_test_split1():

    df =  pd.read_csv('/root/share/project/kaggle/2019/steel/data/sample_submission.csv')
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    uid = df['ImageId'].unique().tolist()

    test = ['test_images/'+i for i in uid]

    #for unsupervsied
    random.shuffle(test)
    num_valid = 500
    valid = test[:500]
    train = test[500:]


    np.save('/root/share/project/kaggle/2019/steel/data/split/test_train_%d.npy'%len(valid),valid)
    np.save('/root/share/project/kaggle/2019/steel/data/split/test_valid_%d.npy'%len(train),train)





def run_make_dummy():

    df = pd.read_csv('/root/share/project/kaggle/2019/steel/data/train.csv')
    df.fillna('', inplace=True)

    image_id =[
        '0007a71bf.jpg',
        '002fc4e19.jpg',
        '008ef3d74.jpg',
        '00ac8372f.jpg',
        '00bc01bfe.jpg', # *
        '00c88fed0.jpg',
        '00ec97699.jpg',
        '012f26693.jpg', # *
        '01cfacf80.jpg',
        '0391d44d6.jpg', # *
        'fff02e9c5.jpg', # *
        'ff6e35e0a.jpg',
        'ff73c8e76.jpg', # *
        'fec86da3c.jpg',
        'fea3da755.jpg',
        'fe2234ba6.jpg', # *
    ]

    image_id =[
        '012f26693.jpg', # *
        '0391d44d6.jpg', # *
        'fff02e9c5.jpg', # *
        'fe2234ba6.jpg', # *
    ]
    for i in image_id:
        print(i)

        rle = [
            df.loc[df['ImageId_ClassId']==i + '_1','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_2','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_3','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread('/root/share/project/kaggle/2019/steel/data/train_images/%s'%(i), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])


        ##---
        step=300
        s = mask.sum(0).sum(0)
        v = [ -s[i: i+640].sum() for i in range(0,1600-640,step) ]
        argsort = np.argsort(v)

        #if 0:
        for k in range(2):
            t = argsort[k]

            print(-v[t])
            x0 = t*step
            x1 = x0+640

            dump_dir = '/root/share/project/kaggle/2019/steel/data/dump'
            os.makedirs(dump_dir+'/256x256/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x256/mask',exist_ok=True)
            os.makedirs(dump_dir+'/256x512/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x512/mask',exist_ok=True)
            os.makedirs(dump_dir+'/256x640/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x640/mask',exist_ok=True)


            if 1:
                cv2.imwrite(dump_dir+'/256x640/image/%s_%d.png'%(i[:-4],k), image[:,x0:x1])
                np.save    (dump_dir+'/256x640/mask/%s_%d.npy'%(i[:-4],k), mask[...,x0:x1])
            if 1:
                cv2.imwrite(dump_dir+'/256x512/image/%s_%d0.png'%(i[:-4],k), image[:,x0:x0+512])
                np.save    (dump_dir+'/256x512/mask/%s_%d0.npy'%(i[:-4],k), mask[...,x0:x0+512])
                cv2.imwrite(dump_dir+'/256x512/image/%s_%d1.png'%(i[:-4],k), image[:,x1-512:x1])
                np.save    (dump_dir+'/256x512/mask/%s_%d1.npy'%(i[:-4],k), mask[...,x1-512:x1])
            if 1:
                cv2.imwrite(dump_dir+'/256x256/image/%s_%d0.png'%(i[:-4],k), image[:,x0:x0+256])
                np.save    (dump_dir+'/256x256/mask/%s_%d0.npy'%(i[:-4],k), mask[...,x0:x0+256])
                cv2.imwrite(dump_dir+'/256x256/image/%s_%d1.png'%(i[:-4],k), image[:,x1-256:x1])
                np.save    (dump_dir+'/256x256/mask/%s_%d1.npy'%(i[:-4],k), mask[...,x1-256:x1])


            #cv2.rectangle(image,(x0,0),(x1,256),(0,0,255),10)
        ##---

        overlay = np.vstack([m for m in mask])

        image_show('image',image,0.5)
        image_show_norm('mask',overlay,0,1,0.5)
        cv2.waitKey(1)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_rle()
    run_make_test_split1()
    #run_make_dummy()

    #run_make_train_split()

