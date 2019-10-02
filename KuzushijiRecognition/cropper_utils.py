import numpy as np
import pandas as pd
import cv2
import os

def make_equal(img, bordertype='constant') :
    '''
    Character crops can have unequal aspect ratio. Instead of stretching, the cropped imaged,
    I will instead pad such that the new image will be a square. Using constant as padding value will use the 
    median pixel value taken from two layers of outermost pixels of the padded side.
    '''
    assert bordertype in ['constant', 'replicate']
    arg = {'top' : 0 , 'bottom' : 0, 'left' : 0, 'right' : 0}
    if bordertype == 'replicate' :
        arg['borderType'] =  cv2.BORDER_REPLICATE
    else :
        arg['borderType'] = cv2.BORDER_CONSTANT
    if img.shape[0] == img.shape[1] :
        return img
    elif img.shape[0] > img.shape[1] :
        border_needed = img.shape[0] - img.shape[1]
        arg['left'] = border_needed//2
        arg['right'] = border_needed - arg['left']
        calc_arr = np.concatenate([img[:,:2,:], img[:,-2:,:]])
        if arg['borderType'] == cv2.BORDER_CONSTANT :
            arg['value'] = np.median(calc_arr , axis = (0,1)).astype(int).tolist()
    elif img.shape[1] > img.shape[0] :
        border_needed = img.shape[1] - img.shape[0]
        arg['top'] = border_needed//2
        arg['bottom'] = border_needed - arg['top']
        calc_arr = np.concatenate([img[:2,:,:], img[-2:,:,:]])
        if arg['borderType'] == cv2.BORDER_CONSTANT :
            arg['value'] = np.median(calc_arr , axis = (0,1)).astype(int).tolist()
    else :
        raise ValueError()
    img = cv2.copyMakeBorder(img, **arg)
    return img

def parse_bounding_box(labels) :
    annot_list = np.array(labels.split(' ')).reshape(-1,5)
    coor_annot = annot_list[:,1:].astype(int)
    coor_annot[:,2] += coor_annot[:,0]
    coor_annot[:,3] += coor_annot[:,1]
    return coor_annot  

def parse_classes(i,row) :
    try :
        len(row['labels'])
    except :
        return
    annot_list = np.array(row['labels'].split(' ')).reshape(-1,5)
    cls_annot = annot_list[:,0].tolist()
    image_ids = []
    for i in range(len(cls_annot)):
        image_ids.append(row['image_id']+ '_' + str(i))
    return pd.DataFrame({'image_id' : image_ids, 'labels' : cls_annot})

def crop_image(img_id, img_dir, bb_annot, crop_dir='./cropped/') :
    img_path = os.path.join(img_dir, img_id  +'.jpg')
    img = cv2.imread(img_path)
    for i in range(len(bb_annot)) :
        left = bb_annot[i,0]
        bottom = bb_annot[i,1]
        right = bb_annot[i,2]
        top = bb_annot[i,3]
        cropped = img[bottom:top,left:right,:]
        cropped = make_equal(cropped)
        cv2.imwrite(os.path.join(crop_dir,img_id + '_' + str(i) + '.jpg'), cropped)

if __name__=='__main__' :
    pass