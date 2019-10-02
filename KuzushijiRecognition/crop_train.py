from sklearn.preprocessing import LabelEncoder
from multiprocessing import Process, current_process, Pool
from cropper_utils import *
from tqdm import tqdm

import os

crop_dir = './cropped'
train_img_dir = './train_images'

train_df = pd.read_csv('train.csv')

if not(os.path.isdir(crop_dir)) :
    os.mkdir(crop_dir)

# processes = []

# for i, (_, row) in tqdm(enumerate(train_df.iterrows()), total = len(train_df)):
#     try :
#         len(row['labels'])
#     except :
#         continue
#     bb_annot = parse_bounding_box(row['labels'])
#     process = Process(target=crop_image, args=(row['image_id'],train_img_dir, bb_annot))
#     if ((i+1) % 100) == 0 :
#         print('Processed {} images'.format(i+1))
#     processes.append(process)
    
#     process.start()


for i, (_, row) in tqdm(enumerate(train_df.iterrows()), total = len(train_df)):
    try :
        len(row['labels'])
    except :
        continue
    bb_annot = parse_bounding_box(row['labels'])
    crop_image(row['image_id'],train_img_dir, bb_annot)
    if ((i+1) % 100) == 0 :
        print('Processed {} images'.format(i+1))

with Pool(os.cpu_count()) as p :
    label_list = p.starmap(parse_classes, train_df.iterrows())
#encoder = LabelEncoder()
cropped_df = pd.concat(label_list).reset_index(drop = True)
#cropped_df['label_encoded'] = encoder.fit_transform(cropped_df['labels'])
cropped_df.to_csv('cropped_train.csv', index = False)