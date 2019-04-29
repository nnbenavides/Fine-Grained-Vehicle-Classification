#!/usr/bin/env python
# coding: utf-8

# In[72]:


import argparse
import random
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# In[73]:


SIZE = 227


# In[74]:


def crop_resize_and_save(filename, output_dir, bbox, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    shape = image.size
    x1 = max(bbox[0]-16, 0)
    y1 = max(bbox[1] - 16, 0)
    x2 = min(bbox[2] + 16, shape[0])
    y2 = min(bbox[3] + 16, shape[1])
    new_box = (x1, y1, x2, y2)
    image = image.crop(box = new_box)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


# In[71]:


if __name__ == '__main__':
    # Define the data directories
    project_dir = '~/Documents/Senior/CS230/CS-230-project/stanford-cars/'
    data_dir = 'stanford-cars/'
    output_dir = 'stanford-cars/car_ims_227/'
    
    # Train/dev/test split
    # 70-15-15 stratified split by class
    annotations = pd.read_csv(os.path.join(project_dir,'full_annotations.csv'), index_col = 0)
    X = annotations[['relative_im_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
    y = annotations[['class']]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
    X_dev, X_test, y_dev, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1, stratify = y_val)
    
    train_filenames = list(X_train['relative_im_path'])
    dev_filenames = list(X_dev['relative_im_path'])
    test_filenames = list(X_test['relative_im_path'])
    
    filenames = {'train': train_filenames,'dev': dev_filenames,'test': test_filenames}
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("Warning: output dir {} already exists".format(output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(output_dir, '{}_cars'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            new_filename = os.path.join(data_dir, filename)
            row = annotations.loc[annotations['relative_im_path']==filename]
            x1 = int(row['bbox_x1'])
            y1 = int(row['bbox_y1'])
            x2 = int(row['bbox_x2'])
            y2 = int(row['bbox_y2'])
            box = (x1, y1, x2, y2)
            crop_resize_and_save(new_filename, output_dir_split, box, size=SIZE)

    print("Done building dataset")

