# Import required packages
import argparse
import random
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import argparse
import random
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Set output image size
SIZE = 227

# crops the image around its specified bounding box with 16 pixel padding in all directions
def crop_and_resize(filename, bbox, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    shape = image.size
    x1 = max(bbox[0]-16, 0)
    y1 = max(bbox[1] - 16, 0)
    x2 = min(bbox[2] + 16, shape[0])
    y2 = min(bbox[3] + 16, shape[1])
    new_box = (x1, y1, x2, y2)
    image = image.crop(box = new_box)
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image)

# Define the data directories
project_dir = '~/Documents/Senior/CS230/CS-230-project/stanford-cars/'
data_dir = 'stanford-cars/'
    
# Read in data, select relevant columns
annotations = pd.read_csv(os.path.join(project_dir,'full_annotations.csv'), index_col = 0)
X = annotations[['relative_im_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
y = annotations[['class']]
    
# Train/dev/test split
# 70-15-15 stratified split by class
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
X_dev, X_test, y_dev, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1, stratify = y_val)
    
# List image filenames by train-dev-test split and compile them into a dictionary
train_filenames = list(X_train['relative_im_path'])
dev_filenames = list(X_dev['relative_im_path'])
test_filenames = list(X_test['relative_im_path'])
filenames = {'train': train_filenames,'dev': dev_filenames,'test': test_filenames}

# Preprocess train, dev and test, save X and Y files for each group
for split in ['train', 'dev', 'test']:
    X = []
    Y = []
    for filename in tqdm(filenames[split]):
        full_path = os.path.join(data_dir, filename)
        row = annotations.loc[annotations['relative_im_path']==filename]
        x1 = int(row['bbox_x1'])
        y1 = int(row['bbox_y1'])
        x2 = int(row['bbox_x2'])
        y2 = int(row['bbox_y2'])
        box = (x1, y1, x2, y2)
        image = crop_and_resize(full_path, box, size=SIZE)
        label = int(row['class'])
        X.append(image.flatten())
        Y.append(label)
    X_filename = data_dir + 'X_' + split
    Y_filename = data_dir + 'Y_' + split
    np.save(X_filename, X)
    np.save(Y_filename, Y)

print("Done building dataset")