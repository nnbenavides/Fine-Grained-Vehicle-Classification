# Note: We leveraged https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/build_dataset.py
# to write this script to generate our dataset.

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
SIZE = 224

# Crop images based on their bounding boxes and resize to SIZE x SIZE x 3
def crop_and_resize(filename, bbox, size=SIZE):
	image = Image.open(filename)
	shape = image.size
	x1 = max(bbox[0]-16, 0)
	y1 = max(bbox[1] - 16, 0)
	x2 = min(bbox[2] + 16, shape[0])
	y2 = min(bbox[3] + 16, shape[1])
	new_box = (x1, y1, x2, y2)
	image = image.crop(box = new_box)
	image = image.resize((size, size), Image.BILINEAR)
	return np.array(image)

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

# set seed for reproducibility
np.random.seed(0)

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
		if image.shape != (SIZE, SIZE, 3):
			continue
		label = int(row['class'])
		X.append(image)
		Y.append(label)

	X_filename = data_dir + 'X_' + split + '_' + str(SIZE)
	Y_filename = data_dir + 'Y_' + split + '_' + str(SIZE)
	print('Saving X')
	np.save(X_filename, X)
	print('Saving Y')
	np.save(Y_filename, Y)
	
	print(len(X))
	print(X[0].shape)
	print(len(Y))

print("Done building dataset")