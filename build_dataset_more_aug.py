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

# crops the image around its specified bounding box with 16 pixel padding in all directions
# Based on aug parameter, either flip the image horizontally, rotate it randomly, or add noise
def crop_resize_and_augment(filename, bbox, aug, size=SIZE):
	image = Image.open(filename)
	shape = image.size
	x1 = max(bbox[0]-16, 0)
	y1 = max(bbox[1] - 16, 0)
	x2 = min(bbox[2] + 16, shape[0])
	y2 = min(bbox[3] + 16, shape[1])
	new_box = (x1, y1, x2, y2)
	image = image.crop(box = new_box)
	image = image.resize((size, size), Image.BILINEAR)

	if aug == 0:
		aug_image = flip_image(image)
	elif aug == 1:
		rotation = np.random.uniform(low = -25.0, high = 25.0)
		aug_image = rotate_image(image, rotation)
	else:
		aug_image = add_random_noise(image)
	return np.array(image), np.array(aug_image)

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

# returns a flipped version of the image
def flip_image(image):
	return image.transpose(Image.FLIP_LEFT_RIGHT)

# rotate an image by a specified amount
def rotate_image(image, angle):
	return image.rotate(angle)

#https://github.com/JaeDukSeo/Python_Basic_Image_Processing/blob/master/4_Add_noise/four.py
# applies random noise to the input image
def add_random_noise(image):
	image = np.array(image)
	return image + 3 * image.std()*np.random.random(image.shape)


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
	X_flip = []
	Y_flip = []
	X_rotate = []
	Y_rotate = []
	X_noise = []
	Y_noise = []

	for filename in tqdm(filenames[split]):
		full_path = os.path.join(data_dir, filename)
		row = annotations.loc[annotations['relative_im_path']==filename]
		x1 = int(row['bbox_x1'])
		y1 = int(row['bbox_y1'])
		x2 = int(row['bbox_x2'])
		y2 = int(row['bbox_y2'])
		box = (x1, y1, x2, y2)

		# generate random number to determine what kind of image augmentation to do
		aug = np.random.randint(low = 0, high = 3)
		if split == 'train':
			image, flipped_image = crop_resize_and_augment(full_path, box, 0, size=SIZE)
			if image.shape != (SIZE, SIZE, 3):
				continue
			label = int(row['class'])
			X.append(image)
			Y.append(label)
			X_flip.append(flipped_image)
			Y_flip.append(label)

			image, rotated_image = crop_resize_and_augment(full_path, box, 1, size = SIZE)
			if image.shape !=  (SIZE, SIZE, 3):
				continue
			X_rotate.append(rotated_image)
			Y_rotate.append(label)

			if aug == 2:
				image, noisy_image = crop_resize_and_augment(full_path, box, 2, size = SIZE)
				if image.shape !=  (SIZE, SIZE, 3):
					continue
				X_noise.append(noisy_image)
				Y_noise.append(label)
		else:
			image = crop_and_resize(full_path, box, size = SIZE)
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
	if split == 'train':
		X_flip_filename = data_dir + 'X_flip_' + split + '_' + str(SIZE)
		Y_flip_filename = data_dir + 'Y_flip_' + split + '_' + str(SIZE)
		print('Saving X_flip')
		np.save(X_flip_filename, X_flip)
		print('Saving Y_flip')
		np.save(Y_flip_filename, Y_flip)

		X_rotate_filename = data_dir + 'X_rotate_' + split + '_' + str(SIZE)
		Y_rotate_filename = data_dir + 'Y_rotate_' + split + '_' + str(SIZE)
		print('Saving X_rotate')
		np.save(X_rotate_filename, X_rotate)
		print('Saving Y_flip')
		np.save(Y_rotate_filename, Y_rotate)

		X_noise_filename = data_dir + 'X_noise_' + split + '_' + str(SIZE)
		Y_noise_filename = data_dir + 'Y_noise_' + split + '_' + str(SIZE)
		print('Saving X_noise')
		np.save(X_noise_filename, X_noise)
		print('Saving Y_noise')
		np.save(Y_noise_filename, Y_noise)
	
		print(len(X))
		print(X[0].shape)
		print(len(Y))

		print(len(X_flip))
		print(X_flip[0].shape)
		print(len(Y_flip))

		print(len(X_rotate))
		print(X_rotate[0].shape)
		print(len(Y_rotate))
		
		print(len(X_noise))
		print(X_noise[0].shape)
		print(len(Y_noise))

print("Done building dataset")