""" Using images for Image Augmentation """

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K
import cv2

img_dir = "./"
height = 96
width = 96
channel = 3
batch_size = 1

val_dir = 'catz/test'
train_dir = 'catz/train'


image_gen = ImageDataGenerator(
			rotation_range = 15,
			width_shift_range = 0.1,
			height_shift_range = 0.1,
			shear_range = 0.01,
			zoom_range = [0.9, 1.25],
			horizontal_flip = True,
			vertical_flip = False,
			fill_mode = 'reflect',
			data_format = 'channels_last',
			brightness_range = [0.5, 1.5])

def get_images(cat_dirs):
	""" Returns images from path"""
	
	# input_images = np.zeros((batch_size, width, height, 3 * 5))
	# output_images = np.zeros((batch_size, width, height, 3))

	input_images = np.zeros((batch_size, width, height,  3))
	output_images = np.zeros((batch_size, width, height, 3))
	
	for p in cat_dirs:
		#print(f"The path is {p}")
		path_to_save = p
		input_imgs = glob.glob(p + "/cat_[0-5]*")
		output_image = p + "/cat_result.jpg"

		# Augment 10 images per image here
		images = [cv2.imread(im, channel) for im in input_imgs[:1]]
		output_image = np.array(cv2.imread(output_image, channel))
	
	input_images[0] = np.concatenate(images, axis=2)
	output_images[0] = output_image
	print(input_images.shape)
	return (input_images, output_images, path_to_save)

def create_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        g_x = image_gen.flow(255*in_x, in_y, batch_size=in_x.shape[0])
        x, y = next(g_x)

        yield x/255.0, y



import re
if __name__ == "__main__":
	path1 = os.getcwd()+'/catz/test'
	path2 = os.getcwd()+'/catz/train'
	cat_dirs_test = glob.glob(path1 + "/*")
	cat_dirs_train = glob.glob(path2 + "/*")
	print(len(cat_dirs_train))
	print(len(cat_dirs_test))

	

	# Get the Test images filled here on each of the test directory
	for p in cat_dirs_test:
		print(f'path is {p}')
		input_imgs = glob.glob(p + "/cat_[0-5]*")
		output_image = p + "/cat_result.jpg"

		# Augment 10 images per image here, and below we go inside each directory of all the test directories
		#TODO - Put this as a Function
		for idx, image in enumerate(input_imgs):
			batches = 0
			input_images = np.zeros((batch_size, width, height,  3))
			output_images = np.zeros((batch_size, width, height, 3))
	
			m=re.findall(r'cat_([0-9]+).jpg', image)
			images = [cv2.imread(image, channel)]
			
			input_images[0] = np.concatenate(images, axis=2)
			output_images[0] = np.array(cv2.imread(output_image, channel))

			#print(input_images.shape, output_images.shape)
			test_gen = (input_images, output_images)
			prefix = "cat_" + m[0]
			g = image_gen.flow(test_gen, shuffle=True, save_format='jpg', save_prefix=prefix, save_to_dir=p, batch_size=1, seed=42)
			for next_im in g:
				batches += 1
				if batches >= 5:
					break


	# Get the Train images filled here on each of the test directory
	#TODO - Put this as a Function
	for p in cat_dirs_train:
		print(f'path is {p}')
		input_imgs = glob.glob(p + "/cat_[0-5]*")
		output_image = p + "/cat_result.jpg"

		# Augment 10 images per image here, and below we go inside each directory of all the test directories
		for idx, image in enumerate(input_imgs):
			batches = 0
			input_images = np.zeros((batch_size, width, height,  3))
			output_images = np.zeros((batch_size, width, height, 3))
	
			m=re.findall(r'cat_([0-9]+).jpg', image)
			images = [cv2.imread(image, channel)]
			
			input_images[0] = np.concatenate(images, axis=2)
			output_images[0] = np.array(cv2.imread(output_image, channel))

			#print(input_images.shape, output_images.shape)
			test_gen = (input_images, output_images)
			prefix = "cat_" + m[0]
			g = image_gen.flow(test_gen, shuffle=True, save_format='jpg', save_prefix=prefix, save_to_dir=p, batch_size=1, seed=44)
			for next_im in g:
				batches += 1
				if batches >= 5:
					break