#!/usr/bin/python
"""
Needle tracking
"""
from __future__ import division
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt 
import cv2
import pdb
from Tkinter import *
import tkFileDialog
import argparse
import random

from color_classifier import is_metallic

verbose = False
zones = None
image = None
img_width = 0
img_height = 0
p_map = None
num_windows = 0


def main():
	"""
	Main function for needle tracker
	"""
	# arg parsing
	parser = argparse.ArgumentParser(description='Detect needle in images \
		and plot probability distribution.')
	parser.add_argument("-i", "--image", type=str, help="input image filename", default=None, required=False)
	parser.add_argument("-v", "--verbose", help="debug printing enable",
                    action="store_true")
	args = parser.parse_args()
	global verbose 
	global image
	global img_height
	global img_width
	global num_windows
	verbose = args.verbose
	image = None
	filename = None

	# check if file specified
	if args.image:
		filename = args.image
		try:
			image = misc.imread(filename)
			if verbose:
				print filename
		except:
			print 'something wrong with the filename'
	else:
		filename = 'images/left1008.jpeg'
		image = misc.imread(filename)

	if verbose:
		plt.imshow(image)
		plt.show()

	img_width = len(image[0])
	img_height = len(image)

	num_windows = 1e4 # The number of zones of probability must be a perfect square
	global zones 
	zones = np.zeros(((int)(np.sqrt(num_windows)),(int)(np.sqrt(num_windows)))) # set up the zones of distinct probability
	set_random_p()
	plot_pixel_map()


def plot_pixel_map():
	global zones
	global image 
	global img_height
	global img_width
	generate_pixel_map()
	plt.imshow(image)
	plt.imshow(p_map, cmap='Blues',interpolation='nearest', alpha=0.3)
	plt.show();

def generate_pixel_map():
	"""
	generates a map of probabilities that can be overlayed on the image
	"""
	global p_map
	p_map = np.zeros((img_height, img_width))
	for i in range(img_height):
		for j in range(img_width):
			p_map[i,j] = get_pixel_zone(j,i)

def get_pixel_zone(x, y):
	global num_windows
	global img_width
	global img_height
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	return zones[y//tmp2,x//tmp1]


def normalize_map():
	global zones
	zones = zones/zones.sum()


def set_random_p():
	global zones
	global image

	for i in range(zones.shape[0]):
		for j in range(zones.shape[1]):
			zones[i,j] = random.random()
	normalize_map()

if __name__ == '__main__':
	main()