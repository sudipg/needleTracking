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
from skimage import measure, filters, data

from color_classifier import is_metallic

verbose = False
zones = None
image = None
img_width = 0
img_height = 0
p_map = None
num_windows = 0
filename = None


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
	global filename
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
	# start off with a uniform probability distribution
	generate_uniform_p()
	add_metal_segmentation_p()
	add_edge_segmentation_p()
	plot_pixel_map()


def add_metal_segmentation_p():
	global zones
	global image
	global img_height
	global img_width

	metal_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:1]
	ys = np.r_[0:img_height:1]
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)

	for x in xs:
		for y in ys:
			if is_metallic(image[y,x]):
				metal_zones[y//tmp2,x//tmp1] += 1
	metal_zones = metal_zones/metal_zones.sum()
	zones = zones*metal_zones
	normalize_map()
	print zones

def add_edge_segmentation_p():
	global zones
	global image
	global img_height
	global img_width
	global filename

	min_contour = img_height/4;
	max_contour = 2*img_height;

	ROI = filters.sobel(data.imread(filename, as_grey=True))
	ROI = ROI*(255/ROI.max())
	binary_ROI = ROI>ROI.max()*0.4
	labels = measure.label(binary_ROI,connectivity=2)
	component_numbers = np.unique(labels)
	components = []
	area = len(ROI)*len(ROI[0])
	for i in component_numbers:
	    if np.count_nonzero(labels == i)>min_contour and np.count_nonzero(labels == i)<=max_contour:
	        components.append((labels == i)*255)
	for i in range(len(components)):
		components[i] = (filters.gaussian_filter(components[i].astype('float'), sigma=3)>(components[i].max()/3))*255
	edges = sum(components)
	
	if verbose:
		plt.imshow(edges)
		plt.show()

	edge_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:1]
	ys = np.r_[0:img_height:1]
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)

	for x in xs:
		for y in ys:
			if edges[y,x]:
				edge_zones[y//tmp2,x//tmp1] += 1
	edge_zones = edge_zones/edge_zones.sum()
	zones = zones*edge_zones
	normalize_map()
	print zones


############################
#### HELPER FUNCTIONS ######
############################

def plot_pixel_map():
	global zones
	global image 
	global img_height
	global img_width
	generate_pixel_map()
	plt.imshow(image)
	plt.imshow(p_map, cmap='seismic',interpolation='nearest', alpha=0.6)
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

def generate_uniform_p():
	global zones
	global image
	zones = np.ones((zones.shape[0], zones.shape[1]));
	normalize_map()

if __name__ == '__main__':
	main()