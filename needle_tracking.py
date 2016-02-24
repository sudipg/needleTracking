#!/usr/bin/python
"""
Needle tracking
"""
from __future__ import division
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cv2
import pdb
from Tkinter import *
import tkFileDialog
import argparse
import random
from skimage import measure, filters, data
from mayavi import mlab
from color_classifier import is_metallic, is_metallic_fast
import pygame
import time

verbose = False
zones = None
image = None
img_width = 0
img_height = 0
p_map = None
num_windows = 0
filename = None
global components

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
	print "starting metal segmentation"
	t0 = time.time()
	add_metal_segmentation_p()
	t1 = time.time()
	print "done metal segmentation in "+str(t1-t0)+" s"
	print "starting edges segmentation"
	add_edge_segmentation_p()
	t2 = time.time()
	print "done edge segmentation in "+str(t2-t1)+" s"
	plot_pixel_map()
	plot_components()


def add_metal_segmentation_p():
	global zones
	global image
	global img_height
	global img_width

	metal_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:2]
	ys = np.r_[0:img_height:2]
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	metal_mask = is_metallic_fast(image)
	for x in xs:
		for y in ys:
			if metal_mask[y,x]:
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
	global components

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
		l = labels==i
		countour_size = np.count_nonzero(l)
		if (countour_size > min_contour and countour_size <= max_contour):
			components.append(l)
	for i in range(len(components)):
		components[i] = (filters.gaussian_filter(components[i].astype('float'), sigma=3)>(components[i].max()/3))*255
	edges = sum(components)
	
	if verbose:
		plt.imshow(edges)
		plt.show()

	edge_zones = np.zeros((zones.shape[0], zones.shape[1]))
	xs = np.r_[0:img_width:2]
	ys = np.r_[0:img_height:2]
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

def add_shape_segmentation():
	pass

############################
#### HELPER FUNCTIONS ######
############################

def plot_pixel_map():
	global zones
	global image 
	global img_height
	global img_width
	generate_pixel_map()
	print "now for 3D!!!"
	X = range(img_width)
	Y = range(img_height)
	print str(p_map)
	m = mlab.surf(X,Y,p_map, warp_scale='auto', opacity=1)
	mlab.colorbar(m, orientation='vertical')
	mlab.axes(m)
	mlab.xlabel('Y')
	mlab.ylabel('X')
	mlab.zlabel('probability')
	mlab.outline(m)
	mlab.show()
	raw_input()
	plt.imshow(image)
	plt.imshow(p_map, cmap='jet',interpolation='nearest', alpha=0.5)
	plt.show()
	

def plot_components():
	global components

	for c in components:
		plt.imshow(c, cmap='gray')
		plt.show()

def generate_pixel_map():
	"""
	generates a map of probabilities that can be overlayed on the image
	"""
	global p_map
	global zones
	p_map = np.zeros((img_height, img_width))
	tmp1 = img_width/np.sqrt(num_windows)
	tmp2 = img_height/np.sqrt(num_windows)
	for i in range(zones.shape[0]):
		for j in range(zones.shape[1]):
			p_map[i*tmp2:(i+1)*tmp2,j*tmp1:(j+1)*tmp1] = zones[i,j]

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
	print zones.sum()


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